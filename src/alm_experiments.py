#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -q install torch torchvision torchaudio tqdm pandas numpy')


# In[2]:


import os, math, time, random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

from tqdm.auto import tqdm

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bootstrap_ci(x: List[float], seed: int = 0, n: int = 5000, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=np.float64)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = []
    for _ in range(n):
        samp = rng.choice(x, size=len(x), replace=True)
        means.append(float(np.mean(samp)))
    means = np.array(means)
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1 - alpha/2))
    return float(np.mean(x)), lo, hi

def paired_permutation_test(x: List[float], y: List[float], seed: int = 0, n: int = 20000):
    """
    H0: mean(x-y) = 0 under random sign flips of paired differences.
    Returns two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    d = x - y
    obs = abs(float(np.mean(d)))
    count = 0
    for _ in range(n):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        stat = abs(float(np.mean(signs * d)))
        if stat >= obs:
            count += 1
    return (count + 1) / (n + 1)


# In[3]:


@dataclass
class DataConfig:
    dataset: str = "cifar10"          # "cifar10" or "cifar100"
    batch_size: int = 128
    num_workers: int = 4
    val_frac: float = 0.1
    data_root: str = "./data"

def make_loaders(data_cfg: DataConfig, seed: int):
    if data_cfg.dataset.lower() == "cifar10":
        num_classes = 10
        ds_cls = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
    elif data_cfg.dataset.lower() == "cifar100":
        num_classes = 100
        ds_cls = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    full_train = ds_cls(root=data_cfg.data_root, train=True, download=True, transform=train_tf)
    test_ds    = ds_cls(root=data_cfg.data_root, train=False, download=True, transform=test_tf)

    n = len(full_train)
    n_val = int(round(data_cfg.val_frac * n))
    n_train = n - n_val

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True,
                              num_workers=data_cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                              num_workers=data_cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=data_cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


# In[4]:


from torchvision.models.resnet import resnet18, resnet34

def make_resnet_cifar(depth: str, num_classes: int):
    depth = depth.lower()
    if depth == "resnet18":
        m = resnet18(num_classes=num_classes)
    elif depth == "resnet34":
        m = resnet34(num_classes=num_classes)
    else:
        raise ValueError("depth must be resnet18 or resnet34")

    # CIFAR stem: 3x3 conv, stride 1, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


# In[5]:


@torch.no_grad()
def _power_iter_sigma(W: torch.Tensor, u: torch.Tensor = None, n_iter: int = 1, eps: float = 1e-12):
    """
    Returns (sigma, u_new).
    W is treated as a matrix.
    """
    if W.ndim > 2:
        Wm = W.flatten(1)   # (out, in*k*k) for conv weights
    else:
        Wm = W

    out_dim = Wm.shape[0]
    if u is None or u.numel() != out_dim:
        u = torch.randn(out_dim, device=W.device, dtype=torch.float32)
    u = u.to(dtype=torch.float32)

    for _ in range(n_iter):
        v = torch.mv(Wm.t(), u)
        v = v / (v.norm() + eps)
        u = torch.mv(Wm, v)
        u = u / (u.norm() + eps)

    sigma = torch.dot(u, torch.mv(Wm, v)).abs()
    sigma = torch.clamp(sigma, min=eps)
    return sigma, u

@torch.no_grad()
def resnet_logK(model: nn.Module, state: Dict[str, Any], power_iter: int = 1):
    """
    Surrogate log Lipschitz upper bound for CIFAR-ResNet:
      logK = log||stem|| + sum_blocks log(1 + K_res(block)) + log||fc||
    with K_res(block) ≈ product of spectral norms along residual path.
    """
    device = next(model.parameters()).device
    logK = torch.zeros((), device=device, dtype=torch.float32)

    # stem conv1
    W = model.conv1.weight
    key = ("stem", id(W))
    sigma, u = _power_iter_sigma(W, state.get(key), n_iter=power_iter)
    state[key] = u
    logK = logK + torch.log(sigma)

    # helper: conv sigma
    def conv_sigma(conv: nn.Conv2d, tag: str):
        W = conv.weight
        key = (tag, id(W))
        sigma, u = _power_iter_sigma(W, state.get(key), n_iter=power_iter)
        state[key] = u
        return sigma

    # each residual block: log(1 + prod(sigmas in residual branch))
    for li, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], start=1):
        for bi, block in enumerate(layer):
            # BasicBlock (2 convs) or Bottleneck (3 convs)
            sigmas = []
            sigmas.append(conv_sigma(block.conv1, f"l{li}b{bi}.conv1"))
            sigmas.append(conv_sigma(block.conv2, f"l{li}b{bi}.conv2"))
            if hasattr(block, "conv3"):  # Bottleneck
                sigmas.append(conv_sigma(block.conv3, f"l{li}b{bi}.conv3"))
            # downsample conv, if present (affects skip path, but for safety we include it into residual path multiplier)
            if block.downsample is not None:
                # typically downsample[0] is conv
                ds0 = block.downsample[0]
                if isinstance(ds0, nn.Conv2d):
                    sigmas.append(conv_sigma(ds0, f"l{li}b{bi}.down"))

            # logK_res = sum log sigma_i
            logK_res = torch.zeros((), device=device, dtype=torch.float32)
            for s in sigmas:
                logK_res = logK_res + torch.log(torch.clamp(s, min=1e-12))

            # logK_block = log(1 + exp(logK_res)) = softplus(logK_res)
            logK = logK + F.softplus(logK_res)

    # fc
    W = model.fc.weight
    key = ("fc", id(W))
    sigma, u = _power_iter_sigma(W, state.get(key), n_iter=power_iter)
    state[key] = u
    logK = logK + torch.log(sigma)

    return logK


# In[6]:


@torch.no_grad()
def alm_update_state(
    logK_value: torch.Tensor,          # scalar tensor
    target_logK: float,
    margin: float,
    st: Dict[str, Any],
    *,
    ema_beta: float = 0.95,
    dual_lr: float = 0.1,              # η_λ
    rho_growth: float = 1.3,           # γ
    rho_min: float = 1e-4,
    rho_max: float = 10.0,
    lam_max: float = 10.0,
    tol_hi: float = 5e-3,
    tol_lo: float = 5e-4,
):
    """
    Constraint: g = logK - (target_logK + margin) <= 0.
    Uses EMA-smoothed g, hinge g_+ = max(0,g), and dual update allowing λ to decrease.
    """
    g = (logK_value - (target_logK + margin)).float()

    g_ema = st.get("g_ema", torch.zeros((), device=g.device, dtype=g.dtype))
    g_ema = ema_beta * g_ema + (1.0 - ema_beta) * g
    st["g_ema"] = g_ema

    g_pos = torch.relu(g_ema)
    st["g_pos"] = g_pos

    lam = float(st.get("lam", 0.0))
    rho = float(st.get("rho", 0.5))

    lam = lam + dual_lr * rho * float(g_ema.item())
    lam = max(0.0, min(lam, lam_max))

    # ρ adaptation based on violation only
    if float(g_pos.item()) > tol_hi:
        rho = min(rho_max, rho * rho_growth)
    elif float(g_pos.item()) < tol_lo:
        rho = max(rho_min, rho / rho_growth)

    st["lam"] = lam
    st["rho"] = rho

    return float(g.item()), float(g_ema.item()), float(g_pos.item()), lam, rho

def alm_penalty(g_pos: torch.Tensor, lam: float, rho: float):
    return (lam * g_pos) + (0.5 * rho * (g_pos ** 2))


# In[7]:


@torch.no_grad()
def alm_update_state(
    logK_value: torch.Tensor,          # scalar tensor
    target_logK: float,
    margin: float,
    st: Dict[str, Any],
    *,
    ema_beta: float = 0.95,
    dual_lr: float = 0.1,              # η_λ
    rho_growth: float = 1.3,           # γ
    rho_min: float = 1e-4,
    rho_max: float = 10.0,
    lam_max: float = 10.0,
    tol_hi: float = 5e-3,
    tol_lo: float = 5e-4,
):
    """
    Constraint: g = logK - (target_logK + margin) <= 0.
    Uses EMA-smoothed g, hinge g_+ = max(0,g), and dual update allowing λ to decrease.
    """
    g = (logK_value - (target_logK + margin)).float()

    g_ema = st.get("g_ema", torch.zeros((), device=g.device, dtype=g.dtype))
    g_ema = ema_beta * g_ema + (1.0 - ema_beta) * g
    st["g_ema"] = g_ema

    g_pos = torch.relu(g_ema)
    st["g_pos"] = g_pos

    lam = float(st.get("lam", 0.0))
    rho = float(st.get("rho", 0.5))

    lam = lam + dual_lr * rho * float(g_ema.item())
    lam = max(0.0, min(lam, lam_max))

    # ρ adaptation based on violation only
    if float(g_pos.item()) > tol_hi:
        rho = min(rho_max, rho * rho_growth)
    elif float(g_pos.item()) < tol_lo:
        rho = max(rho_min, rho / rho_growth)

    st["lam"] = lam
    st["rho"] = rho

    return float(g.item()), float(g_ema.item()), float(g_pos.item()), lam, rho

def alm_penalty(g_pos: torch.Tensor, lam: float, rho: float):
    return (lam * g_pos) + (0.5 * rho * (g_pos ** 2))


# In[8]:


@dataclass
class TrainConfig:
    depth: str = "resnet18"
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.02
    amp: bool = True

    # ALM switches
    use_alm: bool = False
    margin: float = 0.0
    rho0: float = 0.5
    dual_lr: float = 0.1
    rho_growth: float = 1.3
    rho_max: float = 10.0
    lam_max: float = 10.0
    ema_beta: float = 0.95
    power_iter: int = 1

    # target calibration: target_logK = logK_init + budget_delta
    budget_delta: float = 0.5

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += float(ce(logits, y).item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    return loss_sum / total, correct / total

def train_one_run(seed: int, data_cfg: DataConfig, cfg: TrainConfig, device: str = "cuda", log_every: int = 20):
    seed_all(seed)

    train_loader, val_loader, test_loader, num_classes = make_loaders(data_cfg, seed=seed)

    model = make_resnet_cifar(cfg.depth, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.startswith("cuda")))

    # ALM state
    spec_state: Dict[str, Any] = {}
    alm_state: Dict[str, Any] = {"lam": 0.0, "rho": float(cfg.rho0)}

    # Calibrate target logK once per run (weights only, no data)
    with torch.no_grad():
        lk0 = float(resnet_logK(model, spec_state, power_iter=cfg.power_iter).item())
    target_logK = lk0 + float(cfg.budget_delta)

    best_val = -1.0
    best_test_at_best_val = -1.0

    hist_rows = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0

        # epoch aggregates for logging
        last_logK = float("nan")
        last_g = float("nan")
        last_gema = float("nan")
        last_gpos = float("nan")
        last_lam = float("nan")
        last_rho = float("nan")

        for it, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                ce_loss = criterion(logits, y)

                if cfg.use_alm:
                    logK = resnet_logK(model, spec_state, power_iter=cfg.power_iter)
                    g, g_ema, g_pos, lam, rho = alm_update_state(
                        logK, target_logK=target_logK, margin=cfg.margin, st=alm_state,
                        ema_beta=cfg.ema_beta, dual_lr=cfg.dual_lr,
                        rho_growth=cfg.rho_growth, rho_max=cfg.rho_max, lam_max=cfg.lam_max
                    )
                    penalty = alm_penalty(alm_state["g_pos"], lam=alm_state["lam"], rho=alm_state["rho"])
                    loss = ce_loss + penalty

                    last_logK = float(logK.item())
                    last_g, last_gema, last_gpos, last_lam, last_rho = g, g_ema, g_pos, lam, rho
                else:
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
            loss_sum += float(ce_loss.item()) * y.size(0)

        sched.step()

        train_acc = correct / total
        train_ce = loss_sum / total

        val_ce, val_acc = accuracy(model, val_loader, device)
        test_ce, test_acc = accuracy(model, test_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            best_test_at_best_val = test_acc

        row = {
            "seed": seed,
            "method": "ALM" if cfg.use_alm else "AdamW",
            "dataset": data_cfg.dataset,
            "depth": cfg.depth,
            "epoch": ep,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "test_ce": test_ce,
            "logK": last_logK,
            "g": last_g,
            "g_ema": last_gema,
            "g_pos": last_gpos,
            "lam": last_lam,
            "rho": last_rho,
            "target_logK": target_logK
        }
        hist_rows.append(row)

        if ep in (1, 10, 20, 30, 40, 50, 60) or (ep % log_every == 0) or (ep == cfg.epochs):
            tag = "ALM" if cfg.use_alm else "AdamW"
            if cfg.use_alm:
                print(f"[{tag}|{data_cfg.dataset}|{cfg.depth}] ep={ep:03d} "
                      f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                      f"logK={last_logK:.4f} g_ema={last_gema:.6f} lam={last_lam:.3f} rho={last_rho:.3f}")
            else:
                print(f"[{tag}|{data_cfg.dataset}|{cfg.depth}] ep={ep:03d} "
                      f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

    hist_df = pd.DataFrame(hist_rows)

    summary = {
        "seed": seed,
        "method": "ALM" if cfg.use_alm else "AdamW",
        "dataset": data_cfg.dataset,
        "depth": cfg.depth,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "wd": cfg.weight_decay,
        "best_val_acc": float(best_val),
        "test_at_best_val": float(best_test_at_best_val),
        "final_test_acc": float(hist_df[hist_df.epoch == cfg.epochs]["test_acc"].iloc[0]),
        "target_logK": float(target_logK),
        "margin": float(cfg.margin) if cfg.use_alm else float("nan"),
        "rho0": float(cfg.rho0) if cfg.use_alm else float("nan"),
        "budget_delta": float(cfg.budget_delta) if cfg.use_alm else float("nan"),
    }
    return hist_df, summary


# In[9]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

data_cfg = DataConfig(dataset="cifar10", batch_size=128, num_workers=4, val_frac=0.1)
seeds = [1,2,3,4,5]

alm_cfg = TrainConfig(
    depth="resnet18",
    epochs=60,
    lr=3e-4,
    weight_decay=0.02,
    amp=True,

    use_alm=True,
    # IMPORTANT knobs:
    budget_delta=0.5,   # looser than your earlier overly-tight target; raise if accuracy drops
    margin=0.0,
    rho0=0.5,
    dual_lr=0.1,
    rho_growth=1.3,
    rho_max=10.0,
    lam_max=10.0,
    ema_beta=0.95,
    power_iter=1
)

all_hist = []
all_sum = []

for s in seeds:
    print("\n" + "="*80)
    print("ALM RUN seed=", s)
    h, summ = train_one_run(s, data_cfg, alm_cfg, device=device, log_every=20)
    all_hist.append(h)
    all_sum.append(summ)

alm_summary = pd.DataFrame(all_sum)
alm_history = pd.concat(all_hist, ignore_index=True)

display(alm_summary)


# In[ ]:


adam_cfg = TrainConfig(
    depth="resnet18",
    epochs=60,
    lr=3e-4,
    weight_decay=0.02,
    amp=True,
    use_alm=False
)

all_sum_a = []
for s in seeds:
    print("\n" + "="*80)
    print("AdamW RUN seed=", s)
    _, summ = train_one_run(s, data_cfg, adam_cfg, device=device, log_every=20)
    all_sum_a.append(summ)

adam_summary = pd.DataFrame(all_sum_a)
display(adam_summary)


# In[ ]:


def compare_summaries(adam_df: pd.DataFrame, alm_df: pd.DataFrame):
    # Align by seed
    seeds_common = sorted(set(adam_df.seed.tolist()) & set(alm_df.seed.tolist()))
    a = []
    b = []
    for s in seeds_common:
        a.append(float(adam_df[adam_df.seed==s]["test_at_best_val"].iloc[0]))
        b.append(float(alm_df[alm_df.seed==s]["test_at_best_val"].iloc[0]))

    mean_a, lo_a, hi_a = bootstrap_ci(a, seed=0)
    mean_b, lo_b, hi_b = bootstrap_ci(b, seed=1)
    pval = paired_permutation_test(b, a, seed=2)

    print("Paired metric: test_at_best_val (paired by identical seeds)")
    print(f"AdamW mean={mean_a:.4f}  95%CI=[{lo_a:.4f},{hi_a:.4f}]")
    print(f"ALM   mean={mean_b:.4f}  95%CI=[{lo_b:.4f},{hi_b:.4f}]")
    print(f"Paired permutation p-value (ALM vs AdamW): {pval:.6f}")

# If you ran Cell 9:
compare_summaries(adam_summary, alm_summary)


# In[ ]:


get_ipython().system('pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
get_ipython().system('pip -q install numpy pandas tqdm')


# In[10]:


import os, math, random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T
from tqdm.auto import tqdm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# In[11]:


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    batch_size: int = 128
    num_workers: int = 4
    val_frac: float = 0.1
    split_seed: int = 123  # fixed so AdamW vs ALM are paired on same split

def make_loaders(cfg: DataConfig):
    if cfg.dataset.lower() == "cifar10":
        num_classes = 10
        ds_cls = torchvision.datasets.CIFAR10
    elif cfg.dataset.lower() == "cifar100":
        num_classes = 100
        ds_cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    # Standard CIFAR preprocessing
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_full = ds_cls(root="./data", train=True, download=True, transform=tf_train)
    test_set   = ds_cls(root="./data", train=False, download=True, transform=tf_test)

    n = len(train_full)
    idx = np.arange(n)
    rng = np.random.RandomState(cfg.split_seed)
    rng.shuffle(idx)
    n_val = int(cfg.val_frac * n)
    val_idx = idx[:n_val].tolist()
    tr_idx  = idx[n_val:].tolist()

    # For validation, use test transforms (no aug)
    val_set = ds_cls(root="./data", train=True, download=False, transform=tf_test)
    train_set = train_full

    train_loader = DataLoader(Subset(train_set, tr_idx), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(Subset(val_set, val_idx), batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


# In[12]:


def make_resnet_cifar(depth: str, num_classes: int):
    depth = depth.lower()
    if depth == "resnet18":
        m = torchvision.models.resnet18(num_classes=num_classes)
    elif depth == "resnet34":
        m = torchvision.models.resnet34(num_classes=num_classes)
    else:
        raise ValueError("depth must be resnet18 or resnet34")

    # CIFAR stem: 3x3 conv, stride 1, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


# In[13]:


class SpectralTracker:
    """
    Tracks normalized spectral norms with 1-step power iteration and provides
    a stable logK scalar ~ O(1) for CIFAR ResNets.

    We keep u vectors in a separate dict (NOT optimizer.state) to avoid AdamW KeyError.
    """
    def __init__(self, model: nn.Module, eps: float = 1e-12, power_iter: int = 1):
        self.eps = eps
        self.power_iter = power_iter
        self.params: List[torch.nn.Parameter] = []
        self.shapes: List[Tuple[int, ...]] = []
        self.fan_in: List[int] = []
        self.u: Dict[int, torch.Tensor] = {}   # key: id(param)
        self.v: Dict[int, torch.Tensor] = {}
        self._collect(model)

    def _collect(self, model: nn.Module):
        for name, p in model.named_parameters():
            if (p.ndim >= 2) and p.requires_grad:
                # include conv + linear weights; skip biases and BN 1D params
                self.params.append(p)
                self.shapes.append(tuple(p.shape))
                if p.ndim == 2:
                    fin = p.shape[1]
                else:
                    # conv: (out, in, k, k)
                    fin = p.shape[1] * int(np.prod(p.shape[2:]))
                self.fan_in.append(int(fin))

    @torch.no_grad()
    def _reshape2d(self, p: torch.Tensor) -> torch.Tensor:
        if p.ndim == 2:
            return p
        # conv weight -> (out, in*k*k)
        return p.reshape(p.shape[0], -1)

    @torch.no_grad()
    def update_uv_and_logK(self) -> float:
        logs = []
        for p, fin in zip(self.params, self.fan_in):
            W = self._reshape2d(p).detach()
            m, n = W.shape
            key = id(p)

            if key not in self.u:
                # init u,v on same device/dtype as W
                u = torch.randn(m, device=W.device, dtype=torch.float32)
                u = u / (u.norm() + self.eps)
                v = torch.randn(n, device=W.device, dtype=torch.float32)
                v = v / (v.norm() + self.eps)
                self.u[key], self.v[key] = u, v

            u = self.u[key]
            v = self.v[key]
            # power iteration (float32 for stability)
            W32 = W.float()
            for _ in range(self.power_iter):
                v = torch.mv(W32.t(), u)
                v = v / (v.norm() + self.eps)
                u = torch.mv(W32, v)
                u = u / (u.norm() + self.eps)

            sigma = torch.dot(u, torch.mv(W32, v)).abs().clamp_min(self.eps)
            sigma_hat = sigma / math.sqrt(fin)  # normalized
            logs.append(torch.log(sigma_hat + self.eps))

            self.u[key], self.v[key] = u, v

        logK = torch.stack(logs).mean().item()
        return float(logK)

    @torch.no_grad()
    def add_penalty_grads(self, coeff: float):
        """
        Adds coeff * d(logK)/dW to each weight gradient.
        Using fixed (u,v) vectors (no backprop through power iteration).
        """
        if coeff <= 0:
            return

        M = len(self.params)
        for p, fin in zip(self.params, self.fan_in):
            if p.grad is None:
                continue

            W = self._reshape2d(p).detach().float()
            key = id(p)
            u = self.u[key]
            v = self.v[key]

            # sigma ~ u^T W v
            sigma = torch.dot(u, torch.mv(W, v)).abs().clamp_min(self.eps)

            # logK = mean_i log(sigma_i/sqrt(fin_i))
            # d/dW log(sigma/sqrt(fin)) = (1/sigma) * u v^T
            gW = (u[:, None] @ v[None, :]) / sigma
            gW = gW / M  # because mean across layers

            # map back to original shape
            gW = gW.to(device=p.device, dtype=p.grad.dtype)
            if p.ndim == 2:
                p.grad.add_(coeff * gW)
            else:
                p.grad.add_(coeff * gW.reshape_as(p))


# In[14]:


@dataclass
class ALMConfig:
    warmup_epochs: int = 5           # let AdamW learn before constraints kick in
    budget_delta: float = 0.50       # allowed slack above calibrated baseline
    margin: float = 0.0              # can be negative to tighten, but start at 0

    lam0: float = 0.0
    rho0: float = 0.1

    dual_lr: float = 0.02            # soft dual updates (prevents lam blowing up)
    rho_growth: float = 1.3
    rho_shrink: float = 0.9
    rho_max: float = 2.0
    lam_max: float = 3.0

    ema_beta: float = 0.95
    tol: float = 1e-3
    patience: int = 2                # epochs of persistent violation before rho grows

class SoftALMController:
    """
    Controls a single scalar inequality g(W)=logK(W)-B <= 0
    via projected dual ascent and safeguarded rho updates.
    """
    def __init__(self, cfg: ALMConfig):
        self.cfg = cfg
        self.lam = float(cfg.lam0)
        self.rho = float(cfg.rho0)
        self.g_ema = 0.0
        self.B: Optional[float] = None
        self._bad_epochs = 0
        self._calib_logs: List[float] = []

    def observe_logK(self, logK: float, epoch: int):
        # warmup: collect stats for calibration
        if epoch < self.cfg.warmup_epochs:
            self._calib_logs.append(float(logK))
            return

        # on first post-warmup epoch, set budget B from warmup median
        if self.B is None:
            med = float(np.median(self._calib_logs)) if len(self._calib_logs) else float(logK)
            self.B = med + self.cfg.budget_delta + self.cfg.margin

        g = float(logK - self.B)
        self.g_ema = self.cfg.ema_beta * self.g_ema + (1 - self.cfg.ema_beta) * g

    def penalty_coeff(self, epoch: int) -> float:
        if epoch < self.cfg.warmup_epochs or self.B is None:
            return 0.0
        # only penalize when violating (g>0)
        gpos = max(0.0, self.g_ema)
        return max(0.0, self.lam + self.rho * gpos)

    def end_epoch_update(self, epoch: int):
        if epoch < self.cfg.warmup_epochs or self.B is None:
            return

        g = self.g_ema

        # projected dual ascent (soft step size)
        self.lam = float(np.clip(self.lam + self.cfg.dual_lr * self.rho * g, 0.0, self.cfg.lam_max))

        # rho adaptation (safeguarded)
        if g > self.cfg.tol:
            self._bad_epochs += 1
            if self._bad_epochs >= self.cfg.patience:
                self.rho = float(min(self.cfg.rho_max, self.rho * self.cfg.rho_growth))
                self._bad_epochs = 0
        else:
            self._bad_epochs = 0
            self.rho = float(max(self.cfg.rho0, self.rho * self.cfg.rho_shrink))


# In[15]:


@dataclass
class TrainConfig:
    depth: str = "resnet18"
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.02
    amp: bool = True

    use_alm: bool = False
    alm: ALMConfig = ALMConfig()
    constraint_every: int = 20  # steps between logK measurements (cost control)

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / total

def train_one_run(seed: int, data_cfg: DataConfig, cfg: TrainConfig, device: str = "cuda", log_every: int = 10):
    set_seed(seed)
    train_loader, val_loader, test_loader, num_classes = make_loaders(data_cfg)

    model = make_resnet_cifar(cfg.depth, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.startswith("cuda")))

    # Optimizer(s)
    if not cfg.use_alm:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        tracker, ctrl = None, None
        method = "AdamW"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        tracker = SpectralTracker(model, power_iter=1)
        ctrl = SoftALMController(cfg.alm)
        method = "ALM"

    best_val = -1.0
    best_test_at_best_val = -1.0

    global_step = 0
    hist = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        correct, total = 0, 0

        # epoch running logK updates (ALM only)
        last_logK = float("nan")
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # ALM: periodically measure constraint + add penalty grads
            if cfg.use_alm and (global_step % cfg.constraint_every == 0):
                last_logK = tracker.update_uv_and_logK()
                ctrl.observe_logK(last_logK, epoch=ep-1)  # epoch index starting at 0

            if cfg.use_alm:
                coeff = ctrl.penalty_coeff(epoch=ep-1)
                tracker.add_penalty_grads(coeff)

            scaler.step(optimizer)
            scaler.update()

            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
            global_step += 1

        # end epoch eval
        train_acc = correct / total
        val_acc = eval_acc(model, val_loader, device)
        test_acc = eval_acc(model, test_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            best_test_at_best_val = test_acc

        # ALM dual update once per epoch
        if cfg.use_alm:
            ctrl.end_epoch_update(epoch=ep-1)

        if (ep == 1) or (ep % log_every == 0) or (ep == cfg.epochs):
            if not cfg.use_alm:
                print(f"[{method}|{data_cfg.dataset}|{cfg.depth}] ep={ep:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")
            else:
                print(f"[{method}|{data_cfg.dataset}|{cfg.depth}] ep={ep:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                      f"logK={last_logK:.4f} g_ema={ctrl.g_ema:.6f} lam={ctrl.lam:.3f} rho={ctrl.rho:.3f} B={(ctrl.B if ctrl.B is not None else float('nan')):.4f}")

        hist.append({
            "seed": seed, "method": method, "epoch": ep,
            "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
            "best_val_acc": best_val, "test_at_best_val": best_test_at_best_val,
            "logK": last_logK,
            "g_ema": (ctrl.g_ema if cfg.use_alm else np.nan),
            "lam": (ctrl.lam if cfg.use_alm else np.nan),
            "rho": (ctrl.rho if cfg.use_alm else np.nan),
            "B": (ctrl.B if (cfg.use_alm and ctrl.B is not None) else np.nan),
        })

    summ = {
        "seed": seed, "method": method,
        "final_test_acc": float(hist[-1]["test_acc"]),
        "best_val_acc": float(best_val),
        "test_at_best_val": float(best_test_at_best_val),
    }
    return pd.DataFrame(hist), summ


# In[16]:


def bootstrap_ci(x, iters=5000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.array(x, dtype=float)
    n = len(x)
    means = []
    for _ in range(iters):
        samp = x[rng.randint(0, n, size=n)]
        means.append(samp.mean())
    means = np.sort(means)
    return float(x.mean()), float(means[int(0.025*iters)]), float(means[int(0.975*iters)])

def paired_permutation_pvalue(a, b, iters=5000, seed=0):
    # H0: mean(a-b)=0
    rng = np.random.RandomState(seed)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    d = a - b
    obs = abs(d.mean())
    cnt = 0
    for _ in range(iters):
        signs = rng.choice([-1, 1], size=len(d))
        stat = abs((d * signs).mean())
        cnt += (stat >= obs)
    return (cnt + 1) / (iters + 1)

# === CONFIG YOU SHOULD USE (prevents saturation) ===
data_cfg = DataConfig(dataset="cifar10", batch_size=128, num_workers=4, val_frac=0.1, split_seed=123)

adam_cfg = TrainConfig(
    depth="resnet18",
    epochs=60,
    lr=3e-4,
    weight_decay=0.02,
    amp=True,
    use_alm=False
)

alm_cfg = TrainConfig(
    depth="resnet18",
    epochs=60,
    lr=3e-4,
    weight_decay=0.02,
    amp=True,
    use_alm=True,
    alm=ALMConfig(
        warmup_epochs=5,
        budget_delta=0.50,   # start feasible
        margin=0.0,
        lam0=0.0, rho0=0.1,
        dual_lr=0.02,
        rho_growth=1.3, rho_shrink=0.9,
        rho_max=2.0, lam_max=3.0,
        ema_beta=0.95,
        tol=1e-3,
        patience=2
    ),
    constraint_every=20
)

seeds = [1,2,3,4,5]

all_summ = []
for s in seeds:
    print("\n" + "-"*70)
    print("RUN seed=", s, "AdamW")
    _, summA = train_one_run(s, data_cfg, adam_cfg, device=device, log_every=10)
    all_summ.append(summA)

for s in seeds:
    print("\n" + "-"*70)
    print("RUN seed=", s, "ALM")
    _, summB = train_one_run(s, data_cfg, alm_cfg, device=device, log_every=10)
    all_summ.append(summB)

summ_df = pd.DataFrame(all_summ)
display(summ_df)

adam = [float(summ_df[(summ_df.seed==s) & (summ_df.method=="AdamW")]["final_test_acc"].iloc[0]) for s in seeds]
alm  = [float(summ_df[(summ_df.seed==s) & (summ_df.method=="ALM")]["final_test_acc"].iloc[0]) for s in seeds]

ma, lo_a, hi_a = bootstrap_ci(adam, seed=0)
mb, lo_b, hi_b = bootstrap_ci(alm,  seed=1)
pval = paired_permutation_pvalue(np.array(alm), np.array(adam), seed=2)

print("\nFINAL RESULTS (paired by seed)")
print(f"AdamW test_acc: mean={ma:.4f}  95%CI=[{lo_a:.4f},{hi_a:.4f}]")
print(f"ALM   test_acc: mean={mb:.4f}  95%CI=[{lo_b:.4f},{hi_b:.4f}]")
print(f"Paired permutation p-value (ALM vs AdamW): {pval:.6f}")
print(f"Mean diff (ALM-AdamW) = {(np.mean(alm)-np.mean(adam)):.4f}")


# In[17]:


from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    depth: str = "resnet18"
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.02
    amp: bool = True

    use_alm: bool = False
    alm: ALMConfig = field(default_factory=ALMConfig)   # <-- FIX: default_factory
    constraint_every: int = 20  # steps between logK measurements

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / total

def train_one_run(seed: int, data_cfg: DataConfig, cfg: TrainConfig, device: str = "cuda", log_every: int = 10):
    set_seed(seed)
    train_loader, val_loader, test_loader, num_classes = make_loaders(data_cfg)

    model = make_resnet_cifar(cfg.depth, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.startswith("cuda")))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if not cfg.use_alm:
        tracker, ctrl = None, None
        method = "AdamW"
    else:
        tracker = SpectralTracker(model, power_iter=1)
        ctrl = SoftALMController(cfg.alm)
        method = "ALM"

    best_val = -1.0
    best_test_at_best_val = -1.0

    global_step = 0
    hist = []
    last_logK = float("nan")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # ALM: periodically measure constraint + add penalty grads
            if cfg.use_alm and (global_step % cfg.constraint_every == 0):
                last_logK = tracker.update_uv_and_logK()
                ctrl.observe_logK(last_logK, epoch=ep-1)

            if cfg.use_alm:
                coeff = ctrl.penalty_coeff(epoch=ep-1)
                tracker.add_penalty_grads(coeff)

            scaler.step(optimizer)
            scaler.update()

            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
            global_step += 1

        train_acc = correct / total
        val_acc = eval_acc(model, val_loader, device)
        test_acc = eval_acc(model, test_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            best_test_at_best_val = test_acc

        if cfg.use_alm:
            ctrl.end_epoch_update(epoch=ep-1)

        if (ep == 1) or (ep % log_every == 0) or (ep == cfg.epochs):
            if not cfg.use_alm:
                print(f"[{method}|{data_cfg.dataset}|{cfg.depth}] ep={ep:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")
            else:
                print(f"[{method}|{data_cfg.dataset}|{cfg.depth}] ep={ep:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                      f"logK={last_logK:.4f} g_ema={ctrl.g_ema:.6f} lam={ctrl.lam:.3f} rho={ctrl.rho:.3f} B={(ctrl.B if ctrl.B is not None else float('nan')):.4f}")

        hist.append({
            "seed": seed, "method": method, "epoch": ep,
            "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
            "best_val_acc": best_val, "test_at_best_val": best_test_at_best_val,
            "logK": last_logK,
            "g_ema": (ctrl.g_ema if cfg.use_alm else np.nan),
            "lam": (ctrl.lam if cfg.use_alm else np.nan),
            "rho": (ctrl.rho if cfg.use_alm else np.nan),
            "B": (ctrl.B if (cfg.use_alm and ctrl.B is not None) else np.nan),
        })

    summ = {
        "seed": seed, "method": method,
        "final_test_acc": float(hist[-1]["test_acc"]),
        "best_val_acc": float(best_val),
        "test_at_best_val": float(best_test_at_best_val),
    }
    return pd.DataFrame(hist), summ


# In[18]:


def bootstrap_ci(x, iters=5000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.array(x, dtype=float)
    n = len(x)
    means = []
    for _ in range(iters):
        samp = x[rng.randint(0, n, size=n)]
        means.append(samp.mean())
    means = np.sort(means)
    return float(x.mean()), float(means[int(0.025*iters)]), float(means[int(0.975*iters)])

def paired_permutation_pvalue(a, b, iters=5000, seed=0):
    rng = np.random.RandomState(seed)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    d = a - b
    obs = abs(d.mean())
    cnt = 0
    for _ in range(iters):
        signs = rng.choice([-1, 1], size=len(d))
        stat = abs((d * signs).mean())
        cnt += (stat >= obs)
    return (cnt + 1) / (iters + 1)

data_cfg = DataConfig(dataset="cifar10", batch_size=128, num_workers=4, val_frac=0.1, split_seed=123)

adam_cfg = TrainConfig(
    depth="resnet18",
    epochs=60,
    lr=3e-4,
    weight_decay=0.02,
    amp=True,
    use_alm=False
)

alm_cfg = TrainConfig(
    depth="resnet18",
    epochs=60,
    lr=3e-4,
    weight_decay=0.02,
    amp=True,
    use_alm=True,
    alm=ALMConfig(
        warmup_epochs=5,
        budget_delta=0.50,
        margin=0.0,
        lam0=0.0, rho0=0.1,
        dual_lr=0.02,
        rho_growth=1.3, rho_shrink=0.9,
        rho_max=2.0, lam_max=3.0,
        ema_beta=0.95,
        tol=1e-3,
        patience=2
    ),
    constraint_every=20
)

seeds = [1,2,3,4,5]

all_summ = []

for s in seeds:
    print("\n" + "-"*70)
    print("RUN seed=", s, "AdamW")
    _, summA = train_one_run(s, data_cfg, adam_cfg, device=device, log_every=10)
    all_summ.append(summA)

for s in seeds:
    print("\n" + "-"*70)
    print("RUN seed=", s, "ALM")
    _, summB = train_one_run(s, data_cfg, alm_cfg, device=device, log_every=10)
    all_summ.append(summB)

summ_df = pd.DataFrame(all_summ)
display(summ_df)

adam = [float(summ_df[(summ_df.seed==s) & (summ_df.method=="AdamW")]["final_test_acc"].iloc[0]) for s in seeds]
alm  = [float(summ_df[(summ_df.seed==s) & (summ_df.method=="ALM")]["final_test_acc"].iloc[0]) for s in seeds]

ma, lo_a, hi_a = bootstrap_ci(adam, seed=0)
mb, lo_b, hi_b = bootstrap_ci(alm,  seed=1)
pval = paired_permutation_pvalue(np.array(alm), np.array(adam), seed=2)

print("\nFINAL RESULTS (paired by seed)")
print(f"AdamW test_acc: mean={ma:.4f}  95%CI=[{lo_a:.4f},{hi_a:.4f}]")
print(f"ALM   test_acc: mean={mb:.4f}  95%CI=[{lo_b:.4f},{hi_b:.4f}]")
print(f"Paired permutation p-value (ALM vs AdamW): {pval:.6f}")
print(f"Mean diff (ALM-AdamW) = {(np.mean(alm)-np.mean(adam)):.4f}")


# In[8]:


get_ipython().system('pip -q install -U timm scipy pandas matplotlib tqdm')


# In[9]:


import os, time, math, random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Optional: tighter reproducibility (may reduce throughput)
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bootstrap_ci(x, iters=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    means = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        means.append(x[idx].mean())
    means = np.sort(means)
    lo = means[int((alpha/2)*iters)]
    hi = means[int((1-alpha/2)*iters)]
    return float(x.mean()), float(lo), float(hi)

def paired_permutation_test(a, b, iters=10000, seed=0):
    # H0: E[a-b]=0
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    obs = abs(d.mean())
    cnt = 0
    for _ in range(iters):
        signs = rng.choice([-1, 1], size=len(d))
        stat = abs((d * signs).mean())
        cnt += (stat >= obs)
    return (cnt + 1) / (iters + 1)

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# In[10]:


@dataclass
class DataConfig:
    dataset: str = "cifar10"        # "cifar10" or "cifar100"
    batch_size: int = 128
    num_workers: int = 4
    val_frac: float = 0.1
    split_seed: int = 123

    # robustness knobs
    label_noise: float = 0.0        # fraction in [0,1]
    randaugment_N: int = 0          # 0 disables
    randaugment_M: int = 0          # 0 disables

def make_transforms(randaugment_N=0, randaugment_M=0):
    # Standard CIFAR augmentation
    train_tf = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]
    if randaugment_N > 0:
        # Torchvision RandAugment expects PIL images; CIFAR datasets return PIL
        train_tf.append(T.RandAugment(num_ops=randaugment_N, magnitude=randaugment_M))
    train_tf += [T.ToTensor(),
                 T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]
    test_tf = [
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ]
    return T.Compose(train_tf), T.Compose(test_tf)

class LabelNoised(torch.utils.data.Dataset):
    def __init__(self, base_ds, num_classes, noise=0.0, seed=0):
        self.base = base_ds
        self.num_classes = num_classes
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        # build noisy labels once
        self.labels = []
        for i in range(len(base_ds)):
            _, y = base_ds[i]
            if self.noise > 0 and self.rng.random() < self.noise:
                y = int(self.rng.integers(0, num_classes))
            self.labels.append(y)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, self.labels[idx]

def get_dataloaders(cfg: DataConfig):
    train_tf, test_tf = make_transforms(cfg.randaugment_N, cfg.randaugment_M)

    if cfg.dataset.lower() == "cifar10":
        train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
        num_classes = 10
    elif cfg.dataset.lower() == "cifar100":
        train_ds = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=test_tf)
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    # label noise applies only to training set (incl. validation split)
    if cfg.label_noise > 0:
        train_ds = LabelNoised(train_ds, num_classes=num_classes, noise=cfg.label_noise, seed=cfg.split_seed)

    n = len(train_ds)
    n_val = int(cfg.val_frac * n)
    n_tr  = n - n_val
    g = torch.Generator().manual_seed(cfg.split_seed)
    tr_ds, va_ds = random_split(train_ds, [n_tr, n_val], generator=g)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    te_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return tr_loader, va_loader, te_loader, num_classes


# In[11]:


# ---- CIFAR ResNet (He et al. style: 3x3 stem, no maxpool) ----
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)

def resnet18_cifar(num_classes): return CIFARResNet(BasicBlock, [2,2,2,2], num_classes)
def resnet34_cifar(num_classes): return CIFARResNet(BasicBlock, [3,4,6,3], num_classes)

# ---- WideResNet-28-10 (standard CIFAR family) ----
class WRNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, stages[0], 3, 1, 1, bias=False)
        self.block1 = self._make_group(stages[0], stages[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_group(stages[1], stages[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_group(stages[2], stages[3], n, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(stages[3])
        self.fc = nn.Linear(stages[3], num_classes)

    def _make_group(self, in_ch, out_ch, n, stride, drop_rate):
        layers = [WRNBlock(in_ch, out_ch, stride, drop_rate)]
        for _ in range(n-1):
            layers.append(WRNBlock(out_ch, out_ch, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18": return resnet18_cifar(num_classes)
    if name == "resnet34": return resnet34_cifar(num_classes)
    if name == "wrn28_10":  return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, drop_rate=0.0)
    raise ValueError("model must be resnet18, resnet34, wrn28_10")


# In[12]:


class SpectralTracker:
    def __init__(self, model: nn.Module, power_iter: int = 1, eps: float = 1e-12):
        self.power_iter = int(power_iter)
        self.eps = eps
        self.buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.params = []
        for p in model.parameters():
            if p.ndim in (2, 4) and p.requires_grad:
                self.params.append(p)

    def _mat(self, W: torch.Tensor) -> torch.Tensor:
        # conv: (out,in,kh,kw) -> (out, in*kh*kw)
        if W.ndim == 4:
            return W.reshape(W.shape[0], -1)
        return W

    @torch.no_grad()
    def _init_uv(self, Wm: torch.Tensor, key: int, device: str):
        out_dim = Wm.shape[0]
        in_dim  = Wm.shape[1]
        u = torch.randn(out_dim, device=device)
        v = torch.randn(in_dim,  device=device)
        u = u / (u.norm() + self.eps)
        v = v / (v.norm() + self.eps)
        self.buffers[key] = {"u": u, "v": v}

    @torch.no_grad()
    def _power_iter(self, Wm: torch.Tensor, key: int):
        u = self.buffers[key]["u"]
        v = self.buffers[key]["v"]
        for _ in range(self.power_iter):
            v = torch.mv(Wm.t(), u)
            v = v / (v.norm() + self.eps)
            u = torch.mv(Wm, v)
            u = u / (u.norm() + self.eps)
        self.buffers[key]["u"] = u
        self.buffers[key]["v"] = v

    def logK(self, device: str) -> torch.Tensor:
        # returns scalar tensor with grad
        logsigmas = []
        for p in self.params:
            Wm = self._mat(p)
            key = id(p)
            if key not in self.buffers:
                self._init_uv(Wm, key, device=device)
            # update u,v without grad
            self._power_iter(Wm.detach(), key)
            u = self.buffers[key]["u"].detach()
            v = self.buffers[key]["v"].detach()
            sigma = torch.dot(u, torch.mv(Wm, v)).abs() + 1e-12
            logsigmas.append(torch.log(sigma))
        return torch.stack(logsigmas).sum()


# In[13]:


@dataclass
class ALMConfig:
    warmup_epochs: int = 5
    budget_delta: float = 0.50      # slack: B = median(logK_warmup) + log(1+delta)
    margin: float = 0.0             # optional additive slack in log space
    lam0: float = 0.0
    rho0: float = 0.1
    dual_lr: float = 1.0            # multiplier on lambda update
    rho_growth: float = 1.4
    rho_shrink: float = 0.9
    rho_max: float = 2.0
    lam_max: float = 5.0
    ema_beta: float = 0.95
    tol: float = 1e-3
    patience: int = 2

@dataclass
class TrainConfig:
    model: str = "resnet18"
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.02
    amp: bool = True

    method: str = "adamw"           # "adamw", "sgd", "alm_adamw", "alm_sgd"
    momentum: float = 0.9           # for SGD
    nesterov: bool = True
    warmup_epochs: int = 5
    label_smoothing: float = 0.0

    # ALM
    alm: ALMConfig = field(default_factory=ALMConfig)
    constraint_every: int = 20      # steps frequency for applying ALM penalty
    power_iter: int = 1


# In[14]:


def make_optimizer(model, cfg: TrainConfig):
    if cfg.method in ("adamw", "alm_adamw"):
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    if cfg.method in ("sgd", "alm_sgd"):
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                               momentum=cfg.momentum, nesterov=cfg.nesterov)
    raise ValueError("unknown method")

def make_scheduler(optimizer, total_epochs, warmup_epochs):
    # Cosine decay to ~0, with linear warmup
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        t = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# In[15]:


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    tot, corr, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        corr += (pred == y).sum().item()
        tot += y.numel()
    return corr / tot, loss_sum / tot

def train_one_run(seed: int, data_cfg: DataConfig, cfg: TrainConfig, device="cuda", log_every=10):
    seed_all(seed)

    tr_loader, va_loader, te_loader, num_classes = get_dataloaders(data_cfg)
    model = build_model(cfg.model, num_classes).to(device)

    opt = make_optimizer(model, cfg)
    sch = make_scheduler(opt, total_epochs=cfg.epochs, warmup_epochs=cfg.warmup_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.startswith("cuda")))

    # loss fn (optional label smoothing)
    def ce_loss(logits, y):
        if cfg.label_smoothing > 0:
            return F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
        return F.cross_entropy(logits, y)

    # ALM state
    use_alm = cfg.method.startswith("alm_")
    tracker = SpectralTracker(model, power_iter=cfg.power_iter) if use_alm else None

    lam = float(cfg.alm.lam0)
    rho = float(cfg.alm.rho0)
    g_ema = 0.0
    B = None
    warmup_logKs = []
    bad_epochs = 0

    step = 0
    best_val = -1.0
    best_state = None
    best_test_at_val = None

    rows = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        tot, corr = 0, 0
        loss_sum = 0.0

        for x, y in tr_loader:
            step += 1
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                base_loss = ce_loss(logits, y)

                # default: no ALM penalty this step
                aug_loss = base_loss
                logK_val = None
                g_pos = None

                if use_alm and (step % cfg.constraint_every == 0):
                    logK_val = tracker.logK(device=device)

                    if ep <= cfg.alm.warmup_epochs:
                        warmup_logKs.append(float(logK_val.detach().cpu()))
                    else:
                        if B is None:
                            # Calibrate B from warmup distribution
                            med = float(np.median(warmup_logKs)) if len(warmup_logKs) else float(logK_val.detach().cpu())
                            # slack in log-space: log(1+delta) + margin
                            B = med + math.log(1.0 + float(cfg.alm.budget_delta)) + float(cfg.alm.margin)

                        g = logK_val - float(B)
                        g_pos = torch.clamp(g, min=0.0)   # [g]_+
                        # AL objective
                        aug_loss = base_loss + (lam * g_pos) + 0.5 * rho * (g_pos ** 2)

            scaler.scale(aug_loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            pred = logits.argmax(dim=1)
            corr += (pred == y).sum().item()
            tot += y.numel()
            loss_sum += float(base_loss.detach().cpu()) * y.numel()

            # Track constraint EMA and update dual on-the-fly (stable, bounded)
            if use_alm and (step % cfg.constraint_every == 0) and (ep > cfg.alm.warmup_epochs) and (B is not None):
                gp = float(g_pos.detach().cpu())
                g_ema = cfg.alm.ema_beta * g_ema + (1.0 - cfg.alm.ema_beta) * gp

        # Epoch end eval
        train_acc = corr / tot
        val_acc, val_loss = eval_loop(model, va_loader, device)
        test_acc, test_loss = eval_loop(model, te_loader, device)

        # Dual / penalty update once per epoch (KKT-ish outer loop)
        if use_alm and (ep > cfg.alm.warmup_epochs) and (B is not None):
            # Lambda update (project to >=0)
            lam = max(0.0, min(cfg.alm.lam_max, lam + cfg.alm.dual_lr * rho * g_ema))

            # Adaptive rho: if violation not decreasing, grow; else shrink mildly
            if g_ema > cfg.alm.tol:
                bad_epochs += 1
            else:
                bad_epochs = 0

            if bad_epochs >= cfg.alm.patience:
                rho = min(cfg.alm.rho_max, rho * cfg.alm.rho_growth)
                bad_epochs = 0
            else:
                rho = max(1e-8, rho * cfg.alm.rho_shrink)

        sch.step()
        dt = time.time() - t0

        # checkpoint by val
        if val_acc > best_val:
            best_val = val_acc
            best_test_at_val = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Logging
        row = dict(
            seed=seed, method=cfg.method, dataset=data_cfg.dataset, model=cfg.model,
            epoch=ep, train_acc=train_acc, val_acc=val_acc, test_acc=test_acc,
            lr=float(opt.param_groups[0]["lr"]),
            logK=float(logK_val.detach().cpu()) if logK_val is not None else np.nan,
            g_ema=float(g_ema) if use_alm else np.nan,
            lam=float(lam) if use_alm else np.nan,
            rho=float(rho) if use_alm else np.nan,
            B=float(B) if (B is not None) else np.nan,
            seconds=dt
        )
        rows.append(row)

        if ep == 1 or ep % log_every == 0 or ep == cfg.epochs:
            if use_alm:
                print(f"[{cfg.method.upper()}|{data_cfg.dataset}|{cfg.model}] ep={ep:03d} "
                      f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                      f"logK={row['logK']:.4f} g_ema={row['g_ema']:.6f} lam={lam:.3f} rho={rho:.3f}")
            else:
                print(f"[{cfg.method.upper()}|{data_cfg.dataset}|{cfg.model}] ep={ep:03d} "
                      f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

    # restore best-by-val for reporting
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_test_acc, _ = eval_loop(model, te_loader, device)

    hist = pd.DataFrame(rows)
    summary = dict(
        seed=seed, method=cfg.method, dataset=data_cfg.dataset, model=cfg.model,
        final_test_acc=float(final_test_acc),
        best_val_acc=float(best_val),
        test_at_best_val=float(best_test_at_val if best_test_at_val is not None else final_test_acc)
    )
    return hist, summary


# In[16]:


@dataclass
class Sweep:
    seeds: List[int] = field(default_factory=lambda: [1,2,3,4,5])
    epochs: int = 200

    # Baseline grids (small but meaningful)
    adamw_lr: Tuple[float,...] = (3e-4, 5e-4, 8e-4)
    adamw_wd: Tuple[float,...] = (0.02, 0.05, 0.1)

    sgd_lr: Tuple[float,...] = (0.05, 0.1, 0.2)
    sgd_wd: Tuple[float,...] = (5e-4, 1e-3)

    # ALM knobs (keep tight; ablate later)
    budget_delta: Tuple[float,...] = (0.2, 0.5)
    rho0: Tuple[float,...] = (0.05, 0.1)
    dual_lr: Tuple[float,...] = (0.5, 1.0)

def tune_one_seed(data_cfg, model_name, base_method, sweep: Sweep, device):
    # quick tuning: fewer epochs, single seed for each candidate
    tune_epochs = 60
    seed = 0

    candidates = []
    if base_method == "adamw":
        for lr in sweep.adamw_lr:
            for wd in sweep.adamw_wd:
                cfg = TrainConfig(model=model_name, epochs=tune_epochs, lr=lr, weight_decay=wd, method="adamw")
                _, summ = train_one_run(seed, data_cfg, cfg, device=device, log_every=20)
                candidates.append((summ["best_val_acc"], dict(lr=lr, wd=wd)))
    elif base_method == "sgd":
        for lr in sweep.sgd_lr:
            for wd in sweep.sgd_wd:
                cfg = TrainConfig(model=model_name, epochs=tune_epochs, lr=lr, weight_decay=wd, method="sgd")
                _, summ = train_one_run(seed, data_cfg, cfg, device=device, log_every=20)
                candidates.append((summ["best_val_acc"], dict(lr=lr, wd=wd)))
    else:
        raise ValueError("base_method must be adamw or sgd")

    best = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    print("TUNE BEST", base_method, ":", best)
    return best

def tune_alm_one_seed(data_cfg, model_name, base_method, sweep: Sweep, device):
    tune_epochs = 60
    seed = 0

    base_lr_wd = tune_one_seed(data_cfg, model_name, base_method, sweep, device)

    candidates = []
    for bd in sweep.budget_delta:
        for rho0 in sweep.rho0:
            for dlr in sweep.dual_lr:
                alm = ALMConfig(
                    warmup_epochs=5,
                    budget_delta=bd,
                    margin=0.0,
                    lam0=0.0, rho0=rho0,
                    dual_lr=dlr,
                    rho_growth=1.4, rho_shrink=0.9,
                    rho_max=2.0, lam_max=5.0,
                    ema_beta=0.95, tol=1e-3, patience=2
                )
                method = "alm_adamw" if base_method=="adamw" else "alm_sgd"
                cfg = TrainConfig(
                    model=model_name, epochs=tune_epochs,
                    lr=base_lr_wd["lr"], weight_decay=base_lr_wd["wd"],
                    method=method, alm=alm,
                    constraint_every=20, power_iter=1
                )
                _, summ = train_one_run(seed, data_cfg, cfg, device=device, log_every=20)
                candidates.append((summ["best_val_acc"], dict(budget_delta=bd, rho0=rho0, dual_lr=dlr)))

    best_alm = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    print("TUNE BEST ALM", base_method, ":", best_alm)
    return base_lr_wd, best_alm

def final_multi_seed(data_cfg, model_name, method_cfgs: Dict[str, TrainConfig], sweep: Sweep, device):
    all_hist = []
    all_summ = []
    for seed in sweep.seeds:
        for tag, cfg in method_cfgs.items():
            print("\n" + "-"*70)
            print("RUN seed=", seed, tag)
            cfg2 = cfg
            cfg2.epochs = sweep.epochs
            hist, summ = train_one_run(seed, data_cfg, cfg2, device=device, log_every=10)
            summ["tag"] = tag
            all_hist.append(hist)
            all_summ.append(summ)
    return pd.concat(all_hist, ignore_index=True), pd.DataFrame(all_summ)

def paired_stats(summ_df: pd.DataFrame, a_tag: str, b_tag: str):
    # pair by seed
    a = []
    b = []
    for s in sorted(summ_df["seed"].unique()):
        a.append(float(summ_df[(summ_df.seed==s) & (summ_df.tag==a_tag)]["final_test_acc"].iloc[0]))
        b.append(float(summ_df[(summ_df.seed==s) & (summ_df.tag==b_tag)]["final_test_acc"].iloc[0]))
    mean_a, lo_a, hi_a = bootstrap_ci(a, seed=0)
    mean_b, lo_b, hi_b = bootstrap_ci(b, seed=1)
    pval = paired_permutation_test(b, a, seed=2)  # H1: b != a (two-sided)
    diff = float(np.mean(np.array(b) - np.array(a)))
    return dict(
        a_tag=a_tag, b_tag=b_tag,
        mean_a=mean_a, ci_a=(lo_a,hi_a),
        mean_b=mean_b, ci_b=(lo_b,hi_b),
        mean_diff=diff, pval=pval,
        a_list=a, b_list=b
    )

def plot_final_bars(summ_df, title, outpath=None):
    # mean test acc by tag
    tags = sorted(summ_df["tag"].unique())
    means = []
    cis = []
    for t in tags:
        vals = summ_df[summ_df.tag==t]["final_test_acc"].astype(float).values
        m, lo, hi = bootstrap_ci(vals, seed=0)
        means.append(m)
        cis.append((m-lo, hi-m))
    yerr = np.array(cis).T
    plt.figure()
    plt.bar(tags, means, yerr=yerr, capsize=4)
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()


# In[17]:


sweep = Sweep(seeds=[1,2,3,4,5], epochs=200)

# ---- Define experiment suite ----
suite = [
    # Breadth
    dict(name="C10_R18",  data=DataConfig(dataset="cifar10", batch_size=128, val_frac=0.1), model="resnet18"),
    dict(name="C10_R34",  data=DataConfig(dataset="cifar10", batch_size=128, val_frac=0.1), model="resnet34"),
    dict(name="C10_WRN",  data=DataConfig(dataset="cifar10", batch_size=128, val_frac=0.1), model="wrn28_10"),
    dict(name="C100_R18", data=DataConfig(dataset="cifar100", batch_size=128, val_frac=0.1), model="resnet18"),
    dict(name="C100_R34", data=DataConfig(dataset="cifar100", batch_size=128, val_frac=0.1), model="resnet34"),

    # Robustness: label noise
    dict(name="C10_R18_noise20", data=DataConfig(dataset="cifar10", batch_size=128, val_frac=0.1, label_noise=0.2), model="resnet18"),

    # Robustness: stronger augmentation
    dict(name="C10_R18_RA", data=DataConfig(dataset="cifar10", batch_size=128, val_frac=0.1, randaugment_N=2, randaugment_M=9), model="resnet18"),

    # Robustness: batch size shift (harder for some optimizers)
    dict(name="C10_R18_bs512", data=DataConfig(dataset="cifar10", batch_size=512, val_frac=0.1), model="resnet18"),
]

all_stats = []
all_summaries = []
all_histories = []

for ex in suite:
    print("\n" + "="*90)
    print("EXPERIMENT:", ex["name"], "|", ex["data"].dataset, "|", ex["model"])

    # Tune AdamW + ALM(AdamW), and SGD + ALM(SGD)
    best_adamw = tune_one_seed(ex["data"], ex["model"], "adamw", sweep, device)
    best_sgd   = tune_one_seed(ex["data"], ex["model"], "sgd",   sweep, device)

    base_lr_wd_adamw, best_alm_adamw = tune_alm_one_seed(ex["data"], ex["model"], "adamw", sweep, device)
    base_lr_wd_sgd,   best_alm_sgd   = tune_alm_one_seed(ex["data"], ex["model"], "sgd",   sweep, device)

    # Build final configs
    cfg_adamw = TrainConfig(model=ex["model"], method="adamw", lr=best_adamw["lr"], weight_decay=best_adamw["wd"])
    cfg_sgd   = TrainConfig(model=ex["model"], method="sgd",   lr=best_sgd["lr"],   weight_decay=best_sgd["wd"])

    cfg_alm_adamw = TrainConfig(
        model=ex["model"], method="alm_adamw",
        lr=base_lr_wd_adamw["lr"], weight_decay=base_lr_wd_adamw["wd"],
        alm=ALMConfig(budget_delta=best_alm_adamw["budget_delta"], rho0=best_alm_adamw["rho0"], dual_lr=best_alm_adamw["dual_lr"])
    )
    cfg_alm_sgd = TrainConfig(
        model=ex["model"], method="alm_sgd",
        lr=base_lr_wd_sgd["lr"], weight_decay=base_lr_wd_sgd["wd"],
        alm=ALMConfig(budget_delta=best_alm_sgd["budget_delta"], rho0=best_alm_sgd["rho0"], dual_lr=best_alm_sgd["dual_lr"])
    )

    method_cfgs = {
        "AdamW": cfg_adamw,
        "SGD": cfg_sgd,
        "ALM-AdamW": cfg_alm_adamw,
        "ALM-SGD": cfg_alm_sgd,
    }

    hist_df, summ_df = final_multi_seed(ex["data"], ex["model"], method_cfgs, sweep, device)
    summ_df["experiment"] = ex["name"]
    hist_df["experiment"] = ex["name"]

    all_summaries.append(summ_df)
    all_histories.append(hist_df)

    # Stats vs strong baselines
    st1 = paired_stats(summ_df, "AdamW", "ALM-AdamW")
    st2 = paired_stats(summ_df, "SGD",   "ALM-SGD")
    st1["experiment"] = ex["name"]
    st2["experiment"] = ex["name"]
    all_stats += [st1, st2]

    print("\nPAIRED RESULTS:", ex["name"])
    print(f"AdamW mean={st1['mean_a']:.4f}  ALM-AdamW mean={st1['mean_b']:.4f}  diff={st1['mean_diff']:.4f}  p={st1['pval']:.6f}")
    print(f"SGD   mean={st2['mean_a']:.4f}  ALM-SGD   mean={st2['mean_b']:.4f}  diff={st2['mean_diff']:.4f}  p={st2['pval']:.6f}")

    os.makedirs("figs", exist_ok=True)
    plot_final_bars(summ_df, title=f"{ex['name']} final test acc (mean±CI)", outpath=f"figs/{ex['name']}_final.png")

final_summ = pd.concat(all_summaries, ignore_index=True)
final_hist = pd.concat(all_histories, ignore_index=True)
final_stats = pd.DataFrame(all_stats)

display(final_stats)
final_summ.to_csv("results_summary.csv", index=False)
final_stats.to_csv("results_stats.csv", index=False)
print("Saved: results_summary.csv, results_stats.csv, and figs/*.png")


# In[18]:


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Note: full determinism on CUDA can require CUBLAS_WORKSPACE_CONFIG; we do not force it.

def bootstrap_ci(xs: List[float], seed: int = 0, n: int = 4000, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) == 0:
        return (np.nan, np.nan, np.nan)
    means = []
    for _ in range(n):
        sample = rng.choice(xs, size=len(xs), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = np.quantile(means, alpha/2)
    hi = np.quantile(means, 1-alpha/2)
    return float(xs.mean()), float(lo), float(hi)

def paired_permutation_test(a: List[float], b: List[float], seed: int = 0, n: int = 20000):
    """
    H0: E[a-b]=0. Returns two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    obs = abs(d.mean())
    cnt = 0
    for _ in range(n):
        signs = rng.choice([-1, 1], size=len(d))
        perm = abs((d * signs).mean())
        if perm >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


# In[19]:


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

@dataclass
class DataConfig:
    dataset: str = "cifar10"    # "cifar10" or "cifar100"
    batch_size: int = 128
    num_workers: int = 4
    val_frac: float = 0.1
    data_dir: str = "./data"
    split_seed: int = 123

def make_loaders(cfg: DataConfig):
    if cfg.dataset.lower() == "cifar10":
        ds_cls = torchvision.datasets.CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        num_classes = 10
    elif cfg.dataset.lower() == "cifar100":
        ds_cls = torchvision.datasets.CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    full_train = ds_cls(root=cfg.data_dir, train=True, download=True, transform=train_tf)
    test_set   = ds_cls(root=cfg.data_dir, train=False, download=True, transform=test_tf)

    n = len(full_train)
    n_val = int(round(cfg.val_frac * n))
    n_tr  = n - n_val

    g = torch.Generator()
    g.manual_seed(cfg.split_seed)
    train_set, val_set = torch.utils.data.random_split(full_train, [n_tr, n_val], generator=g)

    # IMPORTANT: val set should not use augmentation; swap transform
    val_set.dataset = ds_cls(root=cfg.data_dir, train=True, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0)
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0)
    )
    return train_loader, val_loader, test_loader, num_classes


# In[20]:


def make_cifar_resnet(depth: str, num_classes: int):
    depth = depth.lower()
    if depth == "resnet18":
        m = torchvision.models.resnet18(num_classes=num_classes)
    elif depth == "resnet34":
        m = torchvision.models.resnet34(num_classes=num_classes)
    else:
        raise ValueError("depth must be resnet18 or resnet34")

    # CIFAR tweaks: 3x3 conv, stride1, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def make_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 5):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# In[21]:


@dataclass
class ALMConfig:
    warmup_epochs: int = 5

    # Budget calibration: B = logK_at_warmup_end + budget_delta
    budget_delta: float = 0.50
    margin: float = 0.0  # optional extra slack; budget_delta already gives feasibility

    # Dual / penalty dynamics
    lam0: float = 0.0
    rho0: float = 0.1
    dual_lr: float = 0.02

    rho_growth: float = 1.6
    rho_shrink: float = 0.9
    rho_max: float = 2.0
    rho_min: float = 0.05

    lam_max: float = 3.0

    # EMA on violation (stabilizes noisy stochastic constraint eval)
    ema_beta: float = 0.95
    tol: float = 1e-3
    patience: int = 2

    # Constraint evaluation frequency (steps)
    constraint_every: int = 20

    # Which layers are constrained: "all" or "last"
    which: str = "all"

@dataclass
class TrainConfig:
    depth: str = "resnet18"
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 0.02

    method: str = "adamw"  # "adamw" or "sgd"
    momentum: float = 0.9  # only for sgd

    amp: bool = False
    label_smoothing: float = 0.0

    # ALM switch + parameters
    use_alm: bool = False
    alm: ALMConfig = field(default_factory=ALMConfig)

    grad_clip: float = 1.0


# In[22]:


def iter_constrained_weights(model: nn.Module, which: str = "all"):
    """
    Yields (name, weight_tensor, kind) where kind in {"conv","linear"}.
    which="all": all Conv2d + Linear weights
    which="last": only final fc weight
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, "weight") and m.weight is not None:
            if which == "last":
                continue
            yield f"{name}.weight", m.weight, "conv"
        if isinstance(m, nn.Linear) and hasattr(m, "weight") and m.weight is not None:
            if which == "last" and name != "fc":
                continue
            yield f"{name}.weight", m.weight, "linear"

def power_iter_sigma(Wmat: torch.Tensor, u: torch.Tensor, iters: int = 1):
    """
    One-step (or few-step) power iteration to estimate top singular value.
    Returns (sigma, u_new, v_new). Uses no_grad updates for u/v stability.
    """
    with torch.no_grad():
        for _ in range(iters):
            v = torch.mv(Wmat.t(), u)
            v = v / (v.norm() + 1e-12)
            u = torch.mv(Wmat, v)
            u = u / (u.norm() + 1e-12)
        sigma = torch.dot(u, torch.mv(Wmat, v)).abs().clamp(min=1e-12)
    return sigma, u, v

class SpectralCache:
    def __init__(self):
        self.store: Dict[int, Dict[str, Any]] = {}
        self.last_logK: float = float("nan")
        self.last_B: float = float("nan")
        self.last_sigmas: Dict[int, float] = {}

def compute_logK_and_uv(model: nn.Module, cache: SpectralCache, which: str = "all", iters: int = 1):
    """
    Computes logK = mean_i log sigma_i over constrained layers and returns:
      - logK (float)
      - per-layer u,v,sigma cached for gradient penalty injection
    """
    sigmas = []
    for _, W, kind in iter_constrained_weights(model, which=which):
        key = id(W)
        if key not in cache.store:
            out_dim = W.shape[0]
            u0 = torch.randn(out_dim, device=W.device, dtype=torch.float32)
            u0 = u0 / (u0.norm() + 1e-12)
            cache.store[key] = {"u": u0, "v": None, "sigma": None, "kind": kind, "shape": tuple(W.shape)}

        entry = cache.store[key]
        u = entry["u"]

        Wmat = W.reshape(W.shape[0], -1).detach()
        sigma, u_new, v_new = power_iter_sigma(Wmat, u, iters=iters)

        entry["u"] = u_new
        entry["v"] = v_new
        entry["sigma"] = float(sigma.item())
        cache.last_sigmas[key] = float(sigma.item())

        sigmas.append(sigma)

    if len(sigmas) == 0:
        return float("nan")

    sigmas = torch.stack(sigmas)
    logK = torch.log(sigmas).mean().item()
    cache.last_logK = float(logK)
    return float(logK)

def apply_alm_grad_penalty(model: nn.Module, cache: SpectralCache, g_pos: float, coef: float, which: str = "all"):
    """
    Adds approx grad of (coef * g_pos) where g = logK - B, constraint logK <= B.
    Using fixed u,v from latest power-iter: grad log sigma ≈ (1/sigma) * u v^T.
    Global logK = mean log sigma => scale by 1/L.
    """
    if g_pos <= 0 or coef <= 0:
        return

    # count constrained layers
    keys = []
    tensors = []
    kinds = []
    for _, W, kind in iter_constrained_weights(model, which=which):
        key = id(W)
        if key in cache.store and cache.store[key].get("v", None) is not None:
            keys.append(key); tensors.append(W); kinds.append(kind)
    L = len(keys)
    if L == 0:
        return

    for key, W, kind in zip(keys, tensors, kinds):
        if W.grad is None:
            continue
        entry = cache.store[key]
        sigma = float(entry["sigma"])
        u = entry["u"].to(dtype=torch.float32)
        v = entry["v"].to(dtype=torch.float32)

        # grad(log sigma) ≈ (1/sigma) * u v^T, scaled by 1/L
        scale = (coef * (1.0 / max(sigma, 1e-12)) * (1.0 / L))
        gmat = (u[:, None] * v[None, :]) * scale  # float32

        gmat = gmat.reshape(W.shape[0], -1).reshape_as(W).to(dtype=W.grad.dtype)
        W.grad.add_(gmat)


# In[23]:


def train_one_run(seed: int, data_cfg: DataConfig, train_cfg: TrainConfig, device: str = "cuda", deterministic: bool = False, log_every: int = 10):
    set_seed(seed, deterministic=deterministic)

    train_loader, val_loader, test_loader, num_classes = make_loaders(data_cfg)
    model = make_cifar_resnet(train_cfg.depth, num_classes=num_classes).to(device)

    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    # optimizer
    if train_cfg.method.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    elif train_cfg.method.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
                              momentum=train_cfg.momentum, nesterov=True)
    else:
        raise ValueError("method must be 'adamw' or 'sgd'")

    scheduler = make_scheduler(optimizer, epochs=train_cfg.epochs, steps_per_epoch=len(train_loader), warmup_epochs=5)

    scaler = torch.amp.GradScaler(enabled=(train_cfg.amp and device.startswith("cuda")))

    # ALM state
    alm = train_cfg.alm
    cache = SpectralCache()
    B = None
    lam = float(alm.lam0)
    rho = float(alm.rho0)
    g_ema = 0.0
    bad_count = 0
    good_count = 0

    history = []
    best_val = -1.0
    test_at_best_val = -1.0

    global_step = 0

    def eval_acc(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()
        return correct / max(1, total)

    for ep in range(1, train_cfg.epochs + 1):
        model.train()
        tr_correct, tr_total = 0, 0

        for x, y in train_loader:
            global_step += 1
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(train_cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # IMPORTANT: unscale grads before adding ALM penalty + clipping
            scaler.unscale_(optimizer)

            # ---- ALM budget calibration at end of warmup ----
            if train_cfg.use_alm and ep == alm.warmup_epochs and B is None:
                # compute logK on current weights
                logK = compute_logK_and_uv(model, cache, which=alm.which, iters=1)
                B = logK + float(alm.budget_delta) + float(alm.margin)
                cache.last_B = float(B)

            # ---- Constraint evaluation every N steps (cheap) ----
            logK = float("nan")
            g_pos = 0.0
            if train_cfg.use_alm and (B is not None) and (global_step % alm.constraint_every == 0):
                logK = compute_logK_and_uv(model, cache, which=alm.which, iters=1)
                g = float(logK - B)
                g_pos = max(0.0, g)

                # EMA of violation
                g_ema = alm.ema_beta * g_ema + (1 - alm.ema_beta) * g_pos

                # penalty coefficient for gradient term
                coef = lam + rho * g_pos
                apply_alm_grad_penalty(model, cache, g_pos=g_pos, coef=coef, which=alm.which)

            # grad clip (helps stability; ICML reviewers expect stability story)
            if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_total += y.numel()

        # ---- epoch-end dual / rho update (stable, bounded) ----
        if train_cfg.use_alm and (B is not None):
            # dual ascent on positive constraint violation
            lam = min(alm.lam_max, max(0.0, lam + alm.dual_lr * rho * g_ema))

            # adaptive rho based on sustained violation / feasibility
            if g_ema > alm.tol:
                bad_count += 1
                good_count = 0
                if bad_count >= alm.patience:
                    rho = min(alm.rho_max, rho * alm.rho_growth)
                    bad_count = 0
            elif g_ema < (alm.tol * 0.3):
                good_count += 1
                bad_count = 0
                if good_count >= alm.patience:
                    rho = max(alm.rho_min, rho * alm.rho_shrink)
                    good_count = 0

        train_acc = tr_correct / max(1, tr_total)
        val_acc = eval_acc(val_loader)
        test_acc = eval_acc(test_loader)

        if val_acc > best_val:
            best_val = val_acc
            test_at_best_val = test_acc

        row = {
            "seed": seed,
            "method": ("ALM_"+train_cfg.method.upper()) if train_cfg.use_alm else train_cfg.method.upper(),
            "epoch": ep,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "best_val_acc_so_far": best_val,
            "test_at_best_val_so_far": test_at_best_val,
            "logK": cache.last_logK,
            "B": cache.last_B,
            "g_ema": g_ema,
            "lam": lam,
            "rho": rho,
        }
        history.append(row)

        if (ep == 1) or (ep % log_every == 0) or (ep == train_cfg.epochs):
            tag = "ALM" if train_cfg.use_alm else train_cfg.method.upper()
            extra = ""
            if train_cfg.use_alm and (B is not None):
                extra = f" logK={cache.last_logK:.4f} g_ema={g_ema:.6f} lam={lam:.3f} rho={rho:.3f} B={cache.last_B:.4f}"
            print(f"[{tag}|{data_cfg.dataset}|{train_cfg.depth}] ep={ep:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}{extra}")

    hist_df = pd.DataFrame(history)

    summary = {
        "seed": seed,
        "method": ("ALM_"+train_cfg.method.upper()) if train_cfg.use_alm else train_cfg.method.upper(),
        "final_test_acc": float(hist_df.iloc[-1]["test_acc"]),
        "best_val_acc": float(hist_df["val_acc"].max()),
        "test_at_best_val": float(hist_df.loc[hist_df["val_acc"].idxmax(), "test_acc"]),
    }
    return hist_df, summary


# In[24]:


@dataclass
class TinySweep:
    # Keep this small.
    lr_grid: Tuple[float, ...] = (3e-4, 5e-4)
    wd_grid: Tuple[float, ...] = (0.02, 0.05)
    # ALM knobs (small set)
    budget_delta_grid: Tuple[float, ...] = (0.30, 0.50)
    rho0_grid: Tuple[float, ...] = (0.05, 0.10)

    seed_tune: int = 0
    epochs_tune: int = 20

    seeds_final: Tuple[int, ...] = (1,2,3)
    epochs_final: int = 100

def tiny_tune(data_cfg: DataConfig, depth: str, method: str, use_alm: bool, sweep: TinySweep, device: str):
    """
    Returns best config by max val_acc over a cheap 20-epoch run.
    """
    best = None
    best_score = -1.0

    for lr in sweep.lr_grid:
        for wd in sweep.wd_grid:
            if not use_alm:
                tc = TrainConfig(depth=depth, epochs=sweep.epochs_tune, lr=lr, weight_decay=wd,
                                 method=method, use_alm=False, amp=False)
                df, summ = train_one_run(sweep.seed_tune, data_cfg, tc, device=device, deterministic=False, log_every=20)
                score = float(df["val_acc"].max())
                if score > best_score:
                    best_score = score
                    best = {"lr": lr, "wd": wd}
            else:
                for bd in sweep.budget_delta_grid:
                    for rho0 in sweep.rho0_grid:
                        tc = TrainConfig(
                            depth=depth, epochs=sweep.epochs_tune, lr=lr, weight_decay=wd,
                            method=method, use_alm=True, amp=False,
                            alm=ALMConfig(
                                warmup_epochs=5,
                                budget_delta=bd,
                                rho0=rho0,
                                dual_lr=0.02,
                                rho_growth=1.6, rho_shrink=0.9,
                                rho_max=2.0, lam_max=3.0,
                                constraint_every=20,
                                which="all"
                            )
                        )
                        df, summ = train_one_run(sweep.seed_tune, data_cfg, tc, device=device, deterministic=False, log_every=20)
                        score = float(df["val_acc"].max())
                        if score > best_score:
                            best_score = score
                            best = {"lr": lr, "wd": wd, "budget_delta": bd, "rho0": rho0}

    print(f"[TUNE] best {('ALM_' if use_alm else '')}{method.upper()} score={best_score:.4f} cfg={best}")
    return best

def final_compare(data_cfg: DataConfig, depth: str, sweep: TinySweep, tuned: Dict[str, Dict[str, Any]], device: str):
    all_hist = []
    all_summ = []

    def run_cfg(seed, method, use_alm, cfg):
        tc = TrainConfig(
            depth=depth, epochs=sweep.epochs_final,
            lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]),
            method=method, amp=False,
            use_alm=use_alm
        )
        if use_alm:
            tc.alm = ALMConfig(
                warmup_epochs=5,
                budget_delta=float(cfg["budget_delta"]),
                rho0=float(cfg["rho0"]),
                dual_lr=0.02,
                rho_growth=1.6, rho_shrink=0.9,
                rho_max=2.0, lam_max=3.0,
                constraint_every=20,
                which="all"
            )
        df, summ = train_one_run(seed, data_cfg, tc, device=device, deterministic=False, log_every=20)
        return df, summ

    # Methods included (ICML baseline coverage, minimal compute)
    methods = [
        ("sgd",  False, "SGD"),
        ("adamw",False, "ADAMW"),
        ("adamw",True,  "ALM_ADAMW"),
    ]

    for seed in sweep.seeds_final:
        for method, use_alm, tag in methods:
            cfg = tuned[tag]
            print("\n" + "-"*70)
            print(f"RUN seed={seed} | {tag}")
            df, summ = run_cfg(seed, method, use_alm, cfg)
            all_hist.append(df)
            all_summ.append(summ)

    summ_df = pd.DataFrame(all_summ)
    hist_df = pd.concat(all_hist, ignore_index=True)

    # Paired tests (same seeds): ALM vs AdamW, SGD vs AdamW, ALM vs SGD
    def paired(methodA, methodB):
        a = []
        b = []
        for s in sweep.seeds_final:
            a.append(float(summ_df[(summ_df.seed==s) & (summ_df.method==methodA)]["final_test_acc"].iloc[0]))
            b.append(float(summ_df[(summ_df.seed==s) & (summ_df.method==methodB)]["final_test_acc"].iloc[0]))
        meanA, loA, hiA = bootstrap_ci(a, seed=0)
        meanB, loB, hiB = bootstrap_ci(b, seed=1)
        p = paired_permutation_test(a, b, seed=2)
        diff = float(np.mean(np.array(a) - np.array(b)))
        return (meanA, loA, hiA, meanB, loB, hiB, p, diff)

    print("\n==================== FINAL SUMMARY ====================")
    print(summ_df.sort_values(["method","seed"]).reset_index(drop=True))

    print("\nPaired stats (final_test_acc):")
    mA, loA, hiA, mB, loB, hiB, p, diff = paired("ALM_ADAMW", "ADAMW")
    print(f"ALM_ADAMW mean={mA:.4f} CI=[{loA:.4f},{hiA:.4f}] | ADAMW mean={mB:.4f} CI=[{loB:.4f},{hiB:.4f}] | p={p:.4f} | diff={diff:+.4f}")

    mA, loA, hiA, mB, loB, hiB, p, diff = paired("SGD", "ADAMW")
    print(f"SGD      mean={mA:.4f} CI=[{loA:.4f},{hiA:.4f}] | ADAMW mean={mB:.4f} CI=[{loB:.4f},{hiB:.4f}] | p={p:.4f} | diff={diff:+.4f}")

    mA, loA, hiA, mB, loB, hiB, p, diff = paired("ALM_ADAMW", "SGD")
    print(f"ALM_ADAMW mean={mA:.4f} CI=[{loA:.4f},{hiA:.4f}] | SGD   mean={mB:.4f} CI=[{loB:.4f},{hiB:.4f}] | p={p:.4f} | diff={diff:+.4f}")

    return summ_df, hist_df


# In[ ]:


# Choose ONE experiment first (cheap + decisive). If ALM wins, add cifar100/resnet18.
data_cfg = DataConfig(dataset="cifar10", batch_size=128, num_workers=4, val_frac=0.1, split_seed=123)

depth = "resnet18"
sweep = TinySweep(
    lr_grid=(3e-4, 5e-4),
    wd_grid=(0.02, 0.05),
    budget_delta_grid=(0.30, 0.50),
    rho0_grid=(0.05, 0.10),
    seed_tune=0,
    epochs_tune=20,
    seeds_final=(1,2,3),
    epochs_final=100
)

# Tiny tune each method (objective, cheap)
tuned = {}
tuned["SGD"]      = tiny_tune(data_cfg, depth, method="sgd",   use_alm=False, sweep=sweep, device=device)
tuned["ADAMW"]    = tiny_tune(data_cfg, depth, method="adamw", use_alm=False, sweep=sweep, device=device)
tuned["ALM_ADAMW"]= tiny_tune(data_cfg, depth, method="adamw", use_alm=True,  sweep=sweep, device=device)

# Final 3-seed comparison
summ_df, hist_df = final_compare(data_cfg, depth, sweep, tuned, device=device)


# In[26]:


import torch.optim as optim


# In[ ]:




