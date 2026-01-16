import os
import numpy as np
import pandas as pd

CSV_PATH = os.path.join("results", "tables", "c10_r18_paired_seed_results.csv")
OUT_PATH = os.path.join("results", "tables", "c10_r18_stats_recomputed.txt")

def bootstrap_ci(x, n_boot=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        means.append(float(np.mean(samp)))
    means = np.array(means)
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1 - alpha/2))
    return float(np.mean(x)), lo, hi

def paired_permutation_test(x, y, n_perm=200000, seed=0):
    # H0: mean(x - y) = 0. Use random sign flips.
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    obs = float(np.mean(d))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d), replace=True)
        stat = float(np.mean(d * signs))
        if abs(stat) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1), obs

def main():
    df = pd.read_csv(CSV_PATH)
    adam = df[df["method"] == "AdamW"].sort_values("seed")
    alm  = df[df["method"] == "ALM"].sort_values("seed")

    merged = adam[["seed","final_test_acc"]].merge(
        alm[["seed","final_test_acc"]],
        on="seed",
        suffixes=("_adamw", "_alm"),
        how="inner"
    )

    adam_acc = merged["final_test_acc_adamw"].to_numpy()
    alm_acc  = merged["final_test_acc_alm"].to_numpy()

    mean_a, lo_a, hi_a = bootstrap_ci(adam_acc, seed=0)
    mean_b, lo_b, hi_b = bootstrap_ci(alm_acc, seed=1)
    pval, diff = paired_permutation_test(alm_acc, adam_acc, seed=2)

    text = []
    text.append("FINAL RESULTS (paired by seed)")
    text.append(f"AdamW test_acc: mean={mean_a:.4f}  95%CI=[{lo_a:.4f},{hi_a:.4f}]")
    text.append(f"ALM   test_acc: mean={mean_b:.4f}  95%CI=[{lo_b:.4f},{hi_b:.4f}]")
    text.append(f"Paired permutation p-value (ALM vs AdamW): {pval:.6f}")
    text.append(f"Mean diff (ALM-AdamW) = {diff:.4f}")
    out = "\n".join(text) + "\n"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)

    print(out)
    print("Wrote:", OUT_PATH)

if __name__ == "__main__":
    main()
