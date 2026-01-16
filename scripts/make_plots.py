import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("results", "tables", "c10_r18_paired_seed_results.csv")
OUT_DIR = os.path.join("results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["seed", "method"])

    # Plot test accuracy by seed for each method
    plt.figure()
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        plt.plot(sub["seed"], sub["final_test_acc"], marker="o", label=method)
    plt.xlabel("Seed")
    plt.ylabel("Final Test Accuracy")
    plt.title("CIFAR-10 / ResNet-18: Final Test Accuracy by Seed")
    plt.legend()
    out1 = os.path.join(OUT_DIR, "c10_r18_test_acc_by_seed.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot paired differences (ALM - AdamW) by seed (if both exist)
    if set(df["method"]) >= {"AdamW", "ALM"}:
        a = df[df["method"] == "AdamW"][["seed", "final_test_acc"]].rename(columns={"final_test_acc": "adamw"})
        b = df[df["method"] == "ALM"][["seed", "final_test_acc"]].rename(columns={"final_test_acc": "alm"})
        m = a.merge(b, on="seed", how="inner")
        m["diff"] = m["alm"] - m["adamw"]

        plt.figure()
        plt.axhline(0.0, linewidth=1)
        plt.plot(m["seed"], m["diff"], marker="o")
        plt.xlabel("Seed")
        plt.ylabel("ALM - AdamW (Final Test Acc)")
        plt.title("CIFAR-10 / ResNet-18: Paired Test-Accuracy Differences")
        out2 = os.path.join(OUT_DIR, "c10_r18_paired_diff.png")
        plt.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close()

    print("Wrote figures to:", OUT_DIR)

if __name__ == "__main__":
    main()
