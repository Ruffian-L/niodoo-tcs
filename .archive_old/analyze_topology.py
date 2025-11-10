import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    results_dir = Path("/workspace/Niodoo-Final/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Correlation plots
    for metric in ["betti_1", "spectral_gap", "persistence_entropy"]:
        g = sns.jointplot(data=df, x=metric, y="rouge_l", kind="reg", height=6)
        g.fig.suptitle(f"ROUGE-L vs {metric}")
        out = results_dir / f"topology_correlation_{metric}.png"
        g.fig.tight_layout()
        g.fig.savefig(out)
        print(f"Saved {out}")

    # Learning curve (cumulative mean per mode)
    curves = []
    for mode, group in df.groupby("mode"):
        group = group.copy()
        group["idx"] = range(1, len(group) + 1)
        group["cum_mean"] = group["rouge_l"].expanding().mean()
        curves.append(group)
    curve_df = pd.concat(curves, ignore_index=True)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=curve_df, x="idx", y="cum_mean", hue="mode")
    plt.title("Cumulative Mean ROUGE-L (Learning Curve)")
    plt.xlabel("Sample Index")
    plt.ylabel("Cumulative Mean ROUGE-L")
    out = results_dir / "learning_curve.png"
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved {out}")

    # Ablation table
    ablation = df.groupby("mode")["rouge_l"].mean().reset_index()
    ablation = ablation.sort_values("rouge_l", ascending=False)
    out_csv = results_dir / "ablation_study.csv"
    ablation.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Performance summary
    summary_path = results_dir / "SYSTEM_PERFORMANCE.md"
    best = ablation.iloc[0]
    baseline = ablation[ablation["mode"] == "erag"]["rouge_l"].max() if "erag" in ablation["mode"].values else None
    gain = None
    if baseline is not None and baseline > 0:
        gain = (best["rouge_l"] - baseline) / baseline
    with open(summary_path, "w") as f:
        f.write("# System Performance\n\n")
        f.write(ablation.to_markdown(index=False))
        f.write("\n\n")
        if gain is not None:
            f.write(f"Full vs ERAG gain: {gain*100:.2f}%\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_topology.py <csv>")
        sys.exit(1)
    main(sys.argv[1])



