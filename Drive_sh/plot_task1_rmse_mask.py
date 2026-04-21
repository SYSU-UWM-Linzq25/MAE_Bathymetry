import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 修改这里
# =========================
ROOT = Path(r"/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Upstream_Model_ReTrain/eval_upstream_maskratio_sweep")
OUTDIR = ROOT / "plots_rmse_m_mask"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 你要分析的 mask ratio
MASK_RATIOS = [0.75, 0.50, 0.35, 0.20]

# baseline ratio，用于做 paired delta
BASELINE = 0.75


def ratio_dir_name(r: float) -> str:
    return f"mr_{r:.2f}"


def load_all_results(root: Path, ratios: list[float]) -> tuple[dict, dict]:
    """
    返回:
      summaries[ratio] = summary.json 内容
      metrics[ratio]   = metrics.csv DataFrame
    """
    summaries = {}
    metrics = {}

    for r in ratios:
        d = root / ratio_dir_name(r)
        summary_file = d / "summary.json"
        metrics_file = d / "metrics.csv"

        if not summary_file.exists():
            raise FileNotFoundError(f"Missing: {summary_file}")
        if not metrics_file.exists():
            raise FileNotFoundError(f"Missing: {metrics_file}")

        with open(summary_file, "r", encoding="utf-8") as f:
            summaries[r] = json.load(f)

        df = pd.read_csv(metrics_file)
        if "path" not in df.columns or "rmse_m_mask" not in df.columns:
            raise ValueError(f"{metrics_file} 缺少必要列 path / rmse_m_mask")
        metrics[r] = df

    return summaries, metrics


def make_summary_table(summaries: dict) -> pd.DataFrame:
    rows = []
    for r in sorted(summaries.keys(), reverse=True):
        s = summaries[r]["rmse_m_mask"]
        rows.append({
            "mask_ratio": r,
            "count": s["count"],
            "mean": s["mean"],
            "median": s["median"],
            "p10": s["p10"],
            "p90": s["p90"],
            "p95": s["p95"],
            "min": s["min"],
            "max": s["max"],
        })
    return pd.DataFrame(rows).sort_values("mask_ratio", ascending=False)


def plot_trend(summary_df: pd.DataFrame, out_png: Path):
    x = summary_df["mask_ratio"].values
    mean_y = summary_df["mean"].values
    p95_y = summary_df["p95"].values
    p10_y = summary_df["p10"].values
    p90_y = summary_df["p90"].values

    plt.figure(figsize=(7, 5))
    plt.plot(x, mean_y, marker="o", label="mean rmse_m_mask")
    plt.plot(x, p95_y, marker="s", label="p95 rmse_m_mask")

    # 给 mean 加一个 p10-p90 的阴影带
    plt.fill_between(x, p10_y, p90_y, alpha=0.15, label="p10-p90")

    plt.gca().invert_xaxis()  # 从 0.75 -> 0.20 看起来更直观
    plt.xlabel("mask_ratio")
    plt.ylabel("rmse_m_mask (m)")
    plt.title("Task 1: rmse_m_mask vs mask_ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_boxplot(metrics: dict, out_png: Path):
    ratios_sorted = sorted(metrics.keys(), reverse=True)
    data = [metrics[r]["rmse_m_mask"].dropna().values for r in ratios_sorted]
    labels = [f"{r:.2f}" for r in ratios_sorted]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("mask_ratio")
    plt.ylabel("rmse_m_mask (m)")
    plt.title("Task 1: distribution of rmse_m_mask")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_paired_delta(metrics: dict, baseline: float) -> pd.DataFrame:
    """
    返回一个长表：
      path, mask_ratio, delta_vs_baseline
    delta = rmse(mask_ratio) - rmse(baseline)
    小于 0 代表比 baseline 更好
    """
    if baseline not in metrics:
        raise ValueError(f"baseline {baseline} 不在 metrics 里")

    base = metrics[baseline][["path", "rmse_m_mask"]].copy()
    base = base.rename(columns={"rmse_m_mask": "rmse_base"})

    rows = []
    for r, df in metrics.items():
        if r == baseline:
            continue

        cur = df[["path", "rmse_m_mask"]].copy()
        cur = cur.rename(columns={"rmse_m_mask": "rmse_cur"})

        merged = pd.merge(base, cur, on="path", how="inner")
        merged["mask_ratio"] = r
        merged["delta_vs_baseline"] = merged["rmse_cur"] - merged["rmse_base"]
        rows.append(merged[["path", "mask_ratio", "delta_vs_baseline"]])

    if not rows:
        return pd.DataFrame(columns=["path", "mask_ratio", "delta_vs_baseline"])

    return pd.concat(rows, ignore_index=True)


def plot_delta_boxplot(delta_df: pd.DataFrame, out_png: Path):
    ratios_sorted = sorted(delta_df["mask_ratio"].unique(), reverse=True)
    data = [delta_df.loc[delta_df["mask_ratio"] == r, "delta_vs_baseline"].dropna().values
            for r in ratios_sorted]
    labels = [f"{r:.2f}" for r in ratios_sorted]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("mask_ratio")
    plt.ylabel(f"rmse_m_mask - rmse_m_mask({BASELINE:.2f}) (m)")
    plt.title("Task 1: paired delta vs baseline")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    summaries, metrics = load_all_results(ROOT, MASK_RATIOS)

    # 1) summary table
    summary_df = make_summary_table(summaries)
    summary_csv = OUTDIR / "rmse_m_mask_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 2) trend plot
    plot_trend(summary_df, OUTDIR / "rmse_m_mask_trend.png")

    # 3) boxplot
    plot_boxplot(metrics, OUTDIR / "rmse_m_mask_boxplot.png")

    # 4) paired delta
    delta_df = build_paired_delta(metrics, BASELINE)
    delta_csv = OUTDIR / "rmse_m_mask_delta_vs_baseline.csv"
    delta_df.to_csv(delta_csv, index=False)
    if not delta_df.empty:
        plot_delta_boxplot(delta_df, OUTDIR / "rmse_m_mask_delta_boxplot.png")

    print("[DONE]", summary_csv)
    print("[DONE]", OUTDIR / "rmse_m_mask_trend.png")
    print("[DONE]", OUTDIR / "rmse_m_mask_boxplot.png")
    print("[DONE]", delta_csv)
    if not delta_df.empty:
        print("[DONE]", OUTDIR / "rmse_m_mask_delta_boxplot.png")


if __name__ == "__main__":
    main()
