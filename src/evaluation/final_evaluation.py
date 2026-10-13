"""
final_evaluation.py — Consolidates the final evaluation (model vs baseline,
diagonal vs vertical, broken down by num_obs) into a paper-ready table,
with binomial confidence intervals (Wilson, 95%) -- the rate alone is not
enough to tell whether a difference is real or sampling noise.

Requires:
  - Output folders of a `python3 -m src.model.set_diffusion sample` run
    (containing results_detail.json inside) for the model, one per route.
  - .json from baseline_generator.py --output (with conteos_por_num_obs), one
    per route -- run it with --output if you don't have it saved yet:
        python3 baseline_generator.py --hppm-exe ./hppm --n 2000 --seeds 42 \\
            --num-obs-min 5 --num-obs-max 250 --L auto --route diagonal \\
            --workers 30 --output baseline_diagonal.json

Usage:
    python3 -m src.evaluation.final_evaluation \\
        --model-diagonal final_diagonal_model.pt --model-vertical final_vertical_model.pt \\
        --baseline-diagonal baseline_diagonal.json --baseline-vertical baseline_vertical.json \\
        --output-dir artifacts/evaluacion_paper

Output (in --output-dir):
    tabla.md        markdown table ready to paste into the paper
    tabla.csv       same table as csv (bin, route, source, successes, total, rate, ci_lo, ci_hi)
    figura.png      model vs baseline bars, one panel per route
"""
import argparse
import csv
import json
import math
import os

from src.common.compat import normalize

BINS = [(5, 50), (51, 100), (101, 150), (151, 200), (201, 250)]


def wilson(k, n, z=1.96):
    """Wilson binomial confidence interval (better than the normal one when
    k=0 or k=n, which is exactly what happens in several bins). Returns
    (rate, ci_lo, ci_hi); if n==0, returns (None, None, None)."""
    if n == 0:
        return None, None, None
    p = k / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return p, max(0.0, center - half_width), min(1.0, center + half_width)


def model_counts(result_folders):
    """result_folders: a single folder, or a list of folders (for example one
    per n bin, which is what you get from requesting a punctual
    --num-obs-min/--num-obs-max per bin instead of the full distribution of a
    single run). It gathers the detail of all of them before binning, so it
    makes no difference whether a punctual bin came from its own dedicated
    folder or from a big run that mixes all of them together."""
    if isinstance(result_folders, str):
        result_folders = [result_folders]
    detail = []
    for folder in result_folders:
        json_path = os.path.join(folder, "results_detail.json")
        with open(json_path, encoding="utf-8") as f:
            detail += normalize(json.load(f))
    counts = {}
    for lo, hi in BINS:
        subset = [r for r in detail if lo <= r["num_obs"] <= hi]
        successes = sum(1 for r in subset if r["state"] == "success")
        counts[f"{lo}-{hi}"] = (successes, len(subset))
    return counts


def baseline_counts(json_path):
    with open(json_path, encoding="utf-8") as f:
        report = normalize(json.load(f))
    if "counts_by_num_obs" not in report:
        raise SystemExit(f"{json_path} has no 'counts_by_num_obs' -- it is from an old version "
                         "of baseline_generator.py (without the raw counts fix). Run it again.")
    return {k: tuple(v) for k, v in report["counts_by_num_obs"].items()}


def build_rows(route_name, model_counts_r, baseline_counts_r):
    rows = []
    for lo, hi in BINS:
        key = f"{lo}-{hi}"
        for source, counts in (("model", model_counts_r), ("baseline", baseline_counts_r)):
            k, n = counts.get(key, (0, 0))
            rate, ci_lo, ci_hi = wilson(k, n)
            rows.append({"route": route_name, "bin_n": key, "source": source,
                         "successes": k, "total": n, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi})
    return rows


def write_markdown(rows, output_path):
    routes = sorted(set(f["route"] for f in rows))
    with open(output_path, "w", encoding="utf-8") as f:
        for route_name in routes:
            f.write(f"\n### Route: {route_name}\n\n")
            f.write("| n | model (successes/total) | model % [95% CI] | baseline (successes/total) | "
                    "baseline % [95% CI] | factor |\n")
            f.write("|---|---|---|---|---|---|\n")
            by_bin = {}
            for row in rows:
                if row["route"] != route_name:
                    continue
                by_bin.setdefault(row["bin_n"], {})[row["source"]] = row
            for lo, hi in BINS:
                key = f"{lo}-{hi}"
                bin_rows = by_bin.get(key, {})
                model_row, baseline_row = bin_rows.get("model"), bin_rows.get("baseline")

                def cell(x):
                    if x is None or x["total"] == 0:
                        return "-", "no data"
                    return f"{x['successes']}/{x['total']}", (
                        f"{x['rate']:.1%} [{x['ci_lo']:.1%}, {x['ci_hi']:.1%}]")

                mo_frac, mo_pct = cell(model_row)
                ba_frac, ba_pct = cell(baseline_row)
                factor = "-"
                if model_row and baseline_row and baseline_row["rate"] and baseline_row["rate"] > 0:
                    factor = f"{model_row['rate'] / baseline_row['rate']:.1f}x"
                elif (model_row and baseline_row and model_row["rate"] and model_row["rate"] > 0
                      and baseline_row["rate"] == 0):
                    factor = "baseline=0"
                f.write(f"| {key} | {mo_frac} | {mo_pct} | {ba_frac} | {ba_pct} | {factor} |\n")


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["route", "bin_n", "source", "successes", "total",
                                          "rate", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(rows)


def plot_bars(rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    routes = sorted(set(f["route"] for f in rows))
    fig, axs = plt.subplots(1, len(routes), figsize=(6 * len(routes), 4.5), sharey=True)
    axs = [axs] if len(routes) == 1 else list(axs)
    bin_labels = [f"{lo}-{hi}" for lo, hi in BINS]
    x = np.arange(len(BINS))
    width = 0.35

    for ax, route_name in zip(axs, routes):
        for offset, source, color in ((-width / 2, "model", "#1565c0"), (width / 2, "baseline", "#c62828")):
            rates, err_lo, err_hi = [], [], []
            for lo, hi in BINS:
                key = f"{lo}-{hi}"
                row = next((f for f in rows if f["route"] == route_name and f["bin_n"] == key
                            and f["source"] == source), None)
                t = row["rate"] if row and row["rate"] is not None else 0.0
                lo_ci = row["ci_lo"] if row and row["ci_lo"] is not None else 0.0
                hi_ci = row["ci_hi"] if row and row["ci_hi"] is not None else 0.0
                rates.append(t)
                err_lo.append(max(0, t - lo_ci))
                err_hi.append(max(0, hi_ci - t))
            ax.bar(x + offset, rates, width, yerr=[err_lo, err_hi], capsize=3,
                   label=source, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=0)
        ax.set_xlabel("num. obstacles")
        ax.set_title(f"route: {route_name}")
        ax.legend()
    axs[0].set_ylabel("verified success rate (HPPM)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-diagonal", required=True, nargs="+",
                   help="one or more output folders of a diffusion sampling run (for example one per n "
                        "bin if you used a punctual --num-obs-min/--num-obs-max per bin)")
    p.add_argument("--model-vertical", required=True, nargs="+")
    p.add_argument("--baseline-diagonal", required=True, help=".json from baseline_generator.py --output")
    p.add_argument("--baseline-vertical", required=True)
    p.add_argument("--output-dir", default="artifacts/evaluacion_paper")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    rows += build_rows("diagonal", model_counts(args.model_diagonal),
                       baseline_counts(args.baseline_diagonal))
    rows += build_rows("vertical", model_counts(args.model_vertical),
                       baseline_counts(args.baseline_vertical))

    write_markdown(rows, os.path.join(args.output_dir, "tabla.md"))
    write_csv(rows, os.path.join(args.output_dir, "tabla.csv"))
    plot_bars(rows, os.path.join(args.output_dir, "figura.png"))

    print(f"Done: {args.output_dir}/tabla.md, tabla.csv, figura.png")
    with open(os.path.join(args.output_dir, "tabla.md"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
