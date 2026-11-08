#!/usr/bin/env python3
"""Graph-only Python wrapper for hydro-mena: renders the Figure-5-style
heatmap (module eigengene vs environmental trait correlations, r with p in
parentheses) from the CSVs the Rust CLI writes. No compute happens here."""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def fig5_heatmap(fig5_r_csv: str, fig5_p_csv: str, out_png: str, out_svg: str | None = None):
    R = pd.read_csv(fig5_r_csv, index_col=0)
    P = pd.read_csv(fig5_p_csv, index_col=0)
    R.index = [f"#{i+1}" if not str(i).startswith("#") and not str(i).startswith("module") else str(i).replace("module_", "#") for i in R.index]
    P.index = R.index
    annot = pd.DataFrame(
        [[f"{r:.2f}\n({p:.3f})" for r, p in zip(row_r, row_p)] for row_r, row_p in zip(R.values, P.values)],
        index=R.index, columns=R.columns,
    )
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * R.shape[1] + 3), max(4, 0.7 * R.shape[0] + 2)))
    sns.heatmap(
        R, annot=annot, fmt="", cmap="RdYlGn", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white", cbar_kws={"label": "Pearson r"},
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Module eigengenes vs environmental traits (r, p in parentheses)\n"
                 "Deng et al. 2012 Fig. 5 style — hydro-mena")
    ax.set_xlabel("environmental trait")
    ax.set_ylabel("module eigengene")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if out_svg:
        fig.savefig(out_svg)
    print(f"wrote {out_png}" + (f" and {out_svg}" if out_svg else ""))
    # also print the matrix for the chat
    print("\nr matrix (modules x traits):")
    print(R.round(3).to_string())
    print("\np matrix:")
    print(P.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description="plot hydro-mena outputs (graphs only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f5 = sub.add_parser("fig5", help="render module-eigengene vs trait heatmap")
    f5.add_argument("--r", required=True, help="fig5_r.csv from hydro-mena run")
    f5.add_argument("--p", required=True, help="fig5_p.csv")
    f5.add_argument("--out", required=True, help="output PNG path")
    f5.add_argument("--svg", default=None, help="optional SVG output path")
    args = ap.parse_args()
    if args.cmd == "fig5":
        fig5_heatmap(args.r, args.p, args.out, args.svg)


if __name__ == "__main__":
    sys.exit(main())
