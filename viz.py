#!/usr/bin/env python3
"""Figure suite for hydro-mena runs (graphs only — no compute).

Every figure reads CSVs/JSON written by `hydro-mena run`, so a figure can always
be traced back to the numbers that produced it.

    pixi run python viz.py all <run_dir> --out <fig_dir>
    pixi run python viz.py fig5 <run_dir> --out fig.png
    pixi run python viz.py network <run_dir> --out fig.png

Figures: fig5, network, roles, rmt, spectrum, degrees, modules, bench.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

MODULE_CMAP = "tab20"


# ----------------------------------------------------------------- helpers
def read(run_dir: str, name: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def report(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, out: str) -> str:
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if out.endswith(".png"):
        fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ----------------------------------------------------------------- figures
def _select_traits(r: pd.DataFrame, p: pd.DataFrame, max_cols: int = 12):
    """Pick which traits to display, by effect size ONLY.

    Selecting columns by significance (as an earlier version did) is
    cherry-picking: with ~1,500 tests, some columns are always "significant".
    Rank by max |r| and ignore p entirely; p/q are still annotated in the cells.
    """
    cols = list(r.columns)
    if len(cols) <= max_cols:
        return r, p, False
    keep = sorted(cols, key=lambda c: -r[c].abs().max())[:max_cols]
    keep = [c for c in cols if c in keep]
    return r[keep], p[keep], True


def fig5(run_dir: str, out: str) -> str:
    """Module eigengene x trait correlations, p annotated as in Deng Fig. 5."""
    r = pd.read_csv(os.path.join(run_dir, "fig5_r.csv"), index_col=0)
    p = pd.read_csv(os.path.join(run_dir, "fig5_p.csv"), index_col=0)
    n_all = r.shape[1]
    r, p, filtered = _select_traits(r, p)
    r.index = [str(i).replace("module_", "#") for i in r.index]
    p.index = r.index
    annot = np.array(
        [[f"{v:+.2f}\n{stars(q)}" for v, q in zip(rr, pp)] for rr, pp in zip(r.values, p.values)]
    )
    fig, ax = plt.subplots(figsize=(min(17.0, max(6.0, 0.62 * r.shape[1] + 4.0)), max(3.5, 0.34 * r.shape[0] + 2.2)))
    im = ax.imshow(r.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(r.shape[1]), r.columns, rotation=30, ha="right")
    ax.set_yticks(range(r.shape[0]), r.index)
    ax.set_xlabel("environmental trait")
    ax.set_ylabel("module eigengene")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=7,
                    color="black" if abs(v) < 0.6 else "white")
    ax.set_xticks(np.arange(-0.5, r.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, r.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r")
    ax.set_title("Module eigengenes vs environmental traits"
                 + (f"\nshowing {r.shape[1]} of {n_all} traits" if filtered else "")
                 + "  (* p<0.05, ** p<0.01, *** p<0.001)")
    return save(fig, out)


def _layout(nodes: pd.DataFrame, seed: int = 3) -> np.ndarray:
    """Module-radial layout: one disc per module on a ring, members on a
    golden-angle spiral inside it.

    A force simulation at 800+ edges collapses into an unreadable hairball and
    says nothing about community structure; placing modules explicitly shows the
    partition (the point of the analysis) and keeps labels separable.
    """
    mods = list(dict.fromkeys(nodes["module"].tolist()))
    k = max(1, len(mods))
    index_of = {m: i for i, m in enumerate(mods)}
    counts = nodes.groupby("module").size().to_dict()
    pos = np.zeros((len(nodes), 2))
    for i, m in enumerate(nodes["module"].tolist()):
        b = index_of[m]
        ang = 2 * np.pi * b / k
        ring = 1.0 + 0.30 * np.sqrt(counts[m]) / 3.0
        cx, cy = ring * np.cos(ang), ring * np.sin(ang)
        n_m = max(1, counts[m])
        j = sum(1 for x in nodes["module"].tolist()[:i] if x == m)
        r = 0.34 * np.sqrt((j + 0.5) / n_m)
        th = j * 2.399963 + b
        pos[i] = (cx + r * np.cos(th), cy + r * np.sin(th))
    return pos


def network(run_dir: str, out: str, max_nodes: int = 1200) -> str:
    """Co-occurrence network, nodes coloured by module, sized by degree."""
    nodes = read(run_dir, "nodes.csv")
    edges = read(run_dir, "edges.csv")
    if nodes is None or edges is None:
        raise SystemExit("nodes.csv / edges.csv missing — rerun hydro-mena run")
    idx = {name: i for i, name in enumerate(nodes["OTU"])}
    if len(nodes) > max_nodes:
        keep = set(nodes.nlargest(max_nodes, "degree")["OTU"])
        nodes = nodes[nodes["OTU"].isin(keep)].reset_index(drop=True)
        edges = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
        idx = {name: i for i, name in enumerate(nodes["OTU"])}
    e = np.array([[idx[a], idx[b]] for a, b in zip(edges["source"], edges["target"]) if a in idx and b in idx])
    pos = _layout(nodes)
    fig, ax = plt.subplots(figsize=(9, 9))
    if len(e):
        segments = [(pos[a], pos[b]) for a, b in e]
        ax.add_collection(LineCollection(segments, colors="#c8ccd4", linewidths=0.28, alpha=0.7, zorder=1))
    mods = nodes["module"].to_numpy()
    size = 6 + 2.2 * nodes["degree"].to_numpy()
    ax.scatter(pos[:, 0], pos[:, 1], c=mods, cmap=MODULE_CMAP, s=size, edgecolors="white", linewidths=0.4, zorder=2)
    hubs = nodes.nlargest(min(5, len(nodes)), "degree")
    for _, row in hubs.iterrows():
        i = idx[row["OTU"]]
        ax.annotate(row["OTU"], pos[i], fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_title(
        f"Co-occurrence network — {len(nodes)} nodes, {len(e)} edges, {nodes['module'].nunique()} modules\n"
        "one disc per module, size = degree (top-degree OTUs labelled)"
    )
    ax.set_axis_off()
    ax.margins(0.06)
    return save(fig, out)


def roles(run_dir: str, out: str) -> str:
    """zi-Pi role scatter (Deng Eq. 8-9) with the 2.5 / 0.62 cut-offs."""
    nodes = read(run_dir, "nodes.csv")
    if nodes is None:
        raise SystemExit("nodes.csv missing")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for mod, grp in nodes.groupby("module"):
        ax.scatter(grp["zi"], grp["Pi"], s=14 + 3 * grp["degree"], alpha=0.8, label=f"module {mod}", cmap=None)
    ax.axhline(0.62, color="0.3", ls="--", lw=1)
    ax.axvline(2.5, color="0.3", ls="--", lw=1)
    ax.text(0.02, 0.95, "peripherals", transform=ax.transAxes, fontsize=8, va="top", color="0.35")
    ax.text(0.98, 0.95, "connectors", transform=ax.transAxes, fontsize=8, va="top", ha="right", color="0.35")
    ax.text(0.02, 0.02, "module hubs", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.text(0.98, 0.02, "network hubs", transform=ax.transAxes, fontsize=8, ha="right", color="0.35")
    ax.set_xlabel("within-module connectivity zi")
    ax.set_ylabel("participation coefficient Pi")
    ax.set_title("Module roles (zi <= 2.5, Pi <= 0.62)")
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="center right")
    return save(fig, out)


def rmt(run_dir: str, out: str) -> str:
    """RMT scan: chi-square against the critical value, plus graph size."""
    scan = read(run_dir, "rmt_scan.csv")
    rep = report(run_dir)
    if scan is None or scan.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "no RMT scan (fixed --st run)", ha="center", va="center")
        return save(fig, out)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(scan["st"], scan["chi2"], "o-", ms=3, label=r"$\chi^2$ observed")
    ax.plot(scan["st"], scan["crit"], "s--", ms=3, color="crimson", label=r"$\chi^2_{0.99}$ critical")
    acc = scan[scan["poisson"]]
    if len(acc):
        ax.axvline(acc["st"].iloc[0], color="green", ls=":", lw=1.2)
        ax.annotate("first Poisson-accepted st", (acc["st"].iloc[0], ax.get_ylim()[1] * 0.8),
                    fontsize=7, rotation=90, va="top", ha="right", color="green")
    ax.axvline(rep["rmt"]["st"], color="0.4", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\chi^2$ (log scale)")
    ax.legend(fontsize=7, frameon=False)
    title = f"RMT threshold scan — chosen st={rep['rmt']['st']:.3f}"
    if rep["rmt"].get("fallback"):
        title += "  (no acceptance: diagnostic fallback)"
    ax.set_title(title)
    ax2.plot(scan["st"], scan["edges"], "o-", ms=3, color="0.3")
    ax2.set_ylabel("edges")
    ax2.set_xlabel("similarity threshold st")
    return save(fig, out)


def spectrum(run_dir: str, out: str) -> str:
    """Eigenvalue spectrum + nearest-neighbour spacing vs Poisson and Wigner."""
    spec = read(run_dir, "spectrum.csv")
    if spec is None:
        raise SystemExit("spectrum.csv missing")
    lam = spec["eigenvalue"].to_numpy()
    d = spec["spacing"].dropna().to_numpy()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax.hist(lam, bins=max(10, int(math.sqrt(len(lam)) * 2)), color="#4c72b0", edgecolor="white")
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel("count")
    ax.set_title("Adjacency spectrum of the kept graph")
    if len(d) > 5:
        ax2.hist(d, bins=30, density=True, color="#4c72b0", alpha=0.75, edgecolor="white", label="observed NNSD")
        s = np.linspace(0, max(4.0, float(np.max(d))), 200)
        ax2.plot(s, np.exp(-s), "r-", lw=1.5, label="Poisson $e^{-s}$")
        ax2.plot(s, (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4), "k--", lw=1.2, label="Wigner surmise")
        ax2.set_xlabel("unfolded spacing s")
        ax2.set_ylabel("density")
        ax2.legend(fontsize=7, frameon=False)
        ax2.set_title("Nearest-neighbour spacing distribution")
    else:
        ax2.axis("off")
    return save(fig, out)


def degrees(run_dir: str, out: str) -> str:
    """Degree distribution with the fitted power law."""
    dd = read(run_dir, "degree_dist.csv")
    rep = report(run_dir)
    if dd is None:
        raise SystemExit("degree_dist.csv missing")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    k, pk = dd["k"].to_numpy(float), dd["pk"].to_numpy(float)
    m = (k > 0) & (pk > 0)
    ax.loglog(k[m], pk[m], "o", ms=4, color="#4c72b0", label="P(k)")
    g = rep["topology"].get("gamma")
    if m.sum() >= 2 and g and np.isfinite(g):
        kk = np.linspace(k[m].min(), k[m].max(), 50)
        c = pk[m][0] * k[m][0] ** g
        ax.loglog(kk, c * kk ** (-g), "r--", lw=1.2, label=f"fit $\\gamma$={g:.2f} (R²={rep['topology']['r2_powerlaw']:.2f})")
    ax.set_xlabel("degree k")
    ax.set_ylabel("P(k)")
    ax.set_title("Degree distribution")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, out)


def modules(run_dir: str, out: str) -> str:
    """Module sizes and the variance each eigengene explains."""
    mods = read(run_dir, "modules.csv")
    if mods is None:
        raise SystemExit("modules.csv missing")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.bar(mods["module"].astype(str), mods["size"], color="#4c72b0")
    ax.set_xlabel("module")
    ax.set_ylabel("OTUs")
    ax.set_title(f"Module sizes (n={len(mods)})")
    ax.tick_params(axis="x", labelsize=7)
    ax2.bar(mods["module"].astype(str), mods["var_explained"], color="#55a868")
    ax2.axhline(0.5, color="0.4", ls=":", lw=1)
    ax2.set_xlabel("module")
    ax2.set_ylabel("variance explained by PC1")
    ax2.set_ylim(0, 1)
    ax2.set_title("Eigengene strength")
    ax2.tick_params(axis="x", labelsize=7)
    return save(fig, out)


def bench(csv_path: str, out: str) -> str:
    """CPU stage scaling and GPU-vs-CPU similarity crossover."""
    b = pd.read_csv(csv_path)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    pipe = b[(b["stage"] == "pipeline") & (b["mode"] == "cpu")]
    if len(pipe):
        ax.loglog(pipe["n_otus"], pipe["seconds"], "o-", label="full pipeline")
    sim = b[(b["stage"] == "similarity") & (b["mode"] == "cpu")]
    if len(sim):
        ax.loglog(sim["n_otus"], sim["seconds"], "s--", label="similarity (CPU)")
    gpu = b[(b["stage"] == "similarity") & (b["mode"] == "gpu")]
    if len(gpu):
        ax.loglog(gpu["n_otus"], gpu["seconds"], "^-", label="similarity (GPU)")
    ax.set_xlabel("OTUs")
    ax.set_ylabel("seconds")
    ax.set_title("Scaling")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, which="both", alpha=0.2)
    if len(sim) and len(gpu):
        merged = sim.merge(gpu, on="n_otus", suffixes=("_cpu", "_gpu"))
        ax2.plot(merged["n_otus"], merged["seconds_cpu"] / merged["seconds_gpu"], "o-", color="#c44e52")
        ax2.axhline(1.0, color="0.4", ls="--", lw=1)
        ax2.set_xscale("log")
        ax2.set_xlabel("OTUs")
        ax2.set_ylabel("CPU / GPU time")
        ax2.set_title("GPU speedup (>1 = GPU faster)")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "no GPU rows (build --features gpu)", ha="center", va="center", fontsize=9)
    return save(fig, out)


ALL = ["fig5", "network", "roles", "rmt", "spectrum", "degrees", "modules"]


def main() -> int:
    ap = argparse.ArgumentParser(description="hydro-mena figures")
    ap.add_argument("figure", choices=ALL + ["bench", "all"])
    ap.add_argument("run_dir", help="directory containing report.json")
    ap.add_argument("--out", default="figures", help="output file, or directory for 'all'")
    args = ap.parse_args()
    if args.figure == "all":
        os.makedirs(args.out, exist_ok=True)
        for f in ALL:
            try:
                globals()[f](args.run_dir, os.path.join(args.out, f"{f}.png"))
            except SystemExit as exc:
                print(f"skip {f}: {exc}", file=sys.stderr)
        return 0
    globals()[args.figure](args.run_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
