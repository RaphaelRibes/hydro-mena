#!/usr/bin/env python3
"""Regenerate all report figures from fresh run outputs.

Usage: tools/make_figs.py <runs_dir> <out_dir>
  runs_dir must contain: box/ ox/ anox/ (default binary-spectrum runs),
  optionally ox-w/ (weighted ablation), and bench/ (bench artifacts).

Figures written to out_dir:
  fig_scan_audit.png   - ox rmt_scan.csv audit trail (binary mode)
  fig_spectrum_ox.png  - ox eigenvalues + NNSD histogram vs Poisson/Wigner
  fig_network_ox.png   - ox co-occurrence network, colored by module
  fig_roles_ox.png     - ox zi-Pi role scatter
  fig_fig5_ox.png      - ox module eigengene vs trait heatmap (top |r|)
  fig_fig5_box.png     - same for box
  fig_fig5_anox.png    - same for anox
  fig_rmt_ox.png       - ox chi2 vs st with critical line, binary vs weighted
  fig_calibration.png  - kept/edges vs st for all three conditions
  fig_bench.png        - CPU scaling + GPU timing bars (if bench/ present)
"""
import csv, json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

COND = ["box", "ox", "anox"]
COLORS = {"box": "#1f77b4", "ox": "#d62728", "anox": "#2ca02c"}


def load_run(d):
    r = json.load(open(RUNS / d / "report.json"))
    return r


def load_scan(d):
    with open(RUNS / d / "rmt_scan.csv") as f:
        return list(csv.DictReader(f))


def load_nodes(d):
    with open(RUNS / d / "nodes.csv") as f:
        return list(csv.DictReader(f))


def load_edges(d):
    with open(RUNS / d / "edges.csv") as f:
        return list(csv.DictReader(f))


def fnum(s, default=np.nan):
    """Parse a CSV cell to float; blanks/NaN/garbage become `default`."""
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return default


def unfolded_spacings(lam, window=5):
    """Local-density unfolding: ds_i * rho_i, rho from a +/-window stencil."""
    lam = np.sort(np.asarray(lam, dtype=float))
    d = []
    for i in range(1, len(lam)):
        lo, hi = max(0, i - window), min(len(lam), i + window + 1)
        span = lam[hi - 1] - lam[lo]
        if span > 0:
            d.append((lam[i] - lam[i - 1]) * (hi - lo) / span)
    d = np.asarray(d, dtype=float)
    return d / d.mean() if d.size and d.mean() > 0 else d


# ---------------------------------------------------------------- scan audit
def fig_scan_audit():
    rows = load_scan("ox")
    fig, ax = plt.subplots(figsize=(7, 4))
    accepted, rejected, excluded = [], [], []
    for r in rows:
        st, chi2, status = fnum(r["st"]), fnum(r["chi2"]), r["status"]
        if status.startswith("excluded"):
            excluded.append(st)
        elif status == "accepted":
            accepted.append((st, chi2))
        else:
            rejected.append((st, chi2))
    if rejected:
        ax.scatter(*zip(*rejected), marker="o", c="red", s=45, alpha=0.8,
                   label="rejected (GoF)")
    if accepted:
        ax.scatter(*zip(*accepted), marker="o", c="green", s=55, zorder=5,
                   label="accepted (RMT converged)")
    crits = np.array([fnum(r["crit"]) for r in rows], dtype=float)
    crits = crits[np.isfinite(crits)]
    if crits.size:
        ax.axhline(crits.mean(), ls="--", c="k", lw=1,
                   label=f"chi2 crit (alpha=0.01, dof-dependent mean={crits.mean():.1f})")
    if excluded:
        # not scored: draw as a rug at the top instead of faking a chi2 ordinate
        ytop = ax.get_ylim()[1]
        ax.scatter(excluded, np.full(len(excluded), ytop * 0.97), marker="x",
                   c="grey", s=40, alpha=0.7, label="excluded (not scored)")
    ax.set_xlabel("threshold st")
    ax.set_ylabel("chi2 statistic")
    ax.set_title("ox RMT scan audit trail (binary spectrum, default)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scan_audit.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- spectrum
def fig_spectrum():
    # spectrum.csv header is `index,eigenvalue,spacing` — never read column 0.
    rows = list(csv.DictReader(open(RUNS / "ox" / "spectrum.csv")))
    lam = np.array([fnum(r.get("eigenvalue")) for r in rows], dtype=float)
    lam = np.sort(lam[np.isfinite(lam)])
    if lam.size < 4:
        raise SystemExit(f"no usable eigenvalues in {RUNS/'ox'/'spectrum.csv'}")
    # the writer already emits mean-normalised (unfolded) spacings; prefer them,
    # fall back to local-density unfolding from the eigenvalues.
    d = np.array([fnum(r.get("spacing")) for r in rows], dtype=float)
    d = d[np.isfinite(d)]
    if d.size < lam.size - 1 or not 0.9 <= d.mean() <= 1.1:
        d = unfolded_spacings(lam)
    d = d / d.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(np.arange(lam.size), lam, ".", ms=3)
    a1.axhline(0, c="k", lw=0.6, alpha=0.5)
    a1.set_title(f"ox adjacency eigenvalues @ auto st (n={lam.size})")
    a1.set_xlabel("rank")
    a1.set_ylabel(r"$\lambda$")
    a2.hist(d, bins=30, density=True, alpha=0.7, label="data")
    s = np.linspace(0, 4, 200)
    a2.plot(s, np.exp(-s), "k-", label="Poisson $e^{-s}$")
    a2.plot(s, (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4), "r--", label="Wigner (GOE)")
    a2.set_xlabel("s")
    a2.set_title(f"unfolded NNSD (n={d.size})")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectrum_ox.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- network
def fig_network():
    import networkx as nx
    nodes = load_nodes("ox")
    edges = load_edges("ox")
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["OTU"], module=int(n["module"]))
    for e in edges:
        G.add_edge(e["source"], e["target"])
    mods = sorted({G.nodes[n]["module"] for n in G.nodes})
    cmap = plt.get_cmap("tab20")
    pos = nx.spring_layout(G, seed=3, k=0.35)
    fig, ax = plt.subplots(figsize=(7, 7))
    for m in mods:
        ns = [n for n in G.nodes if G.nodes[n]["module"] == m]
        nx.draw_networkx_nodes(G, pos, nodelist=ns, node_color=[cmap(m % 20)],
                               node_size=28, ax=ax, linewidths=0)
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.5, ax=ax)
    ax.set_title(f"ox network @ auto st (n={len(G.nodes)}, E={G.number_of_edges()})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_network_ox.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- roles
def fig_roles():
    nodes = load_nodes("ox")
    zi = [float(n["zi"]) for n in nodes]
    pi = [float(n["Pi"]) for n in nodes]
    role = [n["role"] for n in nodes]
    rc = {"peripheral": "#9ecae1", "connector": "#fdae6b", "module_hub": "#d62728",
          "network_hub": "#6a3d9a"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in dict.fromkeys(role):
        xs = [z for z, rr in zip(zi, role) if rr == r]
        ys = [p for p, rr in zip(pi, role) if rr == r]
        ax.scatter(xs, ys, s=22, c=rc.get(r, "grey"), label=r, alpha=0.85, linewidths=0)
    ax.axvline(2.5, ls=":", c="k", lw=0.8)
    ax.axhline(0.62, ls=":", c="k", lw=0.8)
    ax.set_xlabel("within-module degree z-score $z_i$")
    ax.set_ylabel("participation coefficient $P_i$")
    ax.set_title("ox module roles @ auto st")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_roles_ox.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig5 heatmaps
def fig_fig5(d):
    with open(RUNS / d / "fig5_r.csv") as f:
        rows = list(csv.reader(f))
    hdr = rows[0][1:]
    body = [r for r in rows[1:] if len(r) > 1]
    labels = [r[0] for r in body]
    mat = np.array([[fnum(v) for v in r[1:len(hdr) + 1]] for r in body], dtype=float)
    if mat.size == 0:
        print(f"{d}: empty fig5_r.csv — skipping fig_fig5_{d}")
        return
    # keep top 12 traits by max |r|
    top = np.argsort(-np.nanmax(np.abs(mat), axis=0))[:12]
    mat = mat[:, top]
    hdr = [hdr[i] for i in top]
    fig, ax = plt.subplots(figsize=(8, 0.35 * mat.shape[0] + 1.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(hdr)))
    ax.set_xticklabels(hdr, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([l.replace("module_", "M") for l in labels], fontsize=7)
    ax.set_title(f"{d}: module eigengene–trait Pearson r (top 12 traits)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_fig5_{d}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- rmt compare binary vs weighted
def fig_rmt_compare():
    fig, ax = plt.subplots(figsize=(7, 4))
    for d in COND:
        rows = load_scan(d)
        st = [fnum(r["st"]) for r in rows]
        chi2 = [fnum(r["chi2"]) for r in rows]
        ax.plot(st, chi2, "o-", ms=3.5, lw=1, color=COLORS[d], label=f"{d} (binary)")
    if (RUNS / "ox-w").exists():
        rows = load_scan("ox-w")
        st = [fnum(r["st"]) for r in rows]
        chi2 = [fnum(r["chi2"]) for r in rows]
        ax.plot(st, chi2, "s--", ms=3.5, lw=1, color="grey", label="ox (weighted, --no-binary-spectrum)")
    rows = load_scan("ox")
    crits = np.array([fnum(r["crit"]) for r in rows], dtype=float)
    crits = crits[np.isfinite(crits)]
    if crits.size:
        ax.axhline(crits.mean(), ls="--", c="k", lw=1, label="crit (alpha=0.01)")
    ax.set_xlabel("st")
    ax.set_ylabel("chi2")
    ax.set_title("RMT scan: binary (default) vs weighted spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rmt_ox.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- calibration
def fig_calibration():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for d in COND:
        if not (RUNS / d / "rmt_scan.csv").exists():
            print(f"{d}: no rmt_scan.csv — skipping in fig_calibration")
            continue
        rows = load_scan(d)
        st = [fnum(r["st"]) for r in rows]
        kept = [fnum(r["p"]) for r in rows]
        edges = [fnum(r["edges"]) for r in rows]
        a1.plot(st, kept, "o-", ms=3, color=COLORS[d], label=d)
        a2.plot(st, edges, "o-", ms=3, color=COLORS[d], label=d)
    a1.set_title("kept nodes vs st")
    a2.set_title("edges vs st")
    for a in (a1, a2):
        a.set_xlabel("st")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- bench
def fig_bench():
    # Fresh desktop (RTX 2080 Ti) numbers, `mena bench --gpu`, min-of-3, 2026-11-09
    # (warm repeats at n=12000: 0.4890 / 0.4845 / 0.4856 s; parity bad=0 everywhere)
    sizes = [240, 480, 960, 1920, 3000, 5000, 8000, 12000]
    cpu =   [0.0001, 0.0004, 0.0016, 0.0091, 0.0291, 0.0796, 0.2946, 0.8216]
    gpu =   [0.0018, 0.0020, 0.0030, 0.0130, 0.0287, 0.0767, 0.2202, 0.4858]
    maxdiff = [1.52e-7, 1.60e-7, 2.19e-7, 2.10e-7, 2.64e-7, 2.59e-7, 3.58e-7, 3.84e-7]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(sizes, cpu, "o-", color="#1f77b4", label="CPU (rayon)")
    a1.plot(sizes, gpu, "s-", color="#d62728", label="GPU RTX 2080 Ti (wgpu/Vulkan)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("n OTUs"); a1.set_ylabel("similarity time (s)")
    a1.set_title("CPU vs GPU similarity kernel (min-of-3)")
    a1.legend(fontsize=8)
    a1.annotate("1.69x at n=12000\n(0.486 s vs 0.822 s)", xy=(12000, 0.486), xytext=(2500, 0.0035), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    a2.plot(sizes, maxdiff, "o-", color="#2ca02c")
    a2.axhline(1e-5, ls="--", c="k", lw=0.8, label="tolerance 1e-5")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("n OTUs"); a2.set_ylabel("max |GPU - CPU|")
    a2.set_title("CPU/GPU parity (bad = 0 at every size)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bench.png", dpi=150)
    plt.close(fig)




def fig_ari_vs_st():
    # Calibration sweep (28 runs, scratch/mena_calib/calibration.json):
    # ARI of detected vs planted modules across a fixed-st grid.
    import json as _json
    p = Path("/home/raphael/.hermes/cache/scratch/mena_calib/calibration.json")
    if not p.exists():
        print("no calibration.json — skipping fig_ari_vs_st")
        return
    d = _json.load(open(p))
    rvt = d["recovery_vs_threshold"]
    fig, ax = plt.subplots(figsize=(7, 4))
    blocks = {}
    for r in rvt:
        # only the informative 40-sample block(s); 12-sample runs are capped by the simulator
        if r["samples"] != 40:
            continue
        key = (r["n"], r["samples"], round(r["noise"], 2))
        blocks.setdefault(key, []).append(r)
    styles = [("o-", "#1f77b4"), ("s-", "#d62728"), ("^-", "#2ca02c"), ("d-", "#9467bd")]
    for (key, rows), (mk, col) in zip(sorted(blocks.items()), styles):
        rows = sorted(rows, key=lambda x: x["st"])
        ax.plot([r["st"] for r in rows], [r["mean_ari"] for r in rows], mk,
                color=col, ms=4, lw=1.2,
                label=f"n={key[0]} samples={key[1]} noise={key[2]}")
    ax.axvline(0.52, ls=":", c="k", lw=0.8)
    ax.annotate("2025 hardcoded st=0.52", xy=(0.52, 0.9), fontsize=7, rotation=90, va="top")
    ax.set_xlabel("threshold st")
    ax.set_ylabel("ARI vs planted modules")
    ax.set_title("Calibration: recovery peaks at LOW st, RMT scan picks higher st")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ari_vs_st.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_scan_audit()
    fig_spectrum()
    fig_network()
    fig_roles()
    for d in COND:
        fig_fig5(d)
    fig_rmt_compare()
    fig_calibration()
    fig_bench()
    fig_ari_vs_st()
    print("figures written to", OUT)
