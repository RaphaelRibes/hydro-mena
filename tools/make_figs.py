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


# ---------------------------------------------------------------- scan audit
def fig_scan_audit():
    rows = load_scan("ox")
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in rows:
        st, chi2, crit, status = float(r["st"]), float(r["chi2"]), float(r["crit"]), r["status"]
        if status.startswith("excluded"):
            ax.scatter(st, 60, marker="x", c="grey", s=40, alpha=0.7)
        elif status == "accepted":
            ax.scatter(st, chi2, marker="o", c="green", s=55, zorder=5)
        else:
            ax.scatter(st, chi2, marker="o", c="red", s=45, alpha=0.8)
    crits = [float(r["crit"]) for r in rows if r["crit"] not in ("nan", "NaN")]
    if crits:
        ax.axhline(np.nanmean(crits), ls="--", c="k", lw=1, label="chi2 crit (alpha=0.01)")
    ax.set_xlabel("threshold st")
    ax.set_ylabel("chi2 statistic")
    ax.set_title("ox RMT scan audit trail (binary spectrum, default)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_scan_audit.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- spectrum
def fig_spectrum():
    lam = []
    with open(RUNS / "ox" / "spectrum.csv") as f:
        for row in csv.reader(f):
            if row and row[0] not in ("lambda",):
                try:
                    lam.append(float(row[0]))
                except ValueError:
                    pass
    lam = np.array(sorted(lam))
    lam = lam[1:-1] if len(lam) > 12 else lam
    # local-density unfolding (window 5)
    d = []
    for i in range(1, len(lam)):
        lo, hi = max(0, i - 5), min(len(lam), i + 6)
        if lam[hi - 1] > lam[lo]:
            rho = (hi - lo) / (lam[hi - 1] - lam[lo])
            d.append((lam[i] - lam[i - 1]) * rho)
    d = np.array(d)
    d = d / d.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(lam, ".", ms=3)
    a1.set_title("ox adjacency eigenvalues @ auto st")
    a1.set_xlabel("rank")
    a2.hist(d, bins=30, density=True, alpha=0.7, label="data")
    s = np.linspace(0, 4, 200)
    a2.plot(s, np.exp(-s), "k-", label="Poisson $e^{-s}$")
    a2.plot(s, (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4), "r--", label="Wigner")
    a2.set_title("unfolded NNSD")
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
    mat = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    # keep top 12 traits by max |r|
    top = np.argsort(-np.abs(mat).max(axis=0))[:12]
    mat = mat[:, top]
    hdr = [hdr[i] for i in top]
    fig, ax = plt.subplots(figsize=(8, 0.35 * mat.shape[0] + 1.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(hdr)))
    ax.set_xticklabels(hdr, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"M{i+1}" for i in range(mat.shape[0])], fontsize=7)
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
        st = [float(r["st"]) for r in rows]
        chi2 = [float(r["chi2"]) if r["chi2"] not in ("nan", "NaN") else np.nan for r in rows]
        ax.plot(st, chi2, "o-", ms=3.5, lw=1, color=COLORS[d], label=f"{d} (binary)")
    if (RUNS / "ox-w").exists():
        rows = load_scan("ox-w")
        st = [float(r["st"]) for r in rows]
        chi2 = [float(r["chi2"]) if r["chi2"] not in ("nan", "NaN") else np.nan for r in rows]
        ax.plot(st, chi2, "s--", ms=3.5, lw=1, color="grey", label="ox (weighted, --no-binary-spectrum)")
    rows = load_scan("ox")
    crits = [float(r["crit"]) for r in rows if r["crit"] not in ("nan", "NaN")]
    if crits:
        ax.axhline(np.nanmean(crits), ls="--", c="k", lw=1, label="crit (alpha=0.01)")
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
        rows = load_scan(d)
        st = [float(r["st"]) for r in rows]
        kept = [int(r["p"]) for r in rows]
        edges = [int(r["edges"]) for r in rows]
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
    bench = RUNS / "bench"
    if not (bench / "report.json").exists():
        print("no bench/ — skipping fig_bench")
        return
    r = json.load(open(bench / "report.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    # expect keys like cpu_n and gpu_n timings; adapt to actual schema
    keys = list(r.keys())
    print("bench keys:", keys[:10])
    # fallback: look for timing fields
    rows = r.get("rows") or r.get("runs") or []
    if rows:
        ns = [row.get("n", 0) for row in rows]
        cpu = [row.get("cpu_s", np.nan) for row in rows]
        gpu = [row.get("gpu_s", np.nan) for row in rows]
        ax.plot(ns, cpu, "o-", label="CPU", color="#1f77b4")
        ax.plot(ns, gpu, "s-", label="GPU (warm)", color="#d62728")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("similarity-matrix time (s)")
        ax.legend()
        ax.set_title("CPU vs GPU similarity kernel")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bench.png", dpi=150)
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
    print("figures written to", OUT)
