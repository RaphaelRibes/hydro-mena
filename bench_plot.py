#!/usr/bin/env python3
"""Plot hydro-mena benchmark CSV (graphs only)."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    pipe = df[(df.stage == "pipeline") & (df.mode == "cpu")]
    ax.loglog(pipe.n_otus, pipe.seconds, "o-", label="full pipeline (cpu)")
    sim = df[(df.stage == "similarity") & (df.mode == "cpu") & (df.name.str.startswith("stage"))]
    ax.loglog(sim.n_otus, sim.seconds, "s--", label="similarity stage")
    topo = df[df.stage == "topology"]
    ax.loglog(topo.n_otus, topo.seconds, "^--", label="topology stage")
    ax.set_xlabel("n OTUs")
    ax.set_ylabel("seconds (min of 3)")
    ax.set_title("hydro-mena CPU pipeline scaling (16 cores rayon)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax = axes[1]
    g = df[(df.stage == "similarity") & (df.name.str.startswith("similarity"))]
    cpu = g[g["mode"] == "cpu"].set_index("n_otus")
    gpu = g[g["mode"] == "gpu"].set_index("n_otus")
    ns = sorted(set(cpu.index) & set(gpu.index))
    ax.loglog(ns, [cpu.loc[n].seconds for n in ns], "o-", label="CPU rayon")
    ax.loglog(ns, [gpu.loc[n].seconds for n in ns], "s-", label="GPU wgpu/Vulkan (RTX 2080 Ti)")
    for n in ns:
        sp = cpu.loc[n].seconds / gpu.loc[n].seconds
        ax.annotate(f"{sp:.2f}x", (n, gpu.loc[n].seconds), textcoords="offset points", xytext=(0, -12), fontsize=8)
    ax.set_xlabel("n OTUs")
    ax.set_ylabel("seconds (min of 3)")
    ax.set_title("similarity matrix: GPU vs CPU (maxdiff <= 5e-7 = parity PASS)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
