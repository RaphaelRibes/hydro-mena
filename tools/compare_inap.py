#!/usr/bin/env python3
"""compare_inap.py — compare a mena network with an iNAP 2.0 network.

The two tools write different artifacts.  This script normalises both sides to
a symmetric, unweighted node x node adjacency and reports:

  * edge overlap against the iNAP graph as reference
    (precision = fraction of mena edges also in iNAP,
     recall    = fraction of iNAP edges also in mena,
     F1        = harmonic mean),
  * Newman modularity Q of each side's own partition (module table), computed
    with the aggregate formula Q = sum_c [L_c/m - (d_c/2m)^2].

iNAP 2.0 is a Galaxy web server (https://inap.denglab.org.cn), not a pip
package.  Export its network matrix / edge list and module table into one
directory; this script auto-detects the files.  See README.md, section
"iNAP 2.0 comparison".

No third-party dependency is required.  If networkx is importable, Louvain is
used when a side has no module table; otherwise a deterministic greedy
modularity merge provides the partition (the Q computation itself is always
the built-in formula above).

Usage
-----
    python3 compare_inap.py --mena real/review_ox/s52 --inap ~/inap_ox
    python3 compare_inap.py --mena real/review_ox/s52 --repo-2025
    python3 compare_inap.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# tabular input
# --------------------------------------------------------------------------


def _sniff_delimiter(line: str) -> str:
    """Pick the delimiter that occurs most often outside the label text."""
    counts = {";": line.count(";"), "\t": line.count("\t"), ",": line.count(",")}
    delim = max(counts, key=counts.get)
    return delim if counts[delim] else ","


def read_table(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: empty table")
    delim = _sniff_delimiter(lines[0])
    rows = [[c.strip() for c in row] for row in csv.reader(lines, delimiter=delim)]
    return [row for row in rows if any(c for c in row)]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _looks_like_matrix(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    ncol = len(rows[0])
    if ncol < 3 or len(rows) < 2:
        return False
    # header row: labels; data rows: one label + ncol-1 numbers
    if len(rows) - 1 != ncol - 1:
        return False
    numbers = 0
    total = 0
    for row in rows[1:]:
        if len(row) != ncol:
            return False
        for cell in row[1:]:
            total += 1
            numbers += _is_number(cell)
    return total > 0 and numbers / total > 0.9


def parse_graph_file(path: Path):
    """Return (nodes, edges, kind, detail).

    Supports:
      * square adjacency matrix, header row + label column (mena, 2025 repo,
        iNAP network matrix exports);
      * square matrix with labels in both header and first column but no
        corner label;
      * 2/3+-column edge list (iNAP SIF/tabular exports).
    """
    rows = read_table(path)
    # Galaxy occasionally prefixes an edge list with "nodes edges".
    if len(rows[0]) == 1 and _is_number(rows[0][0]) and len(rows) > 1:
        rows = rows[1:]

    if _looks_like_matrix(rows):
        header = rows[0]
        col_labels = header[1:]
        row_labels = [r[0] for r in rows[1:]]
        # Some exports repeat the first label in the corner cell.
        if header[0] and header[0] not in ("OTU", "Node", "node", "ID", "id", ""):
            if len(rows) == len(header) and all(
                len(r) == len(header) and r[0] == header[i] for i, r in enumerate(rows)
            ):
                # labels embedded in both axes, no separate corner cell
                col_labels = header
                row_labels = [r[0] for r in rows]
                rows_data = rows
            else:
                rows_data = rows[1:]
        else:
            rows_data = rows[1:]

        nodes = list(dict.fromkeys(row_labels + col_labels))
        idx_col = {n: i for i, n in enumerate(col_labels)}
        idx_row = {n: i for i, n in enumerate(row_labels)}
        edges = set()
        weights = 0
        for rlab, row in zip(row_labels, rows_data):
            for clab, cell in zip(col_labels, row[1:] if len(row) == len(col_labels) + 1 else row):
                if rlab == clab or not _is_number(cell):
                    continue
                if float(cell) != 0.0:
                    edges.add(tuple(sorted((rlab, clab))))
                    weights += 1
        return nodes, edges, "matrix", f"{len(nodes)}x{len(nodes)}, {weights // 2} edges"

    # edge list
    body = rows
    if body and len(body[0]) >= 3 and not _is_number(body[0][2]):
        body = body[1:]  # header
    edges = set()
    nodes = []
    for row in body:
        if len(row) < 2 or not row[0] or not row[1]:
            continue
        if _is_number(row[0]) and _is_number(row[1]):
            continue  # index-only row
        weight = 1.0
        if len(row) >= 3 and _is_number(row[2]):
            weight = float(row[2])
        if weight == 0.0:
            continue
        u, v = row[0], row[1]
        if u == v:
            continue
        edges.add(tuple(sorted((u, v))))
        nodes.extend([u, v])
    nodes = list(dict.fromkeys(nodes))
    return nodes, edges, "edge-list", f"{len(nodes)} nodes, {len(edges)} edges"


def _score_candidate(path: Path) -> int:
    n = path.name.lower()
    if any(x in n for x in ("module", "nodes", "node_attr", "report", "rmt_scan", "spectrum",
                            "degree", "eigengene", "fig5", "pvalue", "p_value")):
        return -1
    score = 0
    if "adj" in n:
        score += 30
    if "network" in n or "net" in n:
        score += 20
    if "edge" in n or "sif" in n:
        score += 15
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        score += 5
    return score


def load_graph_dir(path: Path):
    """Load the first parseable network artifact in a directory."""
    if path.is_file():
        nodes, edges, kind, detail = parse_graph_file(path)
        return nodes, edges, kind, detail, path
    candidates = []
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".txt"):
            s = _score_candidate(p)
            if s >= 0:
                candidates.append((s, p.name, p))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    errors = []
    for _, _, p in candidates:
        try:
            nodes, edges, kind, detail = parse_graph_file(p)
        except Exception as exc:  # noqa: BLE001 - report and try the next file
            errors.append(f"{p.name}: {exc}")
            continue
        if len(nodes) >= 2 and edges:
            return nodes, edges, kind, detail, p
        errors.append(f"{p.name}: parsed but has {len(nodes)} nodes / {len(edges)} edges")
    raise SystemExit(
        f"could not find a usable network file in {path}\n  tried: "
        + ("; ".join(errors) if errors else "no .csv/.tsv/.txt candidates")
    )


def load_modules(path: Path, graph_nodes: list[str] | None = None) -> dict[str, str]:
    """Read node -> module id from a module/node table."""
    rows = read_table(path)
    if not rows:
        return {}
    header = [h.strip().lower() for h in rows[0]]
    module_col = None
    for key in ("module", "modules", "community", "cluster", "modularity_class"):
        if key in header:
            module_col = header.index(key)
            break
    node_col = 0
    for key in ("otu", "node", "id", "name", "species"):
        if key in header:
            node_col = header.index(key)
            break
    if module_col is None:
        # No header: first column node, second column module.
        module_col = 1 if len(rows[0]) > 1 else None
    if module_col is None or len(rows[0]) <= module_col:
        return {}
    if node_col == module_col:
        return {}  # a module summary (module,size,...), not a node assignment
    out = {}
    for row in rows[1:]:
        if len(row) <= max(node_col, module_col):
            continue
        node, mod = row[node_col], row[module_col]
        if node and mod:
            out[node] = mod
    return out


def find_partition_file(directory: Path) -> Path | None:
    """Find a node-level module table, preferring nodes.* over modules.*."""
    candidates = []
    for p in sorted(directory.iterdir()):
        if not (p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".txt")):
            continue
        n = p.name.lower()
        score = -1
        if n in ("nodes.csv", "node.csv"):
            score = 40
        elif "node" in n or "attribute" in n:
            score = 30
        elif "module" in n:
            score = 20
        elif "cluster" in n:
            score = 10
        if score >= 0:
            candidates.append((score, p))
    candidates.sort(key=lambda x: (-x[0], x[1].name))
    for _, p in candidates:
        if load_modules(p):
            return p
    return None


# --------------------------------------------------------------------------
# graphs, metrics, modularity
# --------------------------------------------------------------------------


def make_graph(nodes: list[str], edges: set[tuple[str, str]]):
    g = {n: set() for n in nodes}
    for u, v in edges:
        if u not in g:
            g[u] = set()
        if v not in g:
            g[v] = set()
        g[u].add(v)
        g[v].add(u)
    return g


def label_map(g: dict[str, set[str]], mode: str) -> dict[str, str]:
    """Map each node to its comparison label."""
    if mode != "last":
        return {n: n for n in g}
    mapping: dict[str, str] = {}
    seen: dict[str, set[str]] = {}
    for n in g:
        short = n.split("|")[-1]
        seen.setdefault(short, set()).add(n)
        mapping[n] = short
    # Ambiguous short names (e.g. several "uncultured") keep their full label.
    for short, fulls in seen.items():
        if len(fulls) > 1:
            for f in fulls:
                mapping[f] = f
    return mapping


def apply_label_map(g: dict[str, set[str]], mapping: dict[str, str]):
    out: dict[str, set[str]] = {}
    for n, neighbours in g.items():
        u = mapping.get(n, n)
        out.setdefault(u, set())
        for v in neighbours:
            w = mapping.get(v, v)
            if u != w:
                out[u].add(w)
                out.setdefault(w, set()).add(u)
    return out


def normalize_labels(g: dict[str, set[str]], mode: str):
    """Collapse taxonomy strings to their terminal name when `mode == last`."""
    return apply_label_map(g, label_map(g, mode))


def edges_of(g: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {tuple(sorted((u, v))) for u, ns in g.items() for v in ns if u < v}


def modularity(g: dict[str, set[str]], partition: dict[str, str]) -> float:
    """Newman modularity of a given partition on an undirected binary graph."""
    m = len(edges_of(g))
    if m == 0:
        return float("nan")
    comm_edges: dict[str, int] = {}
    comm_deg: dict[str, int] = {}
    for n, ns in g.items():
        c = partition.get(n, f"__{n}")
        comm_deg[c] = comm_deg.get(c, 0) + len(ns)
    for u, v in edges_of(g):
        cu, cv = partition.get(u, f"__{u}"), partition.get(v, f"__{v}")
        if cu == cv:
            comm_edges[cu] = comm_edges.get(cu, 0) + 1
    q = 0.0
    for c, deg in comm_deg.items():
        q += comm_edges.get(c, 0) / m - (deg / (2.0 * m)) ** 2
    return q


def greedy_partition(g: dict[str, set[str]]) -> dict[str, str]:
    """Deterministic greedy modularity merging (Louvain-free fallback)."""
    nodes = sorted(g)
    m = len(edges_of(g))
    if m == 0:
        return {n: n for n in nodes}
    comm = {n: n for n in nodes}
    members = {n: {n} for n in nodes}
    deg = {n: len(g[n]) for n in nodes}
    # community-level edge counts, rebuilt after each merge (n <= a few hundred)
    while True:
        best = None
        # count inter-community edges
        inter: dict[tuple[str, str], int] = {}
        for u, v in edges_of(g):
            cu, cv = comm[u], comm[v]
            if cu != cv:
                key = tuple(sorted((cu, cv)))
                inter[key] = inter.get(key, 0) + 1
        for (cu, cv), l in inter.items():
            dq = l / m - (deg[cu] * deg[cv]) / (2.0 * m * m)
            if dq > 1e-12 and (best is None or dq > best[0] + 1e-15):
                best = (dq, cu, cv)
        if best is None:
            break
        _, cu, cv = best
        keep, drop = sorted((cu, cv))
        for n in members[drop]:
            comm[n] = keep
        members[keep] |= members[drop]
        deg[keep] += deg[drop]
        del members[drop]
        del deg[drop]
    return dict(comm)


def derive_partition(g: dict[str, set[str]], given: dict[str, str] | None = None):
    """Return (partition, source label, n_modules)."""
    if given:
        part = {n: str(given.get(n, f"__{n}")) for n in g}
        return part, "module table", len(set(part.values()))
    try:
        import networkx as nx  # type: ignore

        G = nx.Graph()
        G.add_nodes_from(g)
        G.add_edges_from(edges_of(g))
        comms = nx.community.louvain_communities(G, seed=42)
        part = {}
        for i, c in enumerate(comms):
            for n in c:
                part[n] = str(i)
        return part, "louvain(networkx)", len(comms)
    except Exception:  # noqa: BLE001 - any failure falls back to greedy
        part = greedy_partition(g)
        return part, "greedy(builtin)", len(set(part.values()))


def edge_metrics(ref: dict[str, set[str]], pred: dict[str, set[str]]):
    common = sorted(set(ref) & set(pred))
    er = {e for e in edges_of(ref) if e[0] in common and e[1] in common}
    ep = {e for e in edges_of(pred) if e[0] in common and e[1] in common}
    tp = len(er & ep)
    precision = tp / len(ep) if ep else float("nan")
    recall = tp / len(er) if er else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and precision + recall > 0 else float("nan"))
    return {
        "common_nodes": len(common),
        "ref_edges_common": len(er),
        "pred_edges_common": len(ep),
        "tp": tp,
        "fp": len(ep - er),
        "fn": len(er - ep),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------------------------------
# comparison driver
# --------------------------------------------------------------------------


def load_mena(directory: Path):
    adj = directory / "adjacency.csv"
    if not adj.exists():
        # fall back to auto-detection (e.g. iNAP exports stored in a mena-like dir)
        nodes, edges, kind, detail, used = load_graph_dir(directory)
        g = make_graph(nodes, edges)
    else:
        nodes, edges, kind, detail = parse_graph_file(adj)
        g = make_graph(nodes, edges)
        used = adj
    part = {}
    nodes_csv = directory / "nodes.csv"
    if nodes_csv.exists():
        part = load_modules(nodes_csv)
    return g, part, used


def print_side(name: str, g: dict[str, set[str]], part: dict[str, str], source, used):
    q, qsrc, nmod = derive_partition(g, part or None)
    print(f"  {name:<10} nodes={len(g):<5} edges={len(edges_of(g)):<6} "
          f"Q={'nan' if math.isnan(modularity(g, q)) else f'{modularity(g, q):.4f}'} "
          f"({qsrc}, {nmod} modules)")
    print(f"  {'':<10} source: {used}")
    return q, qsrc, nmod


def compare(mena_dir: Path, other_dir: Path | None, other_file: Path | None,
            label: str, normalize: str = "auto"):
    gm, pm, mpath = load_mena(mena_dir)
    if other_file is not None:
        nodes, edges, kind, detail = parse_graph_file(other_file)
        go = make_graph(nodes, edges)
        opath = other_file
        po = {}
        if other_file.parent.is_dir():
            mp = find_partition_file(other_file.parent)
            if mp is not None and mp != other_file:
                po = load_modules(mp)
    elif other_dir is not None:
        nodes, edges, kind, detail, opath = load_graph_dir(other_dir)
        go = make_graph(nodes, edges)
        po = {}
        mp = find_partition_file(other_dir)
        if mp is not None:
            po = load_modules(mp)
    else:
        raise SystemExit("internal: no comparison target")

    # Label normalisation.  Try exact first; if very little overlaps, try the
    # terminal taxonomy name (iNAP/legacy exports often drop the full path).
    modes = [normalize] if normalize != "auto" else ["exact", "last"]
    best = None
    for mode in modes:
        a = normalize_labels(gm, mode) if mode == "last" else gm
        b = normalize_labels(go, mode) if mode == "last" else go
        overlap = len(set(a) & set(b))
        score = (overlap, mode == "exact")
        if best is None or score > best[0]:
            best = (score, mode, a, b)
    _, mode, gm_n, go_n = best
    # Keep partitions aligned with the comparison labels.
    pm_n = {label_map(gm, mode).get(k, k): v for k, v in pm.items()} if mode != "exact" else pm
    po_n = {label_map(go, mode).get(k, k): v for k, v in po.items()} if mode != "exact" else po
    m = edge_metrics(go_n, gm_n)  # iNAP/repo is the reference

    print(f"\n=== mena vs {label} ===")
    print(f"  label normalisation: {mode}")
    print_side("mena", gm_n, pm_n, "nodes.csv", mpath)
    print_side(label, go_n, po_n, "module table", opath)
    print(f"\n  common nodes: {m['common_nodes']}  (mena {len(gm_n)}, {label} {len(go_n)})")
    print(f"  edges on common nodes: mena {m['pred_edges_common']}, {label} {m['ref_edges_common']}")
    print(f"  TP={m['tp']}  FP(mena-only)={m['fp']}  FN({label}-only)={m['fn']}")
    print(f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  F1={m['f1']:.4f}")
    print("  (precision = mena edges also in the reference; recall = reference edges recovered by mena)")
    m["label_mode"] = mode
    m["mena_dir"] = str(mena_dir)
    m["reference"] = str(opath)
    m["label"] = label
    return m


def find_2025_repo(root: Path) -> Path | None:
    candidates = [
        root / "data" / "adjacency_matrix_ox.csv",
        root / "adjacency_matrix_ox.csv",
        root / "data" / "clean_adjacency_matrix_ox.csv",
        root / "clean_adjacency_matrix_ox.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def default_mena_dir() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [
        Path.home() / "hermes_scratch" / "hydro-mena" / "real" / "review_ox" / "s52",
        Path("/home/raphael/.hermes/cache/scratch/inap-check/ox-clr-pc0001-auto"),
        here / "real" / "review_ox" / "s52",
    ]
    for p in candidates:
        if (p / "adjacency.csv").is_file():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare a mena network against an iNAP 2.0 / legacy network "
                    "(edge precision/recall/F1 and modularity).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mena", type=Path, help="mena out-dir (contains adjacency.csv, nodes.csv)")
    ap.add_argument("--inap", type=Path, help="iNAP output dir (network matrix / edge list, optional module table)")
    ap.add_argument("--inap-file", type=Path, help="single iNAP network file instead of a directory")
    ap.add_argument("--repo-2025", nargs="?", const="auto", default=None,
                    help="compare against the 2025 repo adjacency (default root: "
                         "~/hermes_scratch/hydro-2025)")
    ap.add_argument("--normalize", choices=["auto", "exact", "last"], default="auto",
                    help="node label mapping; 'last' keeps only the terminal taxonomy name")
    ap.add_argument("--self-test", action="store_true",
                    help="compare the mena ox run against itself (F1 must be 1.0) and, "
                         "when present, against the 2025 repo adjacency")
    ap.add_argument("--json", type=Path, help="write metrics to a JSON file")
    args = ap.parse_args(argv)

    mena_dir = args.mena or default_mena_dir()
    results = []

    if args.self_test:
        if mena_dir is None:
            print("SELF-TEST FAILED: no mena run found; pass --mena <out-dir>", file=sys.stderr)
            return 2
        print(f"[self-test] mena out-dir: {mena_dir}")
        m = compare(mena_dir, mena_dir, None, "mena (self)")
        results.append(m)
        ok = abs(m["f1"] - 1.0) < 1e-12
        print(f"\nSELF-TEST {'PASS' if ok else 'FAIL'}: F1={m['f1']:.12f} (expected 1.0)")
        if not ok:
            return 1

        repo_root = Path.home() / "hermes_scratch" / "hydro-2025"
        adj = find_2025_repo(repo_root)
        if adj is None:
            print(f"\n[2025 repo] no adjacency_matrix_ox.csv under {repo_root} — skipped")
        else:
            print(f"\n[2025 repo] using {adj}")
            r = compare(mena_dir, None, adj, "2025 repo")
            results.append(r)

    else:
        if mena_dir is None:
            ap.error("--mena is required (or use --self-test)")
        if not args.inap and not args.inap_file and args.repo_2025 is None:
            ap.error("one of --inap, --inap-file or --repo-2025 is required")
        if args.inap or args.inap_file:
            m = compare(mena_dir, args.inap, args.inap_file, "iNAP")
            results.append(m)
        if args.repo_2025 is not None:
            root = Path(args.repo_2025) if args.repo_2025 != "auto" else Path.home() / "hermes_scratch" / "hydro-2025"
            adj = find_2025_repo(root) if root.is_dir() else (root if root.is_file() else None)
            if adj is None:
                print(f"error: no adjacency_matrix_ox.csv found under {root}", file=sys.stderr)
                return 2
            print(f"[2025 repo] using {adj}")
            m = compare(mena_dir, None, adj, "2025 repo")
            results.append(m)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
