"""
MultiSinkValueGraph visualizer.

Reads a graph JSON saved by value_graph.save_graph_data() and renders a PNG.

Usage:
    python -m recipe.graft.visualize_task <step_N.json> [uid_or_index]
    python -m recipe.graft.visualize_task   # interactive prompt
"""

import sys
import json
import re
import textwrap
import numpy as np
import matplotlib
import matplotlib.patheffects
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# ── state text extraction ─────────────────────────────────────────────────────

def extract_state_label(state_str, max_chars=80):
    """
    Extract a short display label from a raw obs string.

    - Welcome/first node: extract the task description after "Your task is to:"
    - Other nodes: take text up to (but not including) "Items in hand"
    """
    s = str(state_str).strip()

    # First node: contains "Your task is to:"
    m = re.search(r'[Yy]our task is to[:\s]+(.+?)(?:\.|$)', s)
    if m:
        task = m.group(1).strip().rstrip('.')
        return _wrap(task, width=26, max_lines=3)

    # Other nodes: cut at "Items in hand"
    cut = re.search(r'\bItems in hand\b', s, re.IGNORECASE)
    if cut:
        s = s[:cut.start()].strip().rstrip('.')

    # Remove leading "You arrive at ..." prefix for brevity, keep the rest
    s = s.replace('\n', ' ').strip()
    if len(s) > max_chars:
        s = s[:max_chars - 1] + '…'
    return _wrap(s, width=26, max_lines=3)


def _wrap(s, width=26, max_lines=3):
    lines = textwrap.wrap(s.replace('\n', ' '), width)
    if not lines:
        return s[:width]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1] + '…'
    return '\n'.join(lines)


# ── BFS layout ────────────────────────────────────────────────────────────────

def bfs_layout(node_ids, edge_list):
    """
    Integer grid layout: col = BFS depth, row = index within layer.
    Returns pos {node_id: (col, row)}.
    """
    adj = defaultdict(set)
    in_deg = defaultdict(int)
    for e in edge_list:
        adj[e["src"]].add(e["dst"])
        in_deg[e["dst"]] += 1

    all_ids = set(node_ids)
    roots = [n for n in all_ids if in_deg[n] == 0] or [next(iter(all_ids))]

    visited = {r: 0 for r in roots}
    queue = list(roots)
    layers = defaultdict(list)
    for r in roots:
        layers[0].append(r)

    while queue:
        nxt = []
        for u in queue:
            for v in adj.get(u, set()):
                if v not in visited:
                    visited[v] = visited[u] + 1
                    layers[visited[v]].append(v)
                    nxt.append(v)
        queue = nxt

    for n in all_ids:
        if n not in visited:
            d = max(visited.values(), default=0) + 1
            visited[n] = d
            layers[d].append(n)

    pos = {}
    for col, nds in layers.items():
        for row, nid in enumerate(nds):
            pos[nid] = (col, row)
    return pos


# ── arc midpoint helper ───────────────────────────────────────────────────────

def arc_midpoint(x0, y0, x1, y1, rad):
    """
    Approximate midpoint of a quadratic Bezier arc used by matplotlib arc3.
    The control point is perpendicular to the chord midpoint by rad * chord_len.
    Returns (mx, my) — the point at t=0.5 on the Bezier.
    """
    # chord midpoint
    mx0, my0 = (x0 + x1) / 2, (y0 + y1) / 2
    # perpendicular direction (rotate chord 90°)
    dx, dy = x1 - x0, y1 - y0
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    px, py = -dy / length, dx / length
    # control point
    cx, cy = mx0 + rad * length * px, my0 + rad * length * py
    # Bezier at t=0.5: 0.25*P0 + 0.5*Ctrl + 0.25*P1
    bx = 0.25 * x0 + 0.5 * cx + 0.25 * x1
    by = 0.25 * y0 + 0.5 * cy + 0.25 * y1
    return bx, by


# ── core draw ─────────────────────────────────────────────────────────────────

def draw_graph(json_data, V, uid, output_path, dpi=120):
    nodes_info = {n["id"]: n for n in json_data["nodes"]}
    sinks      = json_data["sinks"]   # {id: R}

    # ── deduplicate edges: keep unique (src, dst, action) ──
    seen_edges = {}
    for e in json_data["edges"]:
        key = (e["src"], e["dst"], e["action"])
        if key not in seen_edges:
            seen_edges[key] = e
    edge_list = list(seen_edges.values())

    node_ids = list(nodes_info.keys())
    pos = bfs_layout(node_ids, edge_list)

    n_cols = max(c for c, r in pos.values()) + 1
    n_rows = max(r for c, r in pos.values()) + 1

    # fixed cell size
    CELL_W, CELL_H = 3.0, 2.2
    fig_w = max(8,  n_cols * CELL_W)
    fig_h = max(5,  n_rows * CELL_H)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title("", pad=0)

    R = 0.30          # node radius in data units
    cmap = plt.cm.RdYlGn
    FONT_STATE  = 8.0   # state text inside node
    FONT_V      = 7.0   # V value below node
    FONT_ACTION = 6.0   # edge action label
    FONT_STATS  = 8.0   # bottom stats line
    # vertical offset (data units) added to label when edge is nearly horizontal
    HORIZ_LABEL_OFFSET = 0.06

    # ── draw edges ──
    # group by (src, dst) to assign arc offsets
    pair_edges = defaultdict(list)
    for e in edge_list:
        pair_edges[(e["src"], e["dst"])].append(e)

    for (src, dst), edges in pair_edges.items():
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        n = len(edges)

        for i, e in enumerate(edges):
            # fan out arcs: 0→0, 1→+0.2, 2→-0.2, 3→+0.4, ...
            if n == 1:
                rad = 0.0
            else:
                sign = 1 if i % 2 == 0 else -1
                rad = sign * (0.2 + (i // 2) * 0.2)

            color = '#d62728' if dst in sinks else '#1f77b4'
            lw = 1.2

            # Offset start/end points to node edge in data coordinates
            dx, dy = x1 - x0, y1 - y0
            dist = max((dx**2 + dy**2) ** 0.5, 1e-6)
            ux, uy = dx / dist, dy / dist
            sx, sy = x0 + ux * R, y0 + uy * R   # start at src edge
            ex, ey = x1 - ux * R, y1 - uy * R   # end at dst edge

            ax.annotate(
                '', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle='-|>', color=color, lw=lw,
                    mutation_scale=14,
                    connectionstyle=f'arc3,rad={rad}',
                ),
                zorder=2,
            )

            # label at arc midpoint
            mx, my = arc_midpoint(x0, y0, x1, y1, rad)
            # if edge is nearly horizontal, shift label upward to avoid overlap
            dy_abs = abs(y1 - y0)
            if dy_abs < 0.3:   # nearly horizontal
                my -= HORIZ_LABEL_OFFSET   # invert_yaxis: subtract = move up visually
            action_str = str(e["action"])[:40] if e["action"] else '—'
            ax.text(mx, my, action_str,
                    fontsize=FONT_ACTION, ha='center', va='center', zorder=4,
                    color='#111111',
                    bbox=dict(boxstyle='round,pad=0.10', fc='#fffde7',
                              ec='#cccccc', lw=0.4, alpha=0.92))

    # ── draw nodes ──
    for nid, ninfo in nodes_info.items():
        if nid not in pos:
            continue
        cx, cy = pos[nid]
        v_val   = float(V.get(nid, 0.0))
        color   = cmap(v_val)
        is_sink = ninfo["is_sink"]

        if is_sink:
            ax.add_patch(plt.Polygon(
                [(cx,     cy - R * 1.5),
                 (cx + R, cy),
                 (cx,     cy + R * 1.5),
                 (cx - R, cy)],
                closed=True, fc=color, ec='#222222', lw=1.5, zorder=5,
            ))
        else:
            ax.add_patch(plt.Circle(
                (cx, cy), R, fc=color, ec='#222222', lw=1.0, zorder=5))

        # state text INSIDE node
        state_label = extract_state_label(ninfo["state"])
        ax.text(cx, cy, state_label,
                fontsize=FONT_STATE, ha='center', va='center',
                color='black', zorder=6,
                multialignment='center')

        # V value BELOW node
        v_str = f"V={v_val:.3f}"
        if is_sink and ninfo.get("sink_R") is not None:
            v_str += f"  R={ninfo['sink_R']}"
        ax.text(cx, cy + R + 0.06, v_str,
                fontsize=FONT_V, ha='center', va='top',
                fontweight='bold', color='#222222', zorder=6)



    plt.tight_layout(pad=0.1)
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── load JSON ──────────────────────────────────────────────────────────────────

def load_graph_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    Gs_simple = {}
    values_per_uid = {}
    for uid, gdata in data.items():
        sinks = {n["id"]: n["sink_R"] for n in gdata["nodes"] if n["is_sink"]}
        Gs_simple[uid] = {
            "nodes": gdata["nodes"],
            "edges": gdata["edges"],
            "sinks": sinks,
        }
        values_per_uid[uid] = {n["id"]: n["V"] for n in gdata["nodes"]}
    return Gs_simple, values_per_uid


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else input("Path to .json file: ").strip()
    uid_sel   = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading {data_path} ...")
    Gs, Vs = load_graph_json(data_path)
    uids = sorted(Gs.keys())

    print(f"\n{'idx':>4}  {'uid':36}  nodes  sinks")
    print("-" * 65)
    for i, uid in enumerate(uids):
        g = Gs[uid]
        print(f"{i:>4}  {uid}  {len(g['nodes']):>5}  {len(g['sinks']):>5}")

    if uid_sel is None:
        print()
        uid_sel = input("Enter index or UID (q to quit): ").strip()
    if uid_sel.lower() == 'q':
        return

    if uid_sel.isdigit():
        uid = uids[int(uid_sel)]
    elif uid_sel in Gs:
        uid = uid_sel
    else:
        print(f"UID '{uid_sel}' not found.")
        return

    V = Vs[uid]
    print(f"\nVisualizing uid={uid}")
    print(f"  nodes={len(Gs[uid]['nodes'])}  edges={len(Gs[uid]['edges'])}")
    print(f"  V range=[{min(V.values()):.4f}, {max(V.values()):.4f}]")

    out = f"value_graph_{uid[:12]}.png"
    draw_graph(Gs[uid], V, uid=uid, output_path=out)


if __name__ == "__main__":
    main()
