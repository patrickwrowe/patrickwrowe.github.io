"""
Generate a 2D amorphous-graphene network and emit it as hairline SVG data.

Method
------
1. Build a honeycomb (graphene) sheet with bond length r0.
2. Apply N random Stone-Wales bond rotations. Each rotation turns four hexagons
   into two pentagons and two heptagons while preserving three-fold coordination
   everywhere. This is the canonical topological defect in graphene and the one
   that dominates graphitising carbons.
3. Relax the geometry: harmonic bonds at r0, plus a harmonic 1-3 term at
   sqrt(3)*r0, which enforces 120 degree bond angles without needing explicit
   angle derivatives.
4. Crop to the display frame; classify rings; emit paths.

The output therefore has honest ring statistics: hexagon-dominated with 5- and
7-membered rings as paired defects.
"""

import json
import math
from collections import deque

import numpy as np

RNG = np.random.default_rng(20)

R0 = 46.0                  # bond length in SVG units
W, H = 1400.0, 470.0       # display frame
NX, NY = 26, 14            # lattice cells (generously larger than the frame)
N_SWITCHES = 26
RELAX_STEPS = 4000
K_BOND, K_13 = 1.0, 0.35
DT = 0.06


# --------------------------------------------------------------------------- #
# 1. honeycomb lattice
# --------------------------------------------------------------------------- #
def honeycomb():
    a1 = np.array([1.5 * R0, math.sqrt(3) / 2 * R0])
    a2 = np.array([0.0, math.sqrt(3) * R0])
    idx, pos = {}, []

    def add(key, p):
        idx[key] = len(pos)
        pos.append(p)

    for m in range(NX):
        for n in range(NY):
            base = m * a1 + n * a2
            add(("A", m, n), base)
            add(("B", m, n), base + np.array([R0, 0.0]))

    bonds = set()
    for m in range(NX):
        for n in range(NY):
            a = idx[("A", m, n)]
            for key in (("B", m, n), ("B", m - 1, n), ("B", m - 1, n + 1)):
                if key in idx:
                    bonds.add(tuple(sorted((a, idx[key]))))
    return np.array(pos), sorted(bonds)


# --------------------------------------------------------------------------- #
# 2. Stone-Wales bond rotations
# --------------------------------------------------------------------------- #
def stone_wales(pos, bonds, n_switches):
    adj = {i: set() for i in range(len(pos))}
    for i, j in bonds:
        adj[i].add(j)
        adj[j].add(i)

    bondset = set(bonds)

    def cross(d, q):
        return d[0] * q[1] - d[1] * q[0]

    # a bond is a valid rotation site only if it and all its neighbours are
    # fully three-coordinate, i.e. genuinely in the bulk of the sheet
    interior = [
        (u, v) for u, v in bonds
        if len(adj[u]) == 3 and len(adj[v]) == 3
        and all(len(adj[w]) == 3 for w in (adj[u] | adj[v]) - {u, v})
    ]
    RNG.shuffle(interior)

    touched, done = set(), 0
    for u, v in interior:
        if done >= n_switches:
            break
        if len(adj[u]) != 3 or len(adj[v]) != 3:
            continue
        nbrs = {u, v} | adj[u] | adj[v]
        if nbrs & touched:
            continue                       # keep defects from overlapping

        d = pos[v] - pos[u]
        a_opts = [(cross(d, pos[w] - pos[u]), w) for w in adj[u] - {v}]
        c_opts = [(cross(d, pos[w] - pos[v]), w) for w in adj[v] - {u}]
        if len(a_opts) != 2 or len(c_opts) != 2:
            continue
        # rotating the bond by 90 degrees swaps partners across the bond axis,
        # so a and c must lie on opposite sides of it
        a = max(a_opts)[1]
        c = min(c_opts)[1]
        if c in adj[u] or a in adj[v] or a == c:
            continue

        for pair in ((u, a), (v, c)):
            bondset.discard(tuple(sorted(pair)))
            adj[pair[0]].discard(pair[1])
            adj[pair[1]].discard(pair[0])
        for pair in ((u, c), (v, a)):
            bondset.add(tuple(sorted(pair)))
            adj[pair[0]].add(pair[1])
            adj[pair[1]].add(pair[0])

        touched |= nbrs
        done += 1

    return sorted(bondset), adj, done


# --------------------------------------------------------------------------- #
# 3. relaxation
# --------------------------------------------------------------------------- #
def relax(pos, bonds, adj):
    pos = pos.copy()
    b = np.array(bonds)
    pairs13 = set()
    for centre, nbrs in adj.items():
        nl = sorted(nbrs)
        for x in range(len(nl)):
            for y in range(x + 1, len(nl)):
                pairs13.add(tuple(sorted((nl[x], nl[y]))))
    p13 = np.array(sorted(pairs13))
    target13 = math.sqrt(3) * R0

    edge = np.zeros(len(pos), dtype=bool)
    edge |= pos[:, 0] < pos[:, 0].min() + R0
    edge |= pos[:, 0] > pos[:, 0].max() - R0
    edge |= pos[:, 1] < pos[:, 1].min() + R0
    edge |= pos[:, 1] > pos[:, 1].max() - R0

    for _ in range(RELAX_STEPS):
        f = np.zeros_like(pos)
        for arr, target, k in ((b, R0, K_BOND), (p13, target13, K_13)):
            d = pos[arr[:, 1]] - pos[arr[:, 0]]
            r = np.linalg.norm(d, axis=1, keepdims=True)
            r = np.maximum(r, 1e-9)
            fv = k * (r - target) * d / r
            np.add.at(f, arr[:, 0], fv)
            np.add.at(f, arr[:, 1], -fv)
        f[edge] = 0.0
        pos += DT * f
    return pos


# --------------------------------------------------------------------------- #
# 4. rings
# --------------------------------------------------------------------------- #
def smallest_ring(adj, u, v, limit=9):
    prev, q = {u: None}, deque([(u, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= limit:
            continue
        for nb in adj[node]:
            if {node, nb} == {u, v}:
                continue
            if nb in prev:
                continue
            prev[nb] = node
            if nb == v:
                path, cur = [], v
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path
            q.append((nb, depth + 1))
    return None


def find_rings(adj, bonds):
    seen, rings = set(), []
    for u, v in bonds:
        path = smallest_ring(adj, u, v)
        if path and 4 <= len(path) <= 8:
            key = tuple(sorted(path))
            if key not in seen:
                seen.add(key)
                rings.append(path)
    return rings


def order_ring(pos, ring):
    c = np.mean([pos[i] for i in ring], axis=0)
    return sorted(ring, key=lambda i: math.atan2(pos[i][1] - c[1], pos[i][0] - c[0]))


# --------------------------------------------------------------------------- #
def main():
    pos, bonds = honeycomb()
    bonds, adj, n_done = stone_wales(pos, bonds, N_SWITCHES)
    pos = relax(pos, bonds, adj)
    rings = find_rings(adj, bonds)

    # centre the most defect-rich region in the frame
    defect_centres = [np.mean([pos[i] for i in r], axis=0) for r in rings if len(r) != 6]
    focus = np.mean(defect_centres, axis=0) if defect_centres else pos.mean(axis=0)
    pos = pos - focus + np.array([W / 2, H / 2])

    def inside(p, m=25.0):
        return -m <= p[0] <= W + m and -m <= p[1] <= H + m

    bond_paths = []
    for i, j in bonds:
        if inside(pos[i], 60) or inside(pos[j], 60):
            bond_paths.append(
                f"M{pos[i][0]:.1f} {pos[i][1]:.1f}L{pos[j][0]:.1f} {pos[j][1]:.1f}"
            )

    defect_polys, counts = [], {}
    for r in rings:
        counts[len(r)] = counts.get(len(r), 0) + 1
        if len(r) == 6:
            continue
        ordered = order_ring(pos, r)
        if not all(inside(pos[i], -5) for i in ordered):
            continue
        pts = " ".join(f"{pos[i][0]:.1f},{pos[i][1]:.1f}" for i in ordered)
        defect_polys.append({"n": len(r), "points": pts})

    nodes = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pos if inside(p)]

    blen = [
        float(np.linalg.norm(pos[i] - pos[j]))
        for i, j in bonds
        if inside(pos[i]) and inside(pos[j])
    ]

    out = {
        "width": W,
        "height": H,
        "bond_paths": bond_paths,
        "defect_polys": defect_polys,
        "nodes": nodes,
        "stats": {
            "switches_applied": n_done,
            "atoms_in_frame": len(nodes),
            "bonds_drawn": len(bond_paths),
            "ring_sizes": dict(sorted(counts.items())),
            "defects_in_frame": len(defect_polys),
            "bond_len_mean": round(np.mean(blen), 2),
            "bond_len_sd_pct": round(100 * np.std(blen) / np.mean(blen), 2),
        },
    }
    with open("network.json", "w") as f:
        json.dump(out, f)
    print(json.dumps(out["stats"], indent=2))


if __name__ == "__main__":
    main()
