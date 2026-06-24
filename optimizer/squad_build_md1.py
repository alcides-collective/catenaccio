#!/usr/bin/env python3
"""
FIFA World Cup Fantasy 2026 - exact MILP squad optimizer (scipy.optimize.milp).

Decision variables (binary, per player i):
    s_i  - in 15-man squad
    x_i  - in starting XI            (x_i <= s_i)
    c_i  - captain                   (c_i <= x_i)

Constraints:
    sum(price_i * s_i) <= 100.0
    squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD  (=> sum s_i = 15)
    max 3 players per country (abbr) in squad
    XI: sum x_i = 11; exactly 1 GK; 3-5 DEF; 3-5 MID; 1-3 FWD
    exactly 1 captain
    differential: >= 6 starters with owned_pct < 5.0   (unless --no-diff-constraint)

Objective (maximize):
    starters:  base_i + diff_ev_i        where base_i = md1 + 0.5 * md23
    captain:   + md1_i                   (doubles MD1 for the captain)
    bench:     0.15 * base_i             for squad players not in the XI

diff_ev mapping (documented):
    Only players with owned_pct < 5.0 earn the differential bonus.
    p4 = P(score > 4 pts in a match), approximated by a linear-in-EV ramp:
        p4_md1  = clip((md1_pts        - 1.5) / 6, 0, 0.85)
        p4_md23 = clip((md23_pts / 2.0 - 1.5) / 6, 0, 0.85)
    Rationale: a player projected at 1.5 EV (bare appearance points) has ~0
    chance of a 4+ haul; each extra point of EV adds ~16.7pp of 4+ probability
    (haul probability grows roughly linearly with EV in this range); capped at
    0.85 because even elite projections retain blank risk. md23_pts covers two
    matches, so it is halved to a per-match EV before mapping.
        diff_ev = 2 * p4_md1 + 0.5 * (2 * p4_md23)  = 2 * p4_md1 + p4_md23
    (the 0.5 mirrors the MD2/3 weighting of the base objective).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def load_players(path):
    with open(path) as f:
        return json.load(f)


def build_and_solve(players, use_diff_constraint=True, exclude=frozenset()):
    n = len(players)
    # variable layout: [s_0..s_{n-1}, x_0..x_{n-1}, c_0..c_{n-1}]
    S, X, C = 0, n, 2 * n
    nv = 3 * n

    base = np.array([p["md1_pts"] + 0.5 * p["md23_pts"] for p in players])
    md1 = np.array([p["md1_pts"] for p in players])
    price = np.array([p["price"] for p in players])

    diff_ev = np.zeros(n)
    for i, p in enumerate(players):
        if p["owned_pct"] < 5.0:
            p4_md1 = clip((p["md1_pts"] - 1.5) / 6.0, 0.0, 0.85)
            p4_md23 = clip((p["md23_pts"] / 2.0 - 1.5) / 6.0, 0.0, 0.85)
            diff_ev[i] = 2.0 * p4_md1 + 0.5 * 2.0 * p4_md23

    # maximize -> milp minimizes, so negate
    cobj = np.zeros(nv)
    cobj[S:S + n] = -(0.15 * base)                      # bench part of s_i
    cobj[X:X + n] = -(base + diff_ev - 0.15 * base)     # starter increment over bench
    cobj[C:C + n] = -md1                                # captain MD1 doubling

    A, lb, ub = [], [], []

    def add(row, lo, hi):
        A.append(row)
        lb.append(lo)
        ub.append(hi)

    # budget
    row = np.zeros(nv); row[S:S + n] = price
    add(row, -np.inf, 100.0)

    # squad composition
    quotas = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, q in quotas.items():
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["pos"] == pos:
                row[S + i] = 1
        add(row, q, q)

    # country limit (max 3 per abbr)
    for abbr in sorted({p["abbr"] for p in players}):
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["abbr"] == abbr:
                row[S + i] = 1
        add(row, 0, 3)

    # XI size
    row = np.zeros(nv); row[X:X + n] = 1
    add(row, 11, 11)

    # XI position windows
    windows = {"GK": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
    for pos, (lo, hi) in windows.items():
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["pos"] == pos:
                row[X + i] = 1
        add(row, lo, hi)

    # exactly one captain
    row = np.zeros(nv); row[C:C + n] = 1
    add(row, 1, 1)

    # linking: x_i - s_i <= 0 ; c_i - x_i <= 0
    for i in range(n):
        row = np.zeros(nv); row[X + i] = 1; row[S + i] = -1
        add(row, -np.inf, 0)
        row = np.zeros(nv); row[C + i] = 1; row[X + i] = -1
        add(row, -np.inf, 0)

    # differential constraint: >= 6 starters with owned_pct < 5.0
    if use_diff_constraint:
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["owned_pct"] < 5.0:
                row[X + i] = 1
        add(row, 6, np.inf)

    # exclusions
    bounds_ub = np.ones(nv)
    for i, p in enumerate(players):
        if p["id"] in exclude:
            bounds_ub[S + i] = 0
            bounds_ub[X + i] = 0
            bounds_ub[C + i] = 0

    res = milp(
        c=cobj,
        constraints=LinearConstraint(np.array(A), np.array(lb), np.array(ub)),
        integrality=np.ones(nv),
        bounds=Bounds(np.zeros(nv), bounds_ub),
    )
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")

    z = np.round(res.x).astype(int)
    squad = [i for i in range(n) if z[S + i] == 1]
    xi = [i for i in range(n) if z[X + i] == 1]
    cap = [i for i in range(n) if z[C + i] == 1]
    objective = -res.fun
    return squad, xi, cap[0], objective, diff_ev, base


def verify(players, squad, xi, cap, use_diff_constraint, exclude):
    errs = []
    sq = [players[i] for i in squad]
    st = [players[i] for i in xi]
    if len(sq) != 15:
        errs.append(f"squad size {len(sq)} != 15")
    if len(st) != 11:
        errs.append(f"XI size {len(st)} != 11")
    cost = sum(p["price"] for p in sq)
    if cost > 100.0 + 1e-9:
        errs.append(f"budget {cost} > 100")
    cnt = {pos: sum(1 for p in sq if p["pos"] == pos) for pos in ("GK", "DEF", "MID", "FWD")}
    if cnt != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}:
        errs.append(f"squad composition {cnt}")
    from collections import Counter
    cc = Counter(p["abbr"] for p in sq)
    bad = {k: v for k, v in cc.items() if v > 3}
    if bad:
        errs.append(f"country limit violated {bad}")
    xc = {pos: sum(1 for p in st if p["pos"] == pos) for pos in ("GK", "DEF", "MID", "FWD")}
    if not (xc["GK"] == 1 and 3 <= xc["DEF"] <= 5 and 3 <= xc["MID"] <= 5 and 1 <= xc["FWD"] <= 3):
        errs.append(f"XI composition {xc}")
    if not set(xi).issubset(set(squad)):
        errs.append("XI not subset of squad")
    if cap not in xi:
        errs.append("captain not in XI")
    ndiff = sum(1 for p in st if p["owned_pct"] < 5.0)
    if use_diff_constraint and ndiff < 6:
        errs.append(f"differential count {ndiff} < 6")
    if any(p["id"] in exclude for p in sq):
        errs.append("excluded player selected")
    return errs, cost, ndiff


def report(players, squad, xi, cap, objective, diff_ev, base, label, use_diff_constraint, exclude):
    errs, cost, ndiff = verify(players, squad, xi, cap, use_diff_constraint, exclude)
    xi_set = set(xi)
    st = sorted(xi, key=lambda i: ("GK DEF MID FWD".split().index(players[i]["pos"]), -base[i]))
    bench = sorted(set(squad) - xi_set,
                   key=lambda i: ("GK DEF MID FWD".split().index(players[i]["pos"]), -base[i]))
    xc = {pos: sum(1 for i in xi if players[i]["pos"] == pos) for pos in ("DEF", "MID", "FWD")}
    formation = f"{xc['DEF']}-{xc['MID']}-{xc['FWD']}"
    # vice = second-best captain option = XI player with 2nd-highest md1_pts
    by_md1 = sorted(xi, key=lambda i: -players[i]["md1_pts"])
    vice = next(i for i in by_md1 if i != cap)

    print(f"\n=== {label} ===")
    print(f"Objective (weighted EV): {objective:.3f}   Cost: {cost:.1f}/100.0   "
          f"Formation: {formation}   XI <5% owned: {ndiff}")
    print(f"Captain: {players[cap]['name']}   Vice: {players[vice]['name']}")
    print(f"{'':4}{'pos':4}{'name':26}{'team':5}{'price':>6}{'own%':>6}{'md1':>5}{'md23':>6}{'dEV':>6}")
    for i in st:
        p = players[i]
        tag = "C" if i == cap else ("V" if i == vice else "")
        print(f" XI {p['pos']:4}{p['name']:26}{p['abbr']:5}{p['price']:>6.1f}{p['owned_pct']:>6.1f}"
              f"{p['md1_pts']:>5.1f}{p['md23_pts']:>6.1f}{diff_ev[i]:>6.2f}  {tag}")
    for i in bench:
        p = players[i]
        print(f" BN {p['pos']:4}{p['name']:26}{p['abbr']:5}{p['price']:>6.1f}{p['owned_pct']:>6.1f}"
              f"{p['md1_pts']:>5.1f}{p['md23_pts']:>6.1f}{diff_ev[i]:>6.2f}")
    if errs:
        print("CONSTRAINT VIOLATIONS:", errs)
        sys.exit(1)
    else:
        print("All constraints verified OK.")
    return {
        "squad": squad, "xi": xi, "cap": cap, "vice": vice, "objective": objective,
        "cost": cost, "formation": formation, "ndiff": ndiff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projections", default=str(ROOT / "data/projections/projections_md1.json"))
    ap.add_argument("--exclude", default="", help="comma-separated player ids to force out")
    ap.add_argument("--no-diff-constraint", action="store_true",
                    help="drop the >=6 sub-5%%-owned starters constraint (pure EV)")
    args = ap.parse_args()

    exclude = frozenset(int(t) for t in args.exclude.split(",") if t.strip())
    players = load_players(args.projections)

    use_diff = not args.no_diff_constraint
    squad, xi, cap, obj, diff_ev, base = build_and_solve(players, use_diff, exclude)
    label = "PURE-EV (no differential constraint)" if args.no_diff_constraint \
        else "DIFFERENTIAL-CONSTRAINED (>=6 sub-5% owned starters)"
    report(players, squad, xi, cap, obj, diff_ev, base, label, use_diff, exclude)


if __name__ == "__main__":
    main()
