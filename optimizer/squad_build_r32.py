#!/usr/bin/env python3
"""
FIFA World Cup Fantasy 2026 - Round of 32 squad optimizer (exact MILP).

Knockout reset: budget rises to $105m, transfers are unlimited (full rebuild),
the per-country cap stays at 3 through R32. Objective maximises projected R32
points (single match), with optional captaincy doubling.

This build is RIVAL-RELATIVE. We are no longer the leader: catenaccio inverts.
A chaser wants variance *relative to the specific leader*, so the captain pick and
a few marginal selections are chosen to differ from the leader's squad. The MILP
produces the EV-optimal $105m squad; the Monte-Carlo step (engine/monte_carlo.py)
then selects the captain / evaluates differential variants by P(overtake leader).

Player schema (data/projections/projections_r32.json):
    {id, name, abbr, pos, price, owned_pct, r32_pts}

Flags:
    --projections PATH   alternate projections file
    --budget FLOAT       squad budget (default 105.0)
    --exclude id,id      force players out
    --rival-penalty F    subtract F * r32_pts for each leader-owned, non-shared
                         premium we also pick (discourages mirroring; default 0)
    --rival-owned ids    comma list of player ids the leader owns (for the penalty)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJ = ROOT / "data/projections/projections_r32.json"

QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_WINDOWS = {"GK": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
POS_ORDER = "GK DEF MID FWD".split()


def load_players(path):
    with open(path) as f:
        return json.load(f)


def build_and_solve(players, budget=105.0, exclude=frozenset(),
                    rival_penalty=0.0, rival_owned=frozenset()):
    n = len(players)
    S, X, C = 0, n, 2 * n
    nv = 3 * n

    pts = np.array([p["r32_pts"] for p in players], dtype=float)
    price = np.array([p["price"] for p in players], dtype=float)

    # scouting bonus (audit R3 #1): sub-5%-owned starters earn +2 when they haul
    # (score >4). r32_pts is a single-match EV, so map it to P(>4) and value the
    # +2. This is NOT doubled by the captain (a bonus, like the group-stage build).
    # Only genuinely-projected players (src=="proj") earn it — form-prior fallbacks
    # carry a capped guess, not a real haul projection, so stacking a scouting bonus
    # on them manufactures fake cheap enablers (audit R3 #1/#6).
    # 0.5 discount: the linear p4 ramp overestimates scouting EV for high-projection
    # sub-5% players (audit R2/R3 #6), so we halve it rather than trust it fully.
    SCOUT_DISCOUNT = 0.5
    diff_ev = np.zeros(n)
    for i, p in enumerate(players):
        if (p.get("owned_pct") or 0.0) < 5.0 and p.get("src") == "proj":
            p4 = max(0.0, min(0.85, (pts[i] - 1.5) / 6.0))
            diff_ev[i] = SCOUT_DISCOUNT * 2.0 * p4

    # rival-mirroring penalty: discourage also-picking a leader-owned premium
    pen = np.zeros(n)
    if rival_penalty > 0:
        for i, p in enumerate(players):
            if p["id"] in rival_owned:
                pen[i] = rival_penalty * pts[i]

    # maximize -> milp minimizes, so negate
    cobj = np.zeros(nv)
    cobj[S:S + n] = -(0.15 * pts)                       # bench contribution
    cobj[X:X + n] = -(pts + diff_ev - 0.15 * pts - pen)  # starter increment (+scout, less penalty)
    cobj[C:C + n] = -pts                                # captain doubles base r32_pts only

    A, lb, ub = [], [], []

    def add(row, lo, hi):
        A.append(row); lb.append(lo); ub.append(hi)

    row = np.zeros(nv); row[S:S + n] = price
    add(row, -np.inf, budget)

    for pos, q in QUOTAS.items():
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["pos"] == pos:
                row[S + i] = 1
        add(row, q, q)

    for abbr in sorted({p["abbr"] for p in players}):
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["abbr"] == abbr:
                row[S + i] = 1
        add(row, 0, 3)

    row = np.zeros(nv); row[X:X + n] = 1
    add(row, 11, 11)

    for pos, (lo, hi) in XI_WINDOWS.items():
        row = np.zeros(nv)
        for i, p in enumerate(players):
            if p["pos"] == pos:
                row[X + i] = 1
        add(row, lo, hi)

    row = np.zeros(nv); row[C:C + n] = 1
    add(row, 1, 1)

    for i in range(n):
        row = np.zeros(nv); row[X + i] = 1; row[S + i] = -1
        add(row, -np.inf, 0)
        row = np.zeros(nv); row[C + i] = 1; row[X + i] = -1
        add(row, -np.inf, 0)

    bounds_ub = np.ones(nv)
    for i, p in enumerate(players):
        if p["id"] in exclude:
            bounds_ub[S + i] = bounds_ub[X + i] = bounds_ub[C + i] = 0

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
    cap = [i for i in range(n) if z[C + i] == 1][0]
    return squad, xi, cap, -res.fun, pts


def verify(players, squad, xi, cap, budget):
    errs = []
    sq = [players[i] for i in squad]; st = [players[i] for i in xi]
    if len(sq) != 15: errs.append(f"squad size {len(sq)} != 15")
    if len(st) != 11: errs.append(f"XI size {len(st)} != 11")
    cost = sum(p["price"] for p in sq)
    if cost > budget + 1e-9: errs.append(f"budget {cost:.1f} > {budget}")
    cnt = {pos: sum(1 for p in sq if p["pos"] == pos) for pos in POS_ORDER}
    if cnt != QUOTAS: errs.append(f"squad composition {cnt}")
    from collections import Counter
    bad = {k: v for k, v in Counter(p["abbr"] for p in sq).items() if v > 3}
    if bad: errs.append(f"country limit violated {bad}")
    xc = {pos: sum(1 for p in st if p["pos"] == pos) for pos in POS_ORDER}
    if not (xc["GK"] == 1 and 3 <= xc["DEF"] <= 5 and 3 <= xc["MID"] <= 5 and 1 <= xc["FWD"] <= 3):
        errs.append(f"XI composition {xc}")
    if not set(xi).issubset(set(squad)): errs.append("XI not subset of squad")
    if cap not in xi: errs.append("captain not in XI")
    return errs, cost


def report(players, squad, xi, cap, objective, pts, budget, label):
    errs, cost = verify(players, squad, xi, cap, budget)
    xi_set = set(xi)
    st = sorted(xi, key=lambda i: (POS_ORDER.index(players[i]["pos"]), -pts[i]))
    bench = sorted(set(squad) - xi_set, key=lambda i: (POS_ORDER.index(players[i]["pos"]), -pts[i]))
    xc = {pos: sum(1 for i in xi if players[i]["pos"] == pos) for pos in ("DEF", "MID", "FWD")}
    formation = f"{xc['DEF']}-{xc['MID']}-{xc['FWD']}"
    by_pts = sorted(xi, key=lambda i: -pts[i])
    vice = next(i for i in by_pts if i != cap)

    print(f"\n{'='*64}\n{label}  ({formation})  cost ${cost:.1f}/{budget}  obj {objective:.2f}\n{'='*64}")
    if errs:
        print("  !! VERIFY ERRORS:", errs)
    print("  XI:")
    for i in st:
        p = players[i]
        mark = " (C)" if i == cap else " (VC)" if i == vice else ""
        print(f"    {p['pos']:<3} {p['name'][:20]:<20} {p['abbr']} ${p['price']:<4} "
              f"own{p.get('owned_pct',0):>4}%  r32 {pts[i]:>4.1f}{mark}")
    print("  Bench:")
    for i in bench:
        p = players[i]
        print(f"    {p['pos']:<3} {p['name'][:20]:<20} {p['abbr']} ${p['price']:<4} r32 {pts[i]:>4.1f}")
    ev_xi = sum(pts[i] for i in xi) + pts[cap]
    print(f"  XI EV (captain doubled): {ev_xi:.1f}")
    return {"squad": [players[i]["id"] for i in squad],
            "xi": [players[i]["id"] for i in xi],
            "captain": players[cap]["id"], "vice": players[vice]["id"],
            "formation": formation, "cost": round(cost, 1), "xi_ev": round(ev_xi, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projections", default=str(DEFAULT_PROJ))
    ap.add_argument("--budget", type=float, default=105.0)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--rival-penalty", type=float, default=0.0)
    ap.add_argument("--rival-owned", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    players = load_players(args.projections)
    exclude = frozenset(int(x) for x in args.exclude.split(",") if x.strip())
    rival_owned = frozenset(int(x) for x in args.rival_owned.split(",") if x.strip())

    squad, xi, cap, obj, pts = build_and_solve(
        players, budget=args.budget, exclude=exclude,
        rival_penalty=args.rival_penalty, rival_owned=rival_owned)
    label = "R32 squad (EV-optimal)" if args.rival_penalty == 0 else \
            f"R32 squad (rival-tilt {args.rival_penalty})"
    out = report(players, squad, xi, cap, obj, pts, args.budget, label)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
