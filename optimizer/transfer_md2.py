#!/usr/bin/env python3
"""
FIFA WC Fantasy 2026 - Matchday 2 TRANSFER optimizer (scipy.optimize.milp).

The 15-man squad already exists. We decide which current players to SELL and
which pool players to BUY (#buys == #sells, squad stays 15), then RE-PICK the
optimal XI + captain (free every round). Three scenarios are solved:
  (a) 0 transfers  - just re-pick XI/captain from the current 15
  (b) <= 2 free transfers
  (c) allow exactly 1 hit (up to 3 transfers, -3 pts)

Decision variables (binary), over the COMBINED universe = current squad U pool:
    s_i  - player i is in the post-transfer 15-man squad
    x_i  - player i is in the starting XI            (x_i <= s_i)
    c_i  - player i is captain                       (c_i <= x_i)
    d_i  - player i is a "differential captain"-eligible scout pick that we
           credit with the differential scouting bonus when captained
           (handled in objective via c_i and a per-player constant; see below)

Transfer accounting:
    For current players, "sell" = (1 - s_i).  For pool-only players, "buy" = s_i.
    #sells == #buys is enforced automatically by fixing squad size = 15 and
    composition, BUT we still need the transfer COUNT for the hit penalty:
        transfers = sum over current players of (1 - s_i)   [= #sold = #bought]
    We cap transfers <= T (scenario param) and the objective already has no
    explicit penalty for <=2; for the hit scenario the -3 is added as a constant
    when T==3 (only pay if 3 transfers actually used -> handled by comparing the
    best 2-transfer vs best 3-transfer solution, see main()).

Objective (maximize), recomputed AFTER transfers:
    sum over XI of md2_pts
  + captain extra: + md2_pts[cap]            (captain doubles)
  + captain differential scouting bonus: for XI players with owned_pct < 5%,
        p4 = clip((md2_pts - 1.5)/6, 0, 0.85)
        bonus = 2 * p4
    credited only to the captain (the scouting/explicit-captain pick). We model
    this by giving each captain candidate an extra term 2*p4_i on c_i.
  + 0.15 * sum over BENCH of md2_pts
  - 3 * max(0, transfers - 2)   [applied via scenario, see main()]

The captain choice therefore maximizes (md2_pts_i + diff_bonus_i) over the XI:
this is exactly the "Haaland vs Mbappe vs differential" decision.
"""

import json
import sys
from collections import Counter

import numpy as np
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]
PROJ_PATH = ROOT / "data/projections/projections_md2.json"

CURRENT_SQUAD = [
    {"id": 477, "name": "Pickford", "abbr": "ENG", "pos": "GK"},
    {"id": 542, "name": "Kimmich", "abbr": "GER", "pos": "DEF"},
    {"id": 400, "name": "Estupinan", "abbr": "ECU", "pos": "DEF"},
    {"id": 1141, "name": "Akanji", "abbr": "SUI", "pos": "DEF"},
    {"id": 1274, "name": "Pulisic", "abbr": "USA", "pos": "MID"},
    {"id": 1597, "name": "Salah", "abbr": "EGY", "pos": "MID"},
    {"id": 1458, "name": "Kubo", "abbr": "JPN", "pos": "MID"},
    {"id": 107, "name": "Sabitzer", "abbr": "AUT", "pos": "MID"},
    {"id": 500, "name": "Mbappe", "abbr": "FRA", "pos": "FWD"},
    {"id": 855, "name": "Haaland", "abbr": "NOR", "pos": "FWD"},
    {"id": 226, "name": "J.David", "abbr": "CAN", "pos": "FWD"},
    {"id": 1522, "name": "Courtois", "abbr": "BEL", "pos": "GK"},
    {"id": 248, "name": "Munoz", "abbr": "COL", "pos": "DEF"},
    {"id": 1963, "name": "Johnston", "abbr": "CAN", "pos": "DEF"},
    {"id": 543, "name": "Wirtz", "abbr": "GER", "pos": "MID"},
]
CURRENT_IDS = [p["id"] for p in CURRENT_SQUAD]

QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_WINDOWS = {"GK": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
BUDGET = 100.0
COUNTRY_MAX = 3
BENCH_W = 0.15


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def load_universe():
    proj = {p["id"]: p for p in json.load(open(PROJ_PATH))}
    fallback = []
    universe = []
    cur_set = set(CURRENT_IDS)
    # current squad first (so they keep pool projection if present)
    for cp in CURRENT_SQUAD:
        if cp["id"] in proj:
            p = dict(proj[cp["id"]])
        else:
            nailed = True  # all current players are nailed XI/bench starters
            p = {"id": cp["id"], "name": cp["name"], "abbr": cp["abbr"],
                 "pos": cp["pos"], "price": None, "owned_pct": 0.0,
                 "md2_pts": 3.0 if nailed else 2.0}
            fallback.append(cp["name"])
        p["is_current"] = True
        universe.append(p)
    # pool players not in current squad
    for pid, p in proj.items():
        if pid in cur_set:
            continue
        q = dict(p)
        q["is_current"] = False
        universe.append(q)
    return universe, fallback


def diff_bonus(p):
    """Captain differential scouting bonus, only for sub-5% owned XI players."""
    if p["owned_pct"] < 5.0:
        p4 = clip((p["md2_pts"] - 1.5) / 6.0, 0.0, 0.85)
        return 2.0 * p4
    return 0.0


def solve(universe, max_transfers):
    """Solve the transfer MILP for a transfer cap. Returns dict with solution.
    The -3 hit penalty is NOT baked in here; callers add it per scenario."""
    n = len(universe)
    S, X, C = 0, n, 2 * n
    nv = 3 * n

    md2 = np.array([p["md2_pts"] for p in universe])
    price = np.array([p["price"] for p in universe])
    is_cur = np.array([1 if p["is_current"] else 0 for p in universe])
    dbonus = np.array([diff_bonus(p) for p in universe])

    # objective (maximize -> negate for milp minimize)
    cobj = np.zeros(nv)
    cobj[S:S + n] = -(BENCH_W * md2)                 # bench credit for squad
    cobj[X:X + n] = -(md2 - BENCH_W * md2)           # starter increment over bench
    cobj[C:C + n] = -(md2 + dbonus)                  # captain doubling + diff scout bonus

    A, lb, ub = [], [], []

    def add(row, lo, hi):
        A.append(row); lb.append(lo); ub.append(hi)

    # budget
    row = np.zeros(nv); row[S:S + n] = price
    add(row, -np.inf, BUDGET)

    # squad composition
    for pos, q in QUOTAS.items():
        row = np.zeros(nv)
        for i, p in enumerate(universe):
            if p["pos"] == pos:
                row[S + i] = 1
        add(row, q, q)

    # country limit
    for abbr in sorted({p["abbr"] for p in universe}):
        row = np.zeros(nv)
        for i, p in enumerate(universe):
            if p["abbr"] == abbr:
                row[S + i] = 1
        add(row, 0, COUNTRY_MAX)

    # XI size
    row = np.zeros(nv); row[X:X + n] = 1
    add(row, 11, 11)

    # XI position windows
    for pos, (lo, hi) in XI_WINDOWS.items():
        row = np.zeros(nv)
        for i, p in enumerate(universe):
            if p["pos"] == pos:
                row[X + i] = 1
        add(row, lo, hi)

    # one captain
    row = np.zeros(nv); row[C:C + n] = 1
    add(row, 1, 1)

    # linking
    for i in range(n):
        # XI must be in squad:  x_i <= s_i
        row = np.zeros(nv); row[X + i] = 1; row[S + i] = -1
        add(row, -np.inf, 0)
        # captain must be in the XI:  c_i <= x_i   (guarantees captain in XI)
        row = np.zeros(nv); row[C + i] = 1; row[X + i] = -1
        add(row, -np.inf, 0)
        # captain must also be in squad (redundant w/ above but explicit): c_i <= s_i
        row = np.zeros(nv); row[C + i] = 1; row[S + i] = -1
        add(row, -np.inf, 0)

    # transfer cap: transfers = #current players sold = sum_cur (1 - s_i) <= T
    # => sum_cur s_i >= (#current) - T
    row = np.zeros(nv); row[S:S + n] = is_cur
    add(row, len(CURRENT_IDS) - max_transfers, np.inf)

    res = milp(
        c=cobj,
        constraints=LinearConstraint(np.array(A), np.array(lb), np.array(ub)),
        integrality=np.ones(nv),
        bounds=Bounds(np.zeros(nv), np.ones(nv)),
    )
    if not res.success:
        raise RuntimeError(f"MILP failed (T={max_transfers}): {res.message}")

    z = np.round(res.x).astype(int)
    squad = [i for i in range(n) if z[S + i] == 1]
    xi = [i for i in range(n) if z[X + i] == 1]
    cap = next(i for i in range(n) if z[C + i] == 1)
    raw_obj = -res.fun

    sold = [i for i in range(n) if universe[i]["is_current"] and z[S + i] == 0]
    bought = [i for i in range(n) if not universe[i]["is_current"] and z[S + i] == 1]
    transfers = len(sold)
    return {
        "squad": squad, "xi": xi, "cap": cap, "raw_obj": raw_obj,
        "sold": sold, "bought": bought, "transfers": transfers,
    }


def gross_xi_points(universe, sol):
    """Expected MD2 points actually scored by the XI + captain doubling +
    differential scouting bonus on captain (no bench, no hit)."""
    md2 = [universe[i]["md2_pts"] for i in sol["xi"]]
    base = sum(md2)
    cap = sol["cap"]
    cap_extra = universe[cap]["md2_pts"] + diff_bonus(universe[cap])
    return base + cap_extra


def verify(universe, sol, max_transfers):
    errs = []
    sq = [universe[i] for i in sol["squad"]]
    st = [universe[i] for i in sol["xi"]]
    if len(sq) != 15:
        errs.append(f"squad size {len(sq)} != 15")
    if len(st) != 11:
        errs.append(f"XI size {len(st)} != 11")
    cost = sum(p["price"] for p in sq)
    if cost > BUDGET + 1e-9:
        errs.append(f"budget {cost:.1f} > {BUDGET}")
    cnt = {pos: sum(1 for p in sq if p["pos"] == pos) for pos in QUOTAS}
    if cnt != QUOTAS:
        errs.append(f"squad composition {cnt} != {QUOTAS}")
    cc = Counter(p["abbr"] for p in sq)
    bad = {k: v for k, v in cc.items() if v > COUNTRY_MAX}
    if bad:
        errs.append(f"country limit violated {bad}")
    xc = {pos: sum(1 for p in st if p["pos"] == pos) for pos in QUOTAS}
    lo_hi = XI_WINDOWS
    for pos in QUOTAS:
        lo, hi = lo_hi[pos]
        if not (lo <= xc[pos] <= hi):
            errs.append(f"XI {pos} count {xc[pos]} outside [{lo},{hi}]")
    if not set(sol["xi"]).issubset(set(sol["squad"])):
        errs.append("XI not subset of squad")
    if sol["cap"] not in sol["xi"]:
        errs.append("captain not in XI")
    if sol["transfers"] > max_transfers:
        errs.append(f"transfers {sol['transfers']} > cap {max_transfers}")
    if len(sol["sold"]) != len(sol["bought"]):
        errs.append(f"sells {len(sol['sold'])} != buys {len(sol['bought'])}")
    return errs, cost, cnt, xc


def describe(universe, sol):
    xi = sol["xi"]
    xc = {pos: sum(1 for i in xi if universe[i]["pos"] == pos) for pos in QUOTAS}
    formation = f"{xc['DEF']}-{xc['MID']}-{xc['FWD']}"
    # vice = XI player with 2nd best (md2 + diff bonus), excluding captain
    rank = sorted(xi, key=lambda i: -(universe[i]["md2_pts"] + diff_bonus(universe[i])))
    cap = sol["cap"]
    vice = next(i for i in rank if i != cap)
    return formation, vice


def main():
    universe, fallback = load_universe()

    print("=" * 78)
    print("FIFA WC Fantasy 2026 - MATCHDAY 2 TRANSFER OPTIMIZER")
    print("=" * 78)
    bank = round(BUDGET - sum(universe[i]["price"] for i in range(len(universe))
                              if universe[i]["is_current"]), 2)
    print(f"Current squad value: {round(BUDGET - bank,2)}   Bank: {bank}")
    if fallback:
        print("Fallback md2_pts assigned to (missing from pool):", fallback)
    else:
        print("All current players found in pool (no fallback projections needed).")

    # Scenario A: 0 transfers
    solA = solve(universe, 0)
    # Scenario B: <=2 free transfers
    solB = solve(universe, 2)
    # Scenario C: up to 3 transfers (1 hit)
    solC = solve(universe, 3)

    grossA = gross_xi_points(universe, solA)
    grossB = gross_xi_points(universe, solB)
    grossC = gross_xi_points(universe, solC)

    netA = grossA - 3 * max(0, solA["transfers"] - 2)
    netB = grossB - 3 * max(0, solB["transfers"] - 2)
    netC = grossC - 3 * max(0, solC["transfers"] - 2)

    scenarios = [
        ("A: 0 transfers (re-pick XI/cap only)", solA, grossA, netA),
        ("B: <=2 free transfers", solB, grossB, netB),
        ("C: allow 1 hit (<=3 transfers, -3)", solC, grossC, netC),
    ]

    for label, sol, gross, net in scenarios:
        errs, cost, cnt, xc = verify(universe, sol, 99)
        formation, vice = describe(universe, sol)
        print("\n" + "-" * 78)
        print(f"[{label}]")
        print(f"  transfers={sol['transfers']}  gross XI EV={gross:.2f}  "
              f"hit={3*max(0,sol['transfers']-2)}  NET EV={net:.2f}  "
              f"cost={cost:.1f}  formation={formation}")
        if sol["transfers"]:
            for o, b in zip(sorted(sol["sold"]), sorted(sol["bought"])):
                # pair by best matching not strictly needed for reporting; list separately
                pass
            outs = [universe[i] for i in sol["sold"]]
            ins = [universe[i] for i in sol["bought"]]
            print("  OUT:", ", ".join(f"{p['name']}({p['pos']},{p['md2_pts']})" for p in outs))
            print("  IN :", ", ".join(f"{p['name']}({p['pos']},{p['md2_pts']})" for p in ins))
        cap = universe[sol["cap"]]
        vp = universe[vice]
        print(f"  Captain: {cap['name']} (md2={cap['md2_pts']}, diffbonus={diff_bonus(cap):.2f}, own={cap['owned_pct']}%)")
        print(f"  Vice:    {vp['name']} (md2={vp['md2_pts']})")
        if errs:
            print("  CONSTRAINT VIOLATIONS:", errs)
            sys.exit(1)

    # Recommendation: default to NOT taking a hit unless C clearly wins (>~3-4 net)
    best_free = max([(netA, "A", solA), (netB, "B", solB)], key=lambda t: t[0])
    hit_gain = netC - best_free[0]
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(f"  Net EV  A={netA:.2f}  B={netB:.2f}  C={netC:.2f}")
    print(f"  Best free-transfer scenario: {best_free[1]} (net {best_free[0]:.2f})")
    print(f"  Hit scenario C net gain over best free: {hit_gain:.2f}")
    if hit_gain > 3.0:
        rec = "C"
    else:
        rec = best_free[1]
    print(f"  => RECOMMEND scenario {rec} "
          f"(manager LEADS; play safe, avoid hit unless net gain > ~3-4).")

    # dump machine-readable result
    result = {
        "bank": bank,
        "scenarios": {
            "A": {"transfers": solA["transfers"], "gross": round(grossA, 2), "net": round(netA, 2)},
            "B": {"transfers": solB["transfers"], "gross": round(grossB, 2), "net": round(netB, 2)},
            "C": {"transfers": solC["transfers"], "gross": round(grossC, 2), "net": round(netC, 2)},
        },
        "recommend": rec,
    }
    for label, sol in [("A", solA), ("B", solB), ("C", solC)]:
        formation, vice = describe(universe, sol)
        result.setdefault("detail", {})[label] = {
            "squad_ids": [universe[i]["id"] for i in sol["squad"]],
            "xi_ids": [universe[i]["id"] for i in sol["xi"]],
            "cap_id": universe[sol["cap"]]["id"],
            "vice_id": universe[vice]["id"],
            "formation": formation,
            "out": [universe[i]["id"] for i in sol["sold"]],
            "in": [universe[i]["id"] for i in sol["bought"]],
        }
    with open(ROOT / "results/transfers_md2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote", ROOT / "results/transfers_md2.json")
    return universe, scenarios, rec


if __name__ == "__main__":
    main()
