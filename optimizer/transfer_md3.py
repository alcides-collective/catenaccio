#!/usr/bin/env python3
"""
FIFA WC Fantasy 2026 - Matchday 3 (final group round) TRANSFER optimizer.
scipy.optimize.milp (HiGHS). No pulp.

The 15-man squad exists. We decide which current players to SELL and which pool
players to BUY (#buys == #sells, squad stays 15), then RE-PICK the optimal XI +
captain (free re-pick every round). Three scenarios:
  (a) 0 transfers          - re-pick XI/captain from current 15
  (b) best with <= 2 free  - no hit
  (c) best with one -3 hit  (up to 3 transfers) - only recommended if net > +3

Manager is 1st place (+14). Protecting a lead -> prefer fewest transfers / no
hit unless a move is clearly better.

Objective (maximize), recomputed AFTER transfers:
    sum over XI of md3_pts
  + scouting bonus for ALL sub-5% XI players:  2 * p4, p4 = clip((md3-1.5)/6,0,.85)
  + captain extra: + md3_pts[cap]          (BASE only; scouting bonus NOT doubled)
  + 0.15 * sum over BENCH of md3_pts
  - 3 * max(0, transfers - 2)
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]
PROJ_PATH = ROOT / "data/projections/projections_md3.json"

CURRENT_SQUAD = [
    {"id":477,"name":"Pickford","pos":"GK"},
    {"id":542,"name":"Kimmich","pos":"DEF"},
    {"id":400,"name":"Estupinan","pos":"DEF"},
    {"id":248,"name":"Munoz","pos":"DEF"},
    {"id":256,"name":"Luis Diaz","pos":"MID"},
    {"id":1597,"name":"Salah","pos":"MID"},
    {"id":1458,"name":"Kubo","pos":"MID"},
    {"id":543,"name":"Wirtz","pos":"MID"},
    {"id":1267,"name":"Balogun","pos":"FWD"},
    {"id":500,"name":"Mbappe","pos":"FWD"},
    {"id":855,"name":"Haaland","pos":"FWD"},
    {"id":1522,"name":"Courtois","pos":"GK"},
    {"id":1141,"name":"Akanji","pos":"DEF"},
    {"id":1963,"name":"Johnston","pos":"DEF"},
    {"id":107,"name":"Sabitzer","pos":"MID"},
]
CURRENT_IDS = set(p["id"] for p in CURRENT_SQUAD)

QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_WIN = {"GK": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
BUDGET = 100.0
COUNTRY_MAX = 3
BENCH_W = 0.15
FREE = 2
HIT = 3.0


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def scout_bonus(p):
    if p["owned_pct"] >= 5.0:
        return 0.0
    p4 = clip((p["md3_pts"] - 1.5) / 6.0, 0.0, 0.85)
    return 2.0 * p4


def load_players():
    with open(PROJ_PATH) as f:
        pool = json.load(f)
    # de-dup by id, keep first
    seen, players = set(), []
    for p in pool:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        players.append(p)
    for c in CURRENT_IDS:
        assert c in seen, f"current player {c} missing from pool"
    return players


def solve(players, max_transfers, pay_hit):
    """Solve one scenario. Returns dict with total (net), squad, XI, captain, vice."""
    n = len(players)
    pos = [p["pos"] for p in players]
    price = np.array([p["price"] for p in players])
    pts = np.array([p["md3_pts"] for p in players])
    sb = np.array([scout_bonus(p) for p in players])
    is_cur = np.array([1 if p["id"] in CURRENT_IDS else 0 for p in players])
    countries = sorted(set(p["abbr"] for p in players))

    # vars: s (squad, n), x (XI, n), c (captain, n) -> 3n
    S = slice(0, n)
    X = slice(n, 2 * n)
    C = slice(2 * n, 3 * n)
    N = 3 * n

    # objective (minimize -> negate):
    #   XI: x_i*(pts + scout)   captain: c_i*pts (base only)   bench: 0.15*(s-x)*pts
    obj = np.zeros(N)
    obj[X] = pts + sb              # XI gets base + scouting
    obj[C] = pts                   # captain doubles BASE only
    obj[S] += BENCH_W * pts        # all squad get bench weight
    obj[X] -= BENCH_W * pts        # XI players: remove bench weight (they get full)
    c = -obj  # minimize

    cons = []

    # squad size = 15
    a = np.zeros(N); a[S] = 1
    cons.append(LinearConstraint(a, 15, 15))

    # position quotas in squad
    for ps, q in QUOTAS.items():
        a = np.zeros(N)
        for i in range(n):
            if pos[i] == ps:
                a[S][i] = 1
        cons.append(LinearConstraint(a, q, q))

    # XI size = 11
    a = np.zeros(N); a[X] = 1
    cons.append(LinearConstraint(a, 11, 11))

    # XI position windows
    for ps, (lo, hi) in XI_WIN.items():
        a = np.zeros(N)
        for i in range(n):
            if pos[i] == ps:
                a[X][i] = 1
        cons.append(LinearConstraint(a, lo, hi))

    # x_i <= s_i ; c_i <= x_i
    for i in range(n):
        a = np.zeros(N); a[X][i] = 1; a[S][i] = -1
        cons.append(LinearConstraint(a, -np.inf, 0))
        a = np.zeros(N); a[C][i] = 1; a[X][i] = -1
        cons.append(LinearConstraint(a, -np.inf, 0))

    # exactly 1 captain
    a = np.zeros(N); a[C] = 1
    cons.append(LinearConstraint(a, 1, 1))

    # budget: sum price * s <= 100
    a = np.zeros(N); a[S] = price
    cons.append(LinearConstraint(a, -np.inf, BUDGET))

    # country limit (squad)
    for ctry in countries:
        a = np.zeros(N)
        for i in range(n):
            if players[i]["abbr"] == ctry:
                a[S][i] = 1
        cons.append(LinearConstraint(a, -np.inf, COUNTRY_MAX))

    # transfers = #current sold = sum(is_cur*(1-s)) ; cap <= max_transfers
    # sum is_cur*(1-s) <= max_transfers -> -sum(is_cur*s) <= max_transfers - 15
    a = np.zeros(N)
    a[S] = -is_cur
    cons.append(LinearConstraint(a, -np.inf, max_transfers - 15))

    integ = np.ones(N)
    bounds = Bounds(np.zeros(N), np.ones(N))
    res = milp(c=c, constraints=cons, integrality=integ, bounds=bounds)
    if not res.success:
        return None

    sol = np.round(res.x).astype(int)
    squad = [i for i in range(n) if sol[S][i]]
    xi = [i for i in range(n) if sol[X][i]]
    cap = [i for i in range(n) if sol[C][i]][0]
    transfers = int(sum(is_cur[i] == 1 and sol[S][i] == 0 for i in range(n)))

    gross = -res.fun
    net = gross - (HIT if (pay_hit and transfers > FREE) else 0.0)

    # pick vice: highest base-pts XI player that isn't captain
    xi_sorted = sorted(xi, key=lambda i: -pts[i])
    vice = next(i for i in xi_sorted if i != cap)

    return {
        "net": net, "gross": gross, "transfers": transfers,
        "squad": squad, "xi": xi, "cap": cap, "vice": vice,
        "players": players,
    }


def fmt_player(players, i, pts):
    p = players[i]
    return f"{p['name']} ({p['abbr']} {p['pos']}) {p['md3_pts']}"


def main():
    players = load_players()
    pts = np.array([p["md3_pts"] for p in players])

    # Scenario (a): 0 transfers
    a = solve(players, 0, pay_hit=False)
    # Scenario (b): best <= 2 free, no hit
    b = solve(players, 2, pay_hit=False)
    # Scenario (c): best with up to 3 transfers, pay -3 if >2 used
    c = solve(players, 3, pay_hit=True)

    scen = {"a_0transfer": round(a["net"], 2),
            "b_2free": round(b["net"], 2),
            "c_hit": round(c["net"], 2)}
    print("SCENARIO NET TOTALS:", scen)
    print(f"  (a) 0T: gross={a['gross']:.2f} transfers={a['transfers']}")
    print(f"  (b) <=2 free: gross={b['gross']:.2f} transfers={b['transfers']}")
    print(f"  (c) hit: gross={c['gross']:.2f} transfers={c['transfers']} net={c['net']:.2f}")

    # Recommend: max net; default to fewest transfers / no hit unless clearly better.
    # Codex audit fix: a +epsilon net gain is NOT enough to justify a -3 hit — projection
    # noise (RMSE ~3.7/player) swamps it, and an extra hit adds variance, which hurts a
    # leader (see mc_engine). Require the hit to clear a real margin over the best free plan.
    EPS = 1e-6
    HIT_MARGIN = 3.0  # net points the hit must beat the best free option by, to be worth it
    best = b
    label = "b_2free"
    if a["net"] >= b["net"] - EPS:
        best, label = a, "a_0transfer"
    if c["net"] > best["net"] + HIT_MARGIN and c["transfers"] > FREE:
        best, label = c, "c_hit"

    print("\nRECOMMENDED:", label)
    return players, pts, a, b, c, best, label, scen


if __name__ == "__main__":
    players, pts, a, b, c, best, label, scen = main()

    def dump(res, name):
        pl = res["players"]
        print(f"\n=== {name} (transfers={res['transfers']}, net={res['net']:.2f}) ===")
        sold = [players[i] for i in range(len(players))
                if players[i]["id"] in CURRENT_IDS and i not in res["squad"]]
        bought = [pl[i] for i in res["squad"] if pl[i]["id"] not in CURRENT_IDS]
        print(" OUT:", [f"{p['name']}({p['md3_pts']})" for p in sold])
        print(" IN :", [f"{p['name']}({p['md3_pts']})" for p in bought])
        print(" CAP:", pl[res["cap"]]["name"], "VICE:", pl[res["vice"]]["name"])
        xi = sorted(res["xi"], key=lambda i: ({"GK":0,"DEF":1,"MID":2,"FWD":3}[pl[i]["pos"]], -pl[i]["md3_pts"]))
        for i in xi:
            tag = " (C)" if i == res["cap"] else (" (V)" if i == res["vice"] else "")
            print(f"   XI {pl[i]['pos']:3} {pl[i]['name']:20} {pl[i]['md3_pts']}{tag}")
        bench = [i for i in res["squad"] if i not in res["xi"]]
        for i in bench:
            print(f"   BN {pl[i]['pos']:3} {pl[i]['name']:20} {pl[i]['md3_pts']}")
        cost = sum(pl[i]["price"] for i in res["squad"])
        print(f"   squad cost={cost:.1f} bank={100-cost:.1f}")

    dump(a, "A 0-transfer")
    dump(b, "B <=2 free")
    dump(c, "C hit")
