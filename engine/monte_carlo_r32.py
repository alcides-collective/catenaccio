#!/usr/bin/env python3
"""
Monte-Carlo R32 head-to-head: our squad vs the league leader (trzymurzyny).

Catenaccio INVERTED. Through the group stage we led and minimised variance vs the
field. After MD3 we are 2nd, -15 to trzymurzyny. A chaser wants variance *relative
to the specific leader*: captain someone they cannot match, and let non-shared
players create swing. This engine quantifies that, and (per audit round 3) does it
ROBUSTLY — it does not assume the leader's captain.

Model (audit R3 #4/#5: add correlation + dispersion, key by id-safe full name)
-----
realised(player) = r32_pts * team_shock[nation] * indiv_mult
  - team_shock[nation]: one unit-mean lognormal per nation, shared by all that
    nation's players in BOTH squads (a team has a good/bad day together; couples
    teammates and makes co-owned players cancel exactly).
  - indiv_mult: unit-mean lognormal, per-position sigma (attackers swing more).
The captain's realised points are doubled. Co-owned players cancel because they
draw the SAME team_shock and the SAME indiv_mult in both teams.

We are -15. Each candidate OUR-captain is scored over an ENSEMBLE of leader captain
choices (we do not know theirs) and under two leader-squad scenarios:
  A = leader re-optimises to $105m, XI ~ ours (captain is then the only lever);
  B = leader keeps the MD3 core (revealed preference).
Reported per our captain: expected P(beat) and P(erase -15) over the leader-captain
prior, plus the worst case across that prior (regret control).
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
N = 200_000
DEFICIT = 15.0

proj = {p["name"]: p for p in json.load(open(ROOT / "data/projections/projections_r32.json"))}
SIG = {"GK": 0.40, "DEF": 0.55, "MID": 0.75, "FWD": 0.80}
SIG_TEAM = 0.28  # correlated team-day shock

def lognorm(sigma, size):
    return rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=size)

# one shared team-day shock per nation (built lazily, reused across both squads)
_team_shock = {}
def team_shock(abbr):
    if abbr not in _team_shock:
        _team_shock[abbr] = lognorm(SIG_TEAM, N)
    return _team_shock[abbr]

# one shared individual multiplier per player (reused -> co-owned players cancel)
_indiv = {}
def indiv(name):
    if name not in _indiv:
        pos = proj[name]["pos"] if name in proj else "MID"
        _indiv[name] = lognorm(SIG[pos], N)
    return _indiv[name]

def realised(name):
    p = proj.get(name)
    base = p["r32_pts"] if p else 1.0
    ab = p["abbr"] if p else "ZZZ"
    return base * team_shock(ab) * indiv(name)

def team_round(xi, cap):
    tot = np.zeros(N)
    for nm in xi:
        r = realised(nm)
        tot += r * (2 if nm == cap else 1)
    return tot

# ---------------- squads ----------------
# Our R32 XI (optimizer/squad_build_r32.py, scouting bonus gated to real projections)
OUR_XI = ["Emiliano Martínez", "Lisandro Martínez", "Alexander Freeman", "Jesús Gallardo",
          "Vinícius Júnior", "Ousmane Dembélé", "Rubén Vargas", "Johan Manzambi",
          "Lionel Messi", "Kylian Mbappé", "Erling Haaland"]

# Leader trzymurzyny's revealed MD3 XI (scenario B core)
THEIR_MD3 = ["Yahia Fofana", "Nikola Katic", "Marc Cucurella", "Achraf Hakimi",
             "Bruno Fernandes", "Luis Díaz", "Florian Wirtz", "Vinícius Júnior",
             "Michael Olise", "Kylian Mbappé", "Yan Diomande"]

# Leader captain prior: they do NOT own Messi; safe premium Mbappe is modal.
LEADER_CAP_PRIOR = {"Kylian Mbappé": 0.60, "Erling Haaland": 0.18,
                    "Vinícius Júnior": 0.15, "Ousmane Dembélé": 0.07}

OUR_CANDS = ["Lionel Messi", "Kylian Mbappé", "Vinícius Júnior", "Ousmane Dembélé"]


def eval_scenario(their_xi, label):
    print(f"\n  === LEADER SCENARIO: {label} ===")
    sh = set(OUR_XI) & set(their_xi)
    print(f"  co-owned (cancel in H2H): {sorted(sh) or 'none'}")
    # leader round under each possible leader captain (must be in their XI)
    their_by_cap = {lc: team_round(their_xi, lc) for lc in LEADER_CAP_PRIOR if lc in their_xi}
    if not their_by_cap:
        their_by_cap = {"(none)": team_round(their_xi, None)}
    print(f"  {'OUR CAPTAIN':<17} {'E[swing]':>9} {'E P(beat)':>10} {'E P(>=+15)':>11} {'worst P(>=+15)':>15}")
    rows = []
    for oc in OUR_CANDS:
        if oc not in OUR_XI:
            continue
        our = team_round(OUR_XI, oc)
        es = ep = e15 = 0.0
        worst15 = 1.0
        for lc, their in their_by_cap.items():
            w = LEADER_CAP_PRIOR.get(lc, 1.0)
            s = our - their
            es += w * s.mean()
            ep += w * (s > 0).mean()
            p15 = (s >= DEFICIT).mean()
            e15 += w * p15
            worst15 = min(worst15, p15)
        rows.append((oc, es, ep, e15, worst15))
    rows.sort(key=lambda r: -r[3])
    for oc, es, ep, e15, w15 in rows:
        print(f"  {oc:<17} {es:>+9.1f} {ep:>9.1%} {e15:>10.1%} {w15:>14.1%}")
    return rows[0]


def main():
    miss = [n for n in set(OUR_XI) | set(THEIR_MD3) if n not in proj]
    if miss:
        print("  [note] no projection for:", ", ".join(miss))
    print("  We are 2nd, -15 to trzymurzyny. P(>=+15) = P(erase the deficit this round).")
    print("  Captain scored over a leader-captain PRIOR (we don't know theirs):")
    print("   ", LEADER_CAP_PRIOR)

    a = eval_scenario(OUR_XI, "leader re-optimises (XI ~ ours) — captain is the only lever")
    b = eval_scenario(THEIR_MD3, "leader keeps MD3 core")
    print(f"\n  => robust chaser captain: A -> {a[0]} (E P>=+15 {a[3]:.1%}, worst {a[4]:.1%});  "
          f"B -> {b[0]} (E P>=+15 {b[3]:.1%})")


if __name__ == "__main__":
    main()
