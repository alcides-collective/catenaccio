#!/usr/bin/env python3
"""Monte Carlo title-defence engine v2 (Codex #1/#6/#7), corrected architecture.

Two fixes over the naive version:
  (1) Anchor each manager's round score to their REALIZED round history (not the
      projection sum, which undershoots the true round total by ~20-30 pts).
  (2) SHARED captain draws: my captain and every rival's captain are scored from
      the SAME Mbappe/Haaland random draws (coupled by the FRA-NOR head-to-head).
      This is what makes "match the field's captain" reduce relative variance — the
      whole point of Codex #7. Independent rival draws cannot represent it.

We isolate the captain lever: each manager's round = (their historical base, with
the implicit captain removed) + 2*(their actual captain's shared draw) + noise.
"""
import json, numpy as np
from pathlib import Path
rng = np.random.default_rng(7)
N = 300_000

ROOT = Path(__file__).resolve().parents[1]
calib = json.load(open(ROOT / "engine/calibration_data.json"))

# ---- realistic, skewed player-score sampler (re-centered residual bootstrap) ----
def bucket(proj): return min(int((proj or 0)//2), 4)
pools = {}
for r in calib:
    pools.setdefault((r["pos"], bucket(r["proj"])), []).append(r["real"])
glob = [r["real"] for r in calib]
def draw(proj, pos):
    pool = pools.get((pos, bucket(proj))) or glob
    s = rng.choice(pool, size=N).astype(float)
    return s - np.mean(pool) + proj            # keep shape, set mean=proj

# ---- shared premium captain draws, coupled by FRA vs NOR ----
mb = draw(5.4, "FWD"); ha = draw(5.6, "FWD")
cs = rng.random(N) < 0.32
fra_cs = cs & (rng.random(N) < 0.5); nor_cs = cs & ~fra_cs
mb = np.maximum(mb - 3.0*nor_cs + 1.0*fra_cs, -2)
ha = np.maximum(ha - 3.0*fra_cs + 1.0*nor_cs, -2)
other_cap = np.maximum(draw(6.0, "FWD"), -2)        # a generic non-FRA/NOR premium captain
CAP = {"Mbappe": mb, "Haaland": ha, "other": other_cap}
CAP_PROJ = {"Mbappe": 5.4, "Haaland": 5.6, "other": 6.0}

# ---- managers: anchor base to realized round history (RD=MD2, MD1=total-RD) ----
table = [("nowodworeksharks",94,185),("trzymurzyny",91,171),("AjWajYakasuka",85,167),
         ("HatchedEnd",83,149),("sadbartosz1906",69,147),("FranSport",91,145),
         ("Spermomix",77,138),("gorszczimane",62,123),("KWASZCZU",60,117),("inkonopka",50,103)]
allrounds = [v for nm,md2,tot in table for v in (tot-md2, md2)]
resid_sd = 7.0                                    # round-to-round noise of the non-captain XI

def manager_round(hist_mean, cap_key):
    # base = historical mean minus the implicit captain contribution it already contains
    base = hist_mean - 2*CAP_PROJ[cap_key]
    return base + 2*CAP[cap_key] + rng.normal(0, resid_sd, N)

# rivals captain by ownership-weighted priors (Mbappe 55% owned, Haaland 28%)
RIVAL_CAP_P = {"Mbappe": 0.55, "Haaland": 0.25, "other": 0.20}
rivals = []
for nm, md2, tot in table[1:]:
    hist = (tot-md2 + md2)/2.0
    rivals.append({"name": nm, "total": tot, "behind": 185-tot, "hist": hist})

# pre-assign each rival a captain mix: simulate as a probabilistic blend across N
def rival_round(r):
    pick = rng.choice(["Mbappe","Haaland","other"], size=N, p=list(RIVAL_CAP_P.values()))
    out = np.empty(N)
    for k in CAP:
        m = pick == k
        out[m] = (r["hist"] - 2*CAP_PROJ[k]) + 2*CAP[k][m] + rng.normal(0, resid_sd, m.sum())
    return out

me_hist = (91+94)/2.0
me_total = 185
rival_draws = {r["name"]: rival_round(r) for r in rivals}

print(f"N={N:,}  my round-history mean={me_hist:.0f}  lead over 2nd={185-table[1][2]} pts")
print(f"(rival captain priors: Mbappe {RIVAL_CAP_P['Mbappe']:.0%}, Haaland {RIVAL_CAP_P['Haaland']:.0%}, other {RIVAL_CAP_P['other']:.0%})\n")
print(f"{'my captain':12}{'E[my rd]':>9}{'SD':>7}{'P(2nd overtakes)':>18}{'P(drop from 1st)':>18}")
for cap in ["Haaland","Mbappe","Salah"]:
    ck = cap if cap in CAP else "other"          # Salah ~ a non-coupled 'other' premium
    if cap == "Salah":
        sal = np.maximum(draw(5.0,"MID"), -2)
        mine = (me_hist - 2*5.0) + 2*sal + rng.normal(0, resid_sd, N)
    else:
        mine = manager_round(me_hist, ck)
    my_after = me_total + mine
    drop = np.zeros(N, bool); over2 = (rivals[0]["total"]+rival_draws["trzymurzyny"]) > my_after
    for r in rivals:
        drop |= (r["total"] + rival_draws[r["name"]]) > my_after
    print(f"{cap:12}{mine.mean():>9.1f}{mine.std():>7.1f}{over2.mean()*100:>17.2f}%{drop.mean()*100:>17.2f}%")

print("\nLeader rule: pick the captain MINIMIZING P(drop from 1st). Matching the field's")
print("modal captain (Mbappe, ~55% owned) shares the draw and shrinks relative variance.")
