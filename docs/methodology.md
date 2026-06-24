# Methodology

## The game (constraints)
- Budget $100m (→ $105m from the Round of 32). Squad = 2 GK / 5 DEF / 5 MID / 3 FWD.
- Starting XI of 11 (1 GK, 3–5 DEF, 3–5 MID, 1–3 FWD). Max 3 players per nation in the
  group stage (rising in the knockouts). **Prices are fixed all tournament.**
- Captain scores double **base** points. Transfers: unlimited pre-tournament and before
  R32; 2 free per group round, each extra = −3 pts.
- **Scouting bonus:** a player owned by <5% of managers gets +2 in any match he scores
  >4 points. This bonus is **not** doubled by the captaincy.

## 1. Data (`data/`)
Pulled from the official open `play.fifa.com` JSON endpoints: every priced player with
live ownership, all squads/groups, the full fixture list and realised per-round points.
Merged into a per-round player pool (`data/processed/`).

## 2. Projections (`data/projections/`)
For each plausible starter we estimate expected fantasy points for the round, blending:
bookmaker-implied win/clean-sheet/goal probabilities, expected line-ups and rotation/
qualification context, and the position-specific scoring table. The figure is expected
points already multiplied by start probability, and **excludes** captain doubling and the
scouting bonus (the optimiser adds those, to avoid double counting).

> These are model estimates anchored on odds, not a trained statistical model. There is no
> per-player variance attached at this stage. This is the main known weakness.

## 3. Optimisation (`optimizer/`)
Exact MILP via `scipy.optimize.milp` (HiGHS).
- **`squad_build_md1.py`** — binary vars per player: in-squad, in-XI (≤ squad), captain
  (≤ XI). Constraints: budget, composition, ≤3/nation, valid XI, one captain. Optional
  "≥6 sub-5%-owned starters" differential constraint, with a pure-EV comparison.
- **`transfer_md2.py` / `transfer_md3.py`** — the squad exists; decide sell/buy (#buys =
  #sells), re-pick XI + captain. Solve 0 / ≤2-free / one-−3-hit scenarios. A hit must clear
  a real margin over the best free plan (not a +ε), because projection noise (~3.7 RMSE/
  player) swamps small gains and an extra hit adds variance.

## 4. Title-defence engine (`engine/`)
The novel part. For a manager **leading** a mini-league, expected points is the wrong
objective — what matters is `P(stay 1st)`, i.e. variance *relative to the field*.
- **`monte_carlo.py`** simulates the round with: skewed player score distributions
  bootstrapped from realised data; **shared captain draws** (your captain and every rival's
  captain scored from the same coupled Mbappé/Haaland draws, so "matching the field's
  captain" correctly reduces relative variance); each manager's base anchored to their
  realised round history. Outputs `P(drop from 1st)` per decision.
- **`calibrate_p4.py`** fits a logistic `P(pts>4)` on realised MD1+MD2 data and reports
  honest prospective validation (GroupKFold by player + leave-one-round-out), not just
  random CV.

Headline finding: defending a +14 lead, captaining the field's modal premium (Mbappé,
~55% owned) gives `P(drop) ≈ 12%` vs `≈ 17%` for the higher-EV differential captain —
because it shares the draw with most rivals. The result is robust to simulation parameters
but **fragile to the nearest rival's captain choice**.

## Known limitations / roadmap
1. Projections are LLM/odds estimates without per-player uncertainty.
2. The `p4` logistic beats the old heuristic on log-loss but not on Brier under
   leave-one-round-out; high-projection sub-5% players' scouting EV is still overestimated.
3. The Monte-Carlo simulator covers **one round** at a time; rivals are approximated from
   ownership priors + history, not observed line-ups.
4. Bookmaker odds are not yet de-vigged before conversion to probabilities.
5. **Next:** a full multi-round knockout `P(win the league)` simulator for the R32 reset
   (unlimited transfers, $105m, relaxed country cap, shrinkage on manager means).
