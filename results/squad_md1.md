# FIFA World Cup Fantasy 2026 — Final Squad ($99.9m / $100m)

**Enter before 20:00 UK / 21:00 CEST tonight (June 11).**
Formation **3-4-3** · Captain **Haaland** · Vice **Kimmich** · 6 sub-5% differentials in XI

## Starting XI

| Pos | Player | Team | Price | Owned | MD1 fixture |
|-----|--------|------|-------|-------|-------------|
| GK | Jordan Pickford | ENG | $4.8m | 14.8% | v CRO, Jun 17 |
| DEF | Joshua Kimmich | GER | $5.5m | 33.9% | v CUW, Jun 14 |
| DEF | Pervis Estupiñán ◆ | ECU | $4.8m | 2.7% | v CIV, Jun 15 |
| DEF | Manuel Akanji ◆ | SUI | $5.0m | 4.9% | v QAT, Jun 13 |
| MID | Christian Pulisic | USA | $7.0m | 5.0% | v PAR, Jun 13 |
| MID | Mohamed Salah ◆ | EGY | $10.0m | 4.4% | v BEL, Jun 15 |
| MID | Takefusa Kubo ◆ | JPN | $7.0m | 1.0% | v NED, Jun 14 |
| MID | Marcel Sabitzer ◆ | AUT | $6.8m | 3.8% | v JOR, Jun 17 |
| FWD | Kylian Mbappé | FRA | $10.5m | 49.3% | v SEN, Jun 16 |
| FWD | Erling Haaland **(C)** | NOR | $10.5m | 33.4% | v IRQ, Jun 16 |
| FWD | Jonathan David ◆ | CAN | $7.0m | 1.2% | v BIH, Jun 12 |

◆ = sub-5% owned → +2 Scouting Bonus whenever they score 4+ points

## Bench (order)

1. GK Thibaut Courtois (BEL, $4.9m)
2. MID Florian Wirtz (GER, $7.5m) — first outfield sub
3. DEF Daniel Muñoz (COL, $4.6m)
4. DEF Alistair Johnston (CAN, $4.0m)

## Why this squad

- Exact MILP optimum (scipy/HiGHS) over 564 projected players from 43-agent research:
  objective = MD1 EV + 0.5×(MD2+MD3) EV + captain doubling + differential-bonus EV.
- The ≥6-differentials constraint cost **zero** EV — the differential build is also the pure-EV optimum (after exclusions).
- Verification excluded: David Raum (lost LB spot to Nathaniel Brown), Hakan Çalhanoğlu
  (soleus strain, not nailed for MD1), Emiliano Martínez (fractured finger, missed both friendlies).
  Cost of exclusions: 1.38 EV. All 15 final picks verified as nailed starters via June 2026 sources.
- Captain Haaland = expert consensus #1 (Norway 84% win prob v Iraq; FFS top xPts 6.97).
- Per-country max 3 respected: GER 2, CAN 2, all others 1.

## Notes for later rounds

- 2 free transfers each for MD2/MD3; unlimited again before R32 (+$5m budget).
- Captain alternative if you want a pivot: Oyarzabal (ESP v CPV, pens, Spain 92% win prob).
- Files: projections.json (per-player EVs), optimize.py (re-run with --exclude / --no-diff-constraint).
