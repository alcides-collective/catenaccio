# Code-audit log

The engine was reviewed twice by an external adversarial agent. Summary of findings and
the fixes applied.

## Round 1 — methodology audit
Key issues raised:
1. **Wrong objective for a leader.** EV-maximisation and a differential lean can lower
   `P(finish 1st)` even at equal EV. → Built the Monte-Carlo title-defence engine.
2. **Captain/scouting bug (MD2).** Scouting EV was attached to the captain variable, which
   wrongly made captaincy depend on the bonus. → Caught pre-deadline (captain switched from
   a cheap differential to Mbappé) and fixed structurally for MD3 (`X = base + scout`,
   `C = base only`).
3. **Unfit `p4`.** The linear clip for `P(pts>4)` was a heuristic. → Replaced with a fitted
   logistic (`engine/calibrate_p4.py`).
4. **False precision.** Per-player RMSE (~3.7) swamps the 0.02–0.6 EV margins decisions were
   made on. → Acknowledged; transfers now require a real margin over the free plan.
5. **Bookmaker overround** not removed before probability conversion. → Open.

## Round 2 — validation of the rebuild
- **p4 calibration: PARTIAL.** Random CV was optimistic; under leave-one-round-out the
  Brier edge vanishes (logistic 0.1837 vs clip 0.1802) though log-loss still improves
  (0.55 vs 1.15). Within sub-5% players, **high-projection** ones are still overestimated
  (~+0.33 on `P>4`) — exactly the players the optimiser picks. → `calibrate_p4.py` now
  reports GroupKFold + leave-one-round-out and warns against over-trusting differential
  scouting EV.
- **Monte-Carlo engine: PARTIAL.** Shared captain draws and history-anchoring are the right
  fixes. Raw two-round history likely overstates persistence of a hot start → shrinkage is a
  roadmap item.
- **Captain conclusion: PARTIAL/robust-with-caveat.** Mbappé over Haaland reproduces and is
  stable to simulation parameters, but **flips if the nearest rival captains Haaland.** The
  most fragile assumption is the rival captain prior.
- **Hit-threshold bug** (`transfer_md3.py`): the code recommended a hit for any positive ε.
  → Fixed: a hit must beat the best free plan by a real margin.

## Net effect
The rebuild moved the system from an EV calculator toward a leader-aware decision engine.
It is **useful as a constraint + variance engine, not yet a full title-probability
optimiser** — that is the next build for the knockout stage.
