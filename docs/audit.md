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

## Round 3 — R32 chaser rebuild
Audited the knockout rebuild (`optimizer/squad_build_r32.py`, `engine/monte_carlo_r32.py`,
`data/projections/projections_r32.json`) after we fell to 2nd (−15) post-MD3. The objective
inverts: no longer minimise variance vs the field but maximise variance *relative to the
specific leader*. Findings and fixes:

1. **Scouting bonus omitted from the R32 objective.** The workflow projections are raw
   `r32_pts` (no scouting), so the MILP undervalued the low-owned differentials a chaser
   wants. → Added it back, but **gated to genuinely-projected players** (`src=="proj"`, not
   the form-prior fallbacks — stacking it on capped priors manufactured fake $3.5 cheap
   enablers) and **discounted 0.5** (the linear p4 ramp overestimates haul probability for
   high-projection sub-5%, per Round 2).
2. **Leader captain hard-coded to Mbappé.** The Messi-captain conclusion was fragile to this.
   → The simulator now scores each of our captains over a **leader-captain prior** (Mbappé
   .60 / Haaland .18 / Vinícius .15 / Dembélé .07) and reports the worst case. Messi
   survives: best in expectation AND its worst case (leader captains Mbappé — the modal case)
   beats every alternative's best case.
3. **Two leader scenarios don't cleanly bracket reality.** Scenario B (leader keeps a stale
   MD3 squad) overstates our edge; scenario A (leader re-optimises, XIs coincide, captain is
   the only lever) is the conservative anchor at ~+1 EV / ~51% round-win / ~12.6% to erase
   −15 in one round. A 20–50 sample leader-squad ensemble is the proper fix — deferred.
4. **Under-dispersed score model.** → Added a shared per-nation team-day shock so teammates
   move together and co-owned players cancel exactly in the head-to-head. (Opponent
   clean-sheet ↔ attacker anti-correlation still open.)
5. **Chip timing: HOLD argument refuted.** Holding the Qualification Booster only wins if the
   leader doesn't fire R32; firing weakly dominates (worst case it cancels, best case a large
   differential) and R32 has the most advancers. → Flipped to **fire at R32**.

Open / accepted: odds not de-vigged (inflates favourites, clean sheets, booster value); LLM
projections likely hot-form-biased (Swiss midfielders ~8.5); simulator keys by name not id.

## Net effect
The rebuild moved the system from an EV calculator toward a leader-aware decision engine, and
the R32 work re-pointed it from "defend a lead" to "chase a leader." It is **useful as a
constraint + variance engine, not yet a full title-probability optimiser** — a finite-horizon
head-to-head simulator (leader policy priors, booster states, correlated draws, optimising
`P(finish ahead)`) is the next build for the deeper knockout rounds.
