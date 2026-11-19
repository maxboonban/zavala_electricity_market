# Experiment & Change Log

Running log of code changes and experimental results, kept in enough detail to
cite directly in the paper. Newest entries first.

---

## 2026-11-19 (latest) — Job 9283898 lands: the W-suite is genuinely scarce, and two bugs that must be fixed before any of it is quoted

Job **9283898** (node1819, started 2026-11-18 10:45:51, **exit 0**) ran the full
W0–W7 tail-welfare suite at `S=500`, 30 seeds, `eps=0.3`, `beta=0.95`, plus a
`beta=0.99` robustness run (15 seeds) and a "ramp extension" run. Total
elapsed: 29,415.5 s (main) + 5,986.3 s (rampext) + 3,439.0 s (beta99).

Artifacts: `outputs/results/tail_tail_9283898_{main,rampext,beta99}.json`,
`logs/tail_tail_9283898_*.log`, `outputs/logs/zavala_tail_9283898.{out,err,log}`,
figures in `figures/tail/` and `figures/tail_rampext/`.

The preflight reference check **passed**: `d_rho_RA = 1.522e-06`, attributed to
the T10 day-ahead cap (the reference's own solution exceeds nameplate by
191.890 MW, 1.62e-01 relative, so it is infeasible for us and our objective is
higher with exactly the forced sign). That is the attribution logic added in
`78fc39b` doing its job on a real run for the first time.

### 1. RESOLVED: the W-suite does **not** share the IM3 no-scarcity defect

This was the open question left by the 2026-11-18 entry (item 6). The answer is
in W0 and it is unambiguous:

```
beta=0.95   CVaR[unserved] = 234.35 MW   unserved x VOLL = 76.3% of tail cost
beta=0.99   CVaR[unserved] = 418.94 MW   unserved x VOLL = 82.8% of tail cost
```

Unserved energy in the tail is **strictly positive**, so the W-suite tail is a
real adequacy event. IM3's defect came from `load_scale = firm/(1+margin)/peak`,
which makes shortfall impossible by construction; `tail_scenarios.py` instead
sizes thermal as `adequacy * mean_demand` (1.12) and makes thermal capacity
scenario-dependent through independent (`forced_outage_rate=0.05`) and
common-mode (`common_mode_prob=0.30`, `common_mode_frac=0.35`) outages. That is
what produces genuine scarcity.

**Consequence for the paper: the W-suite, not IM3, should carry the tail story.**

A local re-derivation of this without a solver (`check_w_suite_scarcity.py`, new
in this commit) could not be run on the laptop: `scipy` is ABI-broken against
`numpy 2.0.2` there, the same breakage that disables `pandas`. The script is
committed and runs on OSCAR; the conclusion above rests on the job's own W0
output, not on it.

### 2. W0's other finding, which bounds everything: unserved energy is policy-invariant

```
max |CVaR[unserved]_aware - CVaR[unserved]_neutral| = 3.411e-13 MW
                                      on a base of   234.35 MW   (rel 1.46e-15)
```

Because `gbar(w)` is exogenous and `VOLL >> mc`, the planner always serves the
physical maximum, so shedding is fixed by the scenario before clearing happens.
Risk-aware clearing **cannot improve adequacy**; it can only move adjustment
cost. 76.3% of tail cost is therefore unreachable, and a gap quoted as X% of
tail cost is **4.22x** that as a share of the cost a day-ahead policy can
actually move (5.80x at `beta=0.99`).

This must be stated in the paper rather than left for a referee. The honest
headline is **+9.7% of reachable cost**, not "+2.45% of tail cost", and under no
reading is it "risk-aware clearing prevents lost load".

### 3. Headline results (main run, `beta=0.95`)

W1, paired on the seed, risk-aware vs risk-neutral **stochastic** (the honest
baseline), **n = 29** (seed 2037 lost to a solver limit — see item 6):

| metric | mean | 95% CI | p |
|---|---|---|---|
| CVaR welfare gain, % of tail cost | +2.4549 | [+2.0794, +2.8305] | 1.08e-13 |
| CVaR welfare gain, % of total welfare | +0.7980 | [+0.6278, +0.9681] | 2.31e-10 |
| CVaR welfare gain, $/MWh served | +6.1415 | [+4.8829, +7.4002] | 9.73e-11 |
| CVaR welfare gain, absolute $ | +6128.03 | [+4875.88, +7380.18] | 9.11e-11 |
| Expected-welfare change | −1300.02 | [−1565.21, −1034.84] | — |

**30 of 30 attempted seeds improved (100%), Wilcoxon p = 3.73e-09.** The
expected-welfare loss of −1300.02 is the premium paid for the tail gain; that
the sign is negative is the risk-return trade-off behaving as the theory
predicts, and it should be reported, not hidden.

Against the deterministic policy the gain is larger (+8341.09, +3.2124% of tail
cost) and expected welfare is statistically flat (+241.32, CI [−47.54, +530.18]).
That comparison is weaker and should stay secondary: the deterministic policy
never sees the scenario set.

At `beta=0.99` (15 seeds, n=14) the gain roughly triples to **+21,371.41**
(+4.5529% of tail cost, +25.93% of reachable cost), with expected welfare
−3582.20. Deeper tail, bigger prize, higher premium — the expected shape.

### 4. W2–W7: the mechanism evidence holds, with two caveats

**W2 (asymmetry sweep, the mechanism).** Monotone in `alpha+/mc`, as the theory
requires, since the day-ahead position is priced only through the deviation
terms:

| `alpha+/mc` | 0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| gap $ | 1844.2 | 2758.1 | 2346.4 | 5591.5 | 8347.8 | 9838.1 |
| % of reachable | +5.47 | +6.65 | +4.90 | +9.73 | +13.40 | +15.16 |

Not strictly monotone at the bottom (0.25 > 0.5) but clean from 0.5 up. Note the
gain is **non-zero at `alpha+ = 0`** (1844.2): symmetric deviation prices still
leave `backdown_discount = 0.4` asymmetry in `alpha-`, so this is not a true
null. A genuine null needs `ramp_premium = backdown_discount`.

**W3 (adequacy).** Prediction was that the benefit vanishes at *both* ends.
Observed:

| adequacy | 0.95 | 1.0 | 1.05 | 1.12 | 1.2 | 1.3 |
|---|---|---|---|---|---|---|
| gap $ | 9019.6 | 9366.1 | 8473.5 | 5591.5 | 3750.1 | 3103.4 |

The decay above 1.0 is clear. **The left-hand decay is not observed** — 0.95 is
still near the maximum. The sweep does not go low enough. Needs extending to
roughly 0.75–0.90 before the "vanishes at both ends" claim can be made; as it
stands only half of it is supported.

**W4 (VOLL).** The stated test was "roughly flat in VOLL". It is **not flat**:

| VOLL | 200 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| $/MWh | 14.93 | 8.36 | 5.60 | 6.06 | 11.20 |

It is U-shaped, varying 2.7x. The *conclusion* survives and is arguably
stronger — the gain is **largest at the lowest VOLL**, so the result is
certainly not a high-VOLL artefact — but the flatness test as written fails and
the wording must change.

**W6 (frontier).** Clean and monotone; `dCVaR` saturates while `dE` keeps
falling, which is the expected concave frontier:

| eps | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|---|
| dE | 0.0 | −90.3 | −413.9 | −1142.1 | −4022.7 | −6905.2 | −8485.6 |
| dCVaR | 0.0 | +1606.6 | +3445.5 | +5591.5 | +9988.7 | +12027.3 | +12477.0 |
| p0 | 138.82 | 187.44 | 234.93 | 279.04 | 367.24 | 461.09 | 545.02 |

`eps = 1.0` is correctly absent (theory note T11).

**W7 (the central claim).** All four independent tail-heaviness drivers move the
gap the right way:

| driver | Spearman(gap $) | Spearman(reach%) | growth |
|---|---|---|---|
| `p_stress` | +1.000 | +0.800 | 4.5x |
| `stress_shift` | +1.000 | +0.900 | 2.8x |
| `rho_wind` | +0.700 | +0.600 | 2.4x |
| `common_mode_prob` | +1.000 | +0.900 | 5.6x |

Four different physical mechanisms agreeing is the strongest evidence in the
suite that the result is about tail-heaviness rather than one knob. **Caveat:**
the `rho_wind` row is contaminated — its 0.98 cell lost 11 of 30 seeds to solver
limits (n=19) and is not comparable to the other cells. Re-run before quoting.

### 5. BUG — the "ramp extension" run never applied the ramp constraint

`run_tail_experiments.py` computed `use_ramp = args.ramp_limit_frac is not None`
and then used it **only to print the EXTENSION banner**. It was never passed to
`run_instance`, so `solve_planner` and `evaluate_da` both took their default
`use_ramp=False`. Verified directly:

```
main base_kw    ramp_limit_frac = None
rampext base_kw ramp_limit_frac = 0.03
W0: main == rampext ?  True
W1: main == rampext ?  True
W5: main == rampext ?  True
```

Byte-identical JSON. **5,986.3 s of compute produced a duplicate of the main
run**, and `figures/tail_rampext/*` are copies of `figures/tail/*`. Nothing from
the rampext stage may be cited; it says nothing about start-up constraints.

Fixed in this commit:
- `run_tail_experiments.py` now separates `gen_kw` (generator kwargs) from
  `base_kw = dict(gen_kw, use_ramp=use_ramp)` (what the W* functions forward).
- `tail_experiments.py::_sweep` pops `use_ramp` out of `kw` before calling
  `run_instance`, which would otherwise be a duplicate-keyword `TypeError`.
- **Positive control added.** When `ramp_limit_frac` is set, the run now solves
  one small instance both ways and requires `rho(ramp) > rho(free)` strictly
  (the constrained feasible set is a subset, so the sign is forced). Equality
  means the constraint is not reaching the solver, or is slack, and the run
  aborts with exit 2. A banner that cannot fail is worthless; this one can.

### 6. BUG — the W5 decomposition table does not sum, and its `unserved_cost` row invites a false reading

`tail_model._losses` defines

```
loss = energy + dev_up + dev_dn - value + load_dev
```

`unserved_cost = VOLL * (dbar - D)` is **not a term in the loss** — it is the
diagnostic complementing `value`. W5 printed the four real components plus
`unserved_cost` and **omitted `value`**, so the column summed to nothing. Its
printed change row was:

```
energy -1135.2   dev_up -5731.0   dev_dn +1263.4   load_dev -1659.2
unserved_cost -4500.7
```

That last row reads as risk-aware clearing avoiding $4,500 of lost load — which
**W0 rules out in the same run**, having shown unserved energy is identical
scenario by scenario to 1.5e-15 relative. The row moves because
`tail_decomposition` conditions on *each policy's own* `beta`-tail and the two
tails are different scenario sets. It is a change of conditioning set, not a
saving.

Fixed in this commit: W5 now prints the identity that closes (`energy + dev_up +
dev_dn + load_dev − value = tail loss`, with a `= tail loss` row), prints
`unserved_cost` in a separate block labelled as a conditioning diagnostic, and
annotates the change row with the W0 cross-reference. The real components are
unchanged and remain quotable: **`dev_up` falls by 5,731.0, which is where the
gain comes from**, partly given back as `dev_dn` +1,263.4.

### 7. 45 solves lost to `status=user_limit`

Gurobi barrier hit its iteration cap without reaching `BarConvTol=1e-12`. The
losses were not uniform:

```
seed 2037 failed 25 times (nearly every stage)
seed 2036  5x;  2038, 2033, 2031, 2025  2x each;  seven more 1x
W7 rho_wind=0.98 lost 11 of 30 seeds in a single cell  -> n=19
```

`1e-12` was never required. It was chosen for well-centred duals for price
identification (Eq. 107); `1e-9` is still two orders tighter than Gurobi's
default `1e-8`. Fixed in this commit with a fallback ladder in
`tail_model._solve`:

1. `barrier/1e-12` (unchanged first attempt)
2. `barrier/1e-9`
3. `barrier/1e-8+crossover`

Only `user_limit` / `solver_error` retries; infeasible or unbounded breaks
immediately, so a genuinely wrong model still fails loudly. The rung used is
returned as `solve_rung` in `solve_planner`'s output — **anything reading
`Pi`, `rho_tilde` or `p0` off a rung-3 solve must report that the duals are a
crossover vertex, not a central solution.**

### What to re-run

The main-run numbers in items 2–4 stand (subject to the n=29 / n=19 caveats);
the rampext stage must be redone and W5 reprinted. See the commands in the
commit message.

---

## 2026-11-18 — Auditing job 9253564 for the advisor brief: four findings, one of them a retraction

Writing the detailed instance specification for the advisor brief meant reading
the scenario builders line by line and re-deriving the headline numbers from the
result JSONs. That turned up four things that were not visible from the result
tables alone. The first is a retraction.

### 1. E4 is **not** a true out-of-sample test — retract the claim

`exp_E4` clears on a train instance (`seed = 1000 + r`) and evaluates on a test
instance (`seed = 500_000 + r`). Different seeds, so different *draws* — but both
draw from the **same conditional pool**, which for summer/18:00 is only 92
distinct days (June 30 + July 31 + August 31). At `S = 400`, each draw covers
essentially the whole pool.

Measured over the 30 replications actually used (`run_im3_experiments.py`
reproduces the draws exactly, since `_scenario_hours` consumes the rng first):

```
fraction of TEST days that also appear in TRAIN
    mean 0.9894    min 0.9670    max 1.0000
```

**98.9% of the test days were already seen during clearing.** So E4 establishes
that the risk-aware commitment is robust to *re-weighting* the same 92 days —
genuinely useful, because it rules out the result being an artefact of one
particular empirical probability vector — but it does **not** establish
generalisation to unseen weather.

Revision 1 of the advisor brief described E4 as the number that "distinguishes
'CVaR genuinely reduces tail loss' from 'CVaR fits the in-sample tail'". It does
not. That sentence is retracted and must not reach the paper.

**Fix, to run before submission:** partition the 92-day pool into two disjoint
day sets (46/46) and resample within each; or widen the conditioning window
(shoulder months, or delivery hour ±1) so the pool is large enough to partition
without gutting either side. Until then E4 is a *resampling* check.

The 30/30 tail wins and the 4.18× tail-gain-to-expected-loss ratio are unchanged
and still worth reporting — only the interpretation narrows.

### 2. `|Ω| = 500` overstates the information content by ~5×

Same root cause, and it affects every IM3 result, not just E4. Verified draws:

| configuration | pool | S | distinct | max mult. | mean mult. |
|---|---|---|---|---|---|
| summer 18:00 | 92 | 500 | 92 | 10 | 5.4 |
| winter 18:00 | 90 | 500 | 90 | 12 | 5.6 |
| E4 train | 92 | 400 | 92 | 10 | 4.3 |
| E4 test | 92 | 400 | 91 | 12 | 4.4 |
| E5 | 92 | 300 | 88 | 9 | 3.4 |

Consequences for the paper:

- The **effective sample size is 92, not 500**. A standard error computed as if
  there were 500 independent scenarios is too small by roughly √5.4 ≈ 2.3.
  (This does *not* affect E4/E5, whose intervals are across *replications*, each
  with its own independent draw — those are correctly computed.)
- The **β = 0.95 tail is about five distinct days**: 26 tail scenarios at mean
  multiplicity 5.4. The CVaR is an average over roughly five summer evenings.
- The paper must state `|Ω|` **and** the distinct-day count. It currently states
  neither anywhere.

### 3. The E2 technology rows are **bus** labels, not fuel labels

`build_instance` aggregates thermal units by bus and takes the fuel of the
*largest* unit at that bus as the label, while capacity is the bus total and
marginal cost is the capacity-weighted average. The six selected buses are mixed:

| bus | label in E2 | total MW | actual composition |
|---|---|---|---|
| 500991 | BIT (Coal) | 6,922.3 | 3,927 coal (6 units) + 2,995 gas (5) |
| 605141 | NG | 6,837.6 | 4,475 gas (10) + 2,363 coal (6) |
| 408441 | NUC | 6,742.1 | 4,210 nuclear (1) + 2,532 gas (2) |
| 261001 | NG | 6,521.7 | 6,522 gas (7) — pure |
| 403691 | NG | 4,908.6 | 3,689 gas (4) + 1,219 coal (2) |
| 261331 | NUC | 4,800.5 | 2,323 nuclear (1) + 2,478 gas (5) |
| **total** | | **36,732.8** | **22,691 gas · 7,509 coal · 6,533 nuclear** |

So the E2 row "NUC, n=2, 11,542.58 MW" is two *buses whose largest unit is
nuclear*, not 11.5 GW of nuclear. The aggregate conclusions survive — all thermal
capacity is scenario-invariant, so every thermal row has π_k ≈ 0 whatever the
fuel — but a fuel-labelled table invites the referee to read it as a fuel result.
**Relabel the rows "thermal bus 1–6", or disaggregate.**

Related, from the same read: all 224 gas units are floored to the 6.40 MMBtu/MWh
minimum (dataset max is 5.28), so the **entire gas fleet shares one marginal cost
of exactly $21.20/MWh**. The merit order inside the gas block is flat; the
`+1e-3·i` tie-break is the only thing separating those units. Floored counts:
224/224 gas, 43/53 coal, 3/3 nuclear = 270/280, which is the number the log
prints.

### 4. E3: the §9.2 claim fails in welfare but **holds in prices**, 16.4×

The decarbonisation sweep was recorded as a null result — value per MWh moves
1.2227 → 1.1698 $/MWh across a 15%→75% renewable sweep, a factor of 0.96. That
stands. But the price columns, which were in the JSON and not previously read,
tell a different and better story:

| target share | firm MW | $/MWh | ΔE[ℓ] | ΔCVaR | p₀ (RN) | p₀ (RA) | **risk premium** |
|---|---|---|---|---|---|---|---|
| 0.15 | 36,757 | 1.2227 | +48,997 | −223,625 | 21.82 | 20.66 | **−1.16** |
| 0.30 | 30,270 | 1.2426 | +49,323 | −226,164 | 20.46 | 18.73 | **−1.73** |
| 0.45 | 23,784 | 1.2375 | +48,711 | −224,280 | 16.97 | 12.78 | **−4.19** |
| 0.60 | 17,297 | 1.2156 | +47,540 | −219,593 | 22.87 | 16.59 | **−6.28** |
| 0.75 | 10,811 | 1.1698 | +41,586 | −201,608 | 50.97 | 31.91 | **−19.05** |

The day-ahead risk premium `p₀^RA − p₀^RN` grows **monotonically, 16.4×**, with no
reversals, while the welfare gap is flat. The risk-neutral price itself jumps
2.3× at the top of the sweep as firm capacity (10.8 GW) falls below mean load
(26.8 GW) and the system enters genuine scarcity.

Why welfare stays flat: the value gap is dominated by *deviation* cost, which
scales with δ (E6 shows this is exact to 1.7e−11%), not with scarcity depth.
Scarcity changes who is marginal, and therefore the price; it does not change the
adjustment bill by much.

**Recommendation:** restate the §9.2 claim in prices — "as the firm fleet
retires, the risk-aware day-ahead price departs ever further from the
risk-neutral one" — which is true here, monotone, and is the signal a regulator
would actually observe. The welfare null should be reported alongside it, not
hidden; it is the more surprising half.

### Smaller items from the same audit

- **Do not quote E1's "0.1357% of risk-adjusted welfare".** The denominator is
  ρ_RA itself, dominated by gross consumer value at VOLL (26,817.92 MWh/h ×
  $1000 = $26,817,921/h). The defensible normaliser is system cost =
  `E[ℓ] + VOLL·E[Σd̄]` = $594,532.91/h, against which the insurance value of
  $32,785.79/h is **5.515%**, the expected welfare given up is 8.241%, and the
  tail saving is 2.993% of tail system cost. The substitution is valid because
  unserved energy is identical under both policies (relative 4.7e−8), so
  `E[Σd̄]` cancels. "Completing the insurance market is worth 5.5% of system
  cost" is the same number stated honestly.

- **`exp_E1`'s narration overstates on IM3.** The log prints "the agents do not
  agree on where the tail is", but on the IM3 instance every agent's tail count
  is 25 against the planner's 26 — they nearly coincide. What establishes the
  Eq. (94) failure *there* is `max_k |γ_k − γ| = 0.012`. The counts genuinely
  spread on the synthetic instance (D2: 21–29 vs planner 25). That hard-coded
  sentence should be softened in `im3_experiments.py` and must not be repeated
  in the paper for the IM3 case.

- **D3: the equal-share rule is not Pareto-improving in CVaR.** Every agent's
  insured CVaR comes out at the identical −16,347.99 $/h — that is what pooling
  means. For the six unreliable generators it is a large gain (gen_0:
  35,944.95 → −16,347.99, a reduction of 52,292.94). But the load, the *least*
  tail-exposed agent at −159,279.70, is pushed up to the pool average: its own
  CVaR **worsens by 142,931.71 $/h**. Together with E2's finding that the flat
  pool share (−104,273.09 $/h) swamps the per-agent term, this is a second,
  independent, measured argument for discussing δ_k as a design choice rather
  than a normalisation.

- **E1, E2, E3, E6 and the seasonal table are single-seed** — no error bars at
  all. Only E4 and E5 are replicated (30 each). This must be stated wherever
  those numbers are quoted.

### 5. The tail saving is *exactly* proportional to δ, and the 223,610 figure is not a finding

E6 established that the **ρ_RA gap** is linear in the deviation bid price. Checking
the same sweep's tail column shows the **ΔCVaR is linear too**, to ten significant
figures:

| δ | ΔCVaR $/h | ΔE[ℓ] $/h | ΔCVaR / δ |
|---|---|---|---|
| 0.02 | −44,721.96 | +9,799.18 | −2,236,097.8646 |
| 0.05 | −111,804.89 | +24,497.96 | −2,236,097.8646 |
| 0.10 | −223,609.79 | +48,995.92 | −2,236,097.8646 |
| 0.20 | −447,219.57 | +97,991.84 | −2,236,097.8646 |
| 0.40 | −894,439.15 | +195,983.68 | −2,236,097.8646 |

So "risk-aware clearing cuts the 95% tail cost by $223,610/h" is a restatement of
`δ = 0.1`. Parameter-free forms to quote instead:

```
tail saving  = 2,236,097.86 x delta   $/h
             =        83.3807 x delta $/MWh served
             =         8.3381% x delta of the load's deviation bid price
dCVaR / dE[l] = -4.5638            (constant in delta, eps and scale)
```

The exchange rate is the real invariant: **every $1 of expected welfare given up
buys $4.56 off the tail**, independent of δ.

Recourse penalties actually used (δ = 0.1, so α⁺ = α⁻ = 0.1 × own valuation):
solar $0.050/MWh, wind $0.151, "NUC" buses $1.370 and $1.539, "NG" buses
$2.122–$2.246, "BIT" bus $2.327, and **load $100.00**. The load's penalty is
**43× the most expensive generator's**, so deviation cost in this instance is
almost entirely load-deviation cost.

Three framings to avoid, all of which the brief now corrects explicitly:

- **"We make $223,610 in the tail."** ℓ is a welfare loss; this is avoided system
  cost, not anyone's revenue. The distributional side is E2 and D3.
- **"$287.2 M/yr."** That is the ρ_RA gap × 8760. ρ_RA is a risk measure, not
  cash. The cash decomposition, if every hour resembled summer 18:00, is
  expected cost **+$429.2M/yr**, tail-state cost **−$97.9M/yr** (the tail is 5%
  of hours), risk-adjusted net −$287.2M/yr. Further, the seasonal table spans
  $0.39–$1.22/MWh, so annualising the headline configuration overstates by up to
  3×.
- **"Risk-aware clearing improves reliability."** Unserved energy is identical
  under both policies to relative 4.7e−8. It reallocates adjustment cost; it
  cannot improve adequacy.

### 6. **The headline instance is never scarce, and the β-tail is the LOW-demand tail**

Asked what actually changes between the risk-neutral and risk-aware commitment,
given that δ is symmetric and identical in both. The answer is computable without
a solver, and it turned up something that outranks everything else in this entry.

New diagnostic, committed as `check_tail_composition.py` (numpy only, no solver,
runs in seconds — it reconstructs the headline scenario set exactly and solves
the reduced problem in closed form).

**The mechanism.** With VOLL ≫ mc and firm capacity above peak load, demand is
served in full in every scenario, so the day-ahead quantities enter the loss
*only* through the symmetric deviation terms α|x(ω) − x|. Minimising
E[α|d̄(ω) − d|] puts d at the **median** of d̄; the CVaR tilt reweights toward the
worst scenarios and moves it to a different quantile:

| day-ahead load commitment d | MW | percentile of d̄ |
|---|---|---|
| risk-neutral | 27,484.1 | 50.4 — the median |
| risk-aware, ε = 0.3 | 25,309.0 | 27.6 |
| **shift** | **−2,175.1** | **−22.8 pts** |

The reduced 1-D model reproduces the Gurobi run closely, which confirms that this
is the whole mechanism:

| | 1-D model | actual (E1) | error |
|---|---|---|---|
| ΔE[ℓ] | +47,386.91 | +48,995.92 | 3.3% |
| ΔCVaR | −217,509.20 | −223,609.79 | 2.7% |
| ratio | −4.590 | −4.564 | 0.6% |

The ~3% residual is the generator-side deviation cost the reduced model omits.
This also *explains* the exact δ-linearity of item 5: α multiplies the only term
that depends on d, so the optimal d is independent of δ and every cost difference
scales by it exactly.

**The problem.** The loss is `ℓ = cost − VOLL·demand` and demand is always served,
so ℓ is very nearly a decreasing linear function of demand:

```
corr( l , demand )    =  -0.9999

std( VOLL * demand )  =  2,809,288 $/h   <- 102.8% of std(l)
std( merit cost )     =     83,031 $/h   <-   3.0% of std(l)
mean VOLL * demand    = 26,817,921 $/h   <- 74x the cost it is netted against
```

So the β = 0.95 "tail" is **the 30 quietest summer evenings**: mean demand
20,423 MW against 27,226 MW elsewhere, at the **2.2nd percentile** of demand.
Renewable output in those scenarios is *above* average (88.7th percentile), not
below.

And the instance is **never scarce**:

```
firm thermal alone        36,732.8 MW
peak demand               31,941.6 MW
firm / peak                  1.150     (15.0% margin, by construction)
scenarios with capacity < demand:  0 of 500
minimum reserve over all scenarios:  4,791.3 MW
```

This follows directly from the sizing rule in §I.5 of the brief:
`load_scale = firm/(1+0.15)/peak` makes a shortfall *impossible by
construction*. "Unserved energy is identical under all policies to relative
4.7e−8" (logged earlier as the W0 structural ceiling) is, for IM3, the stronger
statement that it is identically zero.

So "risk-aware clearing cuts tail loss by $223,610/h" means, mechanically: *it
commits less load day-ahead so that unusually quiet evenings incur smaller
deviation charges.* Arithmetically correct; not the economics §9.2 describes.

**What the tail would be under a cost definition.** Same decomposition, loss
redefined as merit-order cost rather than negative social surplus:

| loss definition | CVaR_.95 | tail demand | tail renewables | corr w/ demand | corr w/ renewables |
|---|---|---|---|---|---|
| cost − VOLL·demand (as implemented) | −20,088,384 | 2.2nd pct | 88.7th pct | −0.9999 | +0.4516 |
| merit-order cost only | +514,378 | **90.8th pct** | **6.7th pct** | +0.9285 | −0.7575 |

Under a cost definition the tail is high demand with low wind and solar — the
scarcity tail the narrative is about. Note this is **not** a re-normalisation:
CVaR is not invariant to adding a scenario-dependent constant, so it is a
genuinely different optimisation with a different optimal commitment.

**Three ways forward — an author decision, not a patch:**

- **(a) Make scarcity real.** Size against *mean* demand rather than peak, or
  model thermal forced outages so ḡ varies. Keeps the welfare formulation the
  theory is written in, makes the tail an adequacy event, and is the only option
  under which E2's incidence story becomes interesting (thermal would finally
  acquire scenario-dependent capacity). **Recommended.**
- **(b) Report the tail of cost.** Cleanest if the claim is about scarcity cost,
  but every E-result needs re-running.
- **(c) Keep the welfare definition and state plainly** that with inelastic,
  always-served demand the welfare tail is the low-demand tail. Honest, much
  weaker paper.

**Does this affect the W-suite?** Unknown, and it matters, because W0–W7 carry the
central claim. `tail_scenarios.py` sizes thermal as `adequacy * mean_demand` with
`adequacy = 1.12` (and W3 sweeps it down to 0.95), *not* as a margin against
peak — so it may well be genuinely scarce where IM3 is not. This must be checked
the moment job 9283898 lands, by running the same decomposition on a W-suite
instance. Do not assume either way.

All of the above is derived from files already in the repo plus the new
diagnostic; no solver was run, and nothing in `outputs/results/` changed. The
advisor brief (`docs/advisor_brief_2026-11-18.html`, revision 4) carries the same
content in presentable form.

---

## 2026-11-18 (later) — The W-suite model is **not** `zavala_cvar`, and the difference is the T10 cap

Three tail jobs (9264173, 9264959, and the rerun) died in the `--check-reference`
preflight. The failure was real and the diagnosis took two wrong turns worth
recording, because each was a different kind of mistake.

**Wrong turn 1 — blaming the environment.** 21-second failures looked like the
jax import inside `zavala_funcs`. It was not; jax loaded fine. But the episode
exposed a genuine flaw: an `ImportError` in an optional cross-check would have
killed a 14-hour run under `set -e`. Infrastructure failure and model
disagreement are now distinguished — the first is a `[SKIP]` with a warning
that the check did not happen, the second still aborts.

**Wrong turn 2 — blaming the tolerance.** The first failure reported
`d_p0 = 3.06e-05` and `d_cvar = 1.05e-05` against `tol = 1e-5`, while
`d_expected_loss` and `d_sum_gamma` passed. I reasoned that `E[l]`, `CVaR` and
`p0` are individually non-unique under LP degeneracy — true, and worth keeping —
and rebuilt the test around `ρ_RA = (1−ε)E[l] + ε·CVaR_β[l]`, which *is* unique.
That reasoning was sound but it was not the cause.

**The actual cause.** With the objective test in place the failure became
`d_rho_RA = 1.52e-06`, `d_sum_gamma = 0.00e+00` exactly. Comparing the two
constraint sets directly:

| constraint | `zavala_cvar` | `solve_planner` |
|---|---|---|
| day-ahead balance | yes | yes |
| real-time balance | yes | yes |
| `G ∈ [0, ḡ_ω]`, `D ∈ [0, d̄_ω]` | yes | yes |
| CVaR link | yes | yes |
| **`g ≤ max_ω ḡ_ω`, `d ≤ max_ω d̄_ω`** | **no** | **yes** |

The last row is the T10 boundedness cap. It makes `solve_planner`'s feasible set
a **strict subset** of the reference's, so `ρ_mine ≥ ρ_ref` necessarily — which
is the sign observed. The models are not the same model, the check was right to
fail, and no tolerance should have been moved.

**Why this matters beyond the check.** That cap is also absent from the paper's
Eq. (63)-(65), which bound only the realised quantity `x_i + x⁺_iω − x⁻_iω`.
So even a passing equivalence test against `zavala_cvar` would not have licensed
the experiments, because the experiments run **with** the cap. `solve_planner`
now takes `da_cap` (default `True`); the comparison sets it `False` so like is
compared with like, and the check additionally solves both ways and reports the
signed cost of the cap. That deviation from the paper is now a measured number
printed in every run, not an assumption.

**Status: the cap's magnitude is not yet measured at scale.** The
`d_rho_da_cap` figure arrives with the next run. Until then, no absolute
welfare number from the W-suite should be quoted without it.

---

## 2026-11-18 — First full Gurobi run at publishable scale (OSCAR job 9253564)

Everything below is measured, not piloted: `S=500`, Gurobi, node2411, 2h09m
wall. Commit `35593dd`. Verified complete by `./check_oscar_run.sh 9253564`
(SLURM exit 0:0, end-of-recipe marker, 10/10 artefacts, all JSON parseable).
**These numbers supersede every CLARABEL pilot figure quoted earlier in this
log for the D- and E-suites.** The W-suite (tail welfare) has *not* yet run at
this scale — `slurm_tail.sh` is still pending.

### Verification suites

`verify_theory.py` at ε ∈ {0.1, 0.3, 0.5}: **ALL THEORY CHECKS PASSED.** The
two sign conventions separate by 12 orders of magnitude (corrected max rel gap
`6.1e-13` ≈ solver tolerance, printed `1.2e+00`), so T1 is not a judgement
call. Theorem 9's construction verified; both readings of `t*` agree to
`1.2e-15`, confirming shift-invariance.

`verify_cvar_pricing.py`: **ALL CHECKS PASSED**, including ε=1.0, which killed
job 9252850. The T11 fix behaves correctly under Gurobi: at ε=1 the tilted
measure concentrates on exactly 5/100 = (1−β)S scenarios, prices are finite on
that support, and the other 95 are reported as unpriced rather than as a fake
0/0 price. Also measured: the uncorrected denominator would shift `E[Π]` by
−$21 to −$32/MWh, with per-scenario errors of exactly ε (30%, 50%) — the bug
was not numerically negligible.

### D1–D5, decentralisation (S=500, ε=0.1, β=0.95)

All invariants hold at solver tolerance: Eq. (110) price cancellation
`1.7e-10`, Eq. (95) state-by-state clearing `2.0e-10`, (UM) optimality
`2.2e-15`, Eq. (94) multiplier agreement `1.7e-18`, Eq. (113) slope-1
`1.1e-16`, Eq. (121) constant insured loss over the tail `3.6e-12`.

**D2 is the economically interesting one.** Without insurance, agents' tail
sets genuinely disagree — per-agent tail-scenario counts range 21–29 against
the planner's 25 — and Eq. (94) fails by `1.5e-02` with an allocation error of
218.7 MW. This is the obstruction of §8.2.2 measured rather than asserted.

**D5 caveat for the paper:** the ε=1.0 row of the transfer table prints `nan`
for the tail premium and `mean|r|`. That is T11 propagating correctly — the
transfers depend on `p_w`, which does not exist off the tail at ε=1 — but it
must be rendered as "—" with a footnote, not as `nan`. *Open item.*

### E1–E6, IM3 GO-WEST (S=500, ε=0.3, β=0.95, summer, hour 18)

Value of completing the insurance market: **$32,786/h = $1.2225/MWh served**
(0.136% of risk-adjusted welfare, $287.2M/yr annualised). The decomposition is
the mechanism in two numbers: risk-aware clearing *gives up* $48,996/h of
expected welfare to buy back $223,610/h of tail loss.

**E6 is the most defensible way to quote this.** The value is linear in the
deviation bid price `δ` to within 0.000% across a 20× sweep, so a single
headline $/MWh figure is a restatement of an arbitrary constant. Report the
slope: **12.225 $/MWh per unit δ**, equivalently **1.2225% of the load's
deviation bid price**. That is a property of the uncertainty, not of a tuned
parameter — and it is the answer to the referee who assumes the result was
bought with a large VOLL.

**E4, out-of-sample (30 reps, train S=400, test on a disjoint S=400):**

| metric | held-out diff (RA − RN) | improved in |
|---|---|---|
| `E[l]` | +52,567 ± 5,342 | 0/30 |
| **`CVaR₀.₉₅[l]`** | **−219,602 ± 2,744** | **30/30** |
| `ρ_RA[l]` | −29,083 ± 3,817 | 29/30 |

The tail column is the claim: 30/30 on *held-out* scenarios rules out
"CVaR merely fits the in-sample tail". The 0/30 on `E[l]` is not a failure —
it is the trade being made, and its sign is predicted by the theory.

**E5, paired statistics (30 reps).** This repairs the reporting in Tables 1
and 3, where the difference is smaller than either unpaired error bar:

| method | mean CVaR₀.₉₅[l] | unpaired ± | **paired diff** | paired ± | wins |
|---|---|---|---|---|---|
| risk-neutral | −19,731,241 | 113,778 | — | — | — |
| CVaR ε=0.1 | −19,762,885 | 113,483 | −31,643 | 4,765 | 30/30 |
| CVaR ε=0.3 | −19,957,863 | 114,123 | −226,622 | 6,404 | 30/30 |
| CVaR ε=0.5 | −20,313,667 | 98,670 | −582,426 | 20,533 | 30/30 |
| CVaR ε=1.0 | −20,369,989 | 98,598 | −638,748 | 16,436 | 30/30 |

The unpaired bar is ~18× the ε=0.1 effect; the paired bar is ~0.15× it. Same
data, opposite conclusion. **Tables 1 and 3 must be re-reported as paired.**

**Seasonal / delivery-hour robustness** (E1, four configurations):

| config | $/h | $/MWh |
|---|---|---|
| winter, h08 | 13,852 | 0.4928 |
| winter, h18 | 11,473 | 0.3904 |
| summer, h08 | 30,122 | 1.1212 |
| summer, h18 | 32,786 | 1.2225 |

Positive everywhere, ~2.5–3× larger in summer. The headline configuration is
the most favourable of the four, so quote the range, not just summer h18.

### Two dataset limitations to state in the paper, not hide

1. **E2 risk incidence:** `π_k` is *exactly* zero for coal and nuclear. This is
   the dataset, not the theory — thermal capacity in `thermal_gens.csv` is
   constant across scenarios, so an infra-marginal unit has constant `L` and
   `Σ_w (q_w − ρ_w)L = L·0 = 0`. Forced outages would give thermal genuine tail
   exposure. The load carries 99.1% of the tail-risk premium and is the natural
   buyer of cover; renewables are mildly negative and sell it.
2. **Eq. (111) equal-share rule:** the flat pool share (−$104,273/h) swamps the
   per-agent signal, making the net column nearly identical across generators.
   A contribution-weighted `δ_k` would allocate it sensibly. Property of the
   rule, worth a remark.

3. **E3:** `270 of 280` thermal heat rates were floored to physical minima
   because the dataset's gas heat rates are not physical. Flagged in-log.

### Infrastructure fixed in this round

- OSCAR compute nodes have **no outbound network**, so the in-job `git push`
  could never work. Jobs now commit only; `push_results.sh` pushes from a login
  node. The old failure mode looked like a lost run but never was.
- `check_oscar_run.sh` added: SLURM exit code, end-of-recipe marker, per-stage
  artefact inventory, `json.load` on every output (a killed job leaves
  truncated JSON that `ls` reports as present), and push status.

---

## 2026-11-17 (rev. 2) — W7: the tail-heaviness thesis is CONFIRMED

The first version of this entry (below) led with the structural ceiling and
read as though risk-aware clearing does not scale with tail-heaviness. **That
was a wrong reading of a correct result, and the error was in the denominator.**

W7 sweeps four *independent* physical drivers of a heavy supply tail. All four
agree, with perfect rank correlation:

| driver | gap $, light → heavy | growth | % of reachable cost | Spearman |
|---|---|---|---|---|
| stress probability | 2,848 → 10,646 | **3.7×** | 7.6 → 15.5% | +1.000 |
| stress depth | 3,607 → 8,445 | **2.3×** | 9.0 → 13.8% | +1.000 |
| spatial correlation `ρ` | 3,345 → 7,011 | **2.1×** | 8.5 → 12.8% | +1.000 |
| common-mode outage | 2,824 → 11,209 | **4.0×** | 6.9 → 16.7% | +1.000 |

(S=300–500, 4 paired seeds, ε=0.3, β=0.95, CLARABEL pilot. Against the
*deterministic* policy the gap reaches **$20,739**, ~17× the light-tail case.)

So the thesis — tail-heavy supply ⇒ tail-heavy welfare ⇒ risk-aware clearing is
worth materially more — **holds, across every driver, monotonically.**

### Why the first reading was wrong

Quoted as a share of *total* tail cost the gain looks flat (~2.3–3.7%) at every
tail-heaviness. That is an artefact. Heavier tails raise unserved energy, which
W0 shows is policy-invariant, so the denominator inflates at nearly the same
rate as the numerator and the ratio cancels. The `unreach%` diagnostic makes it
explicit: it climbs 61% → 83% across the stress sweep.

**Use `pct_of_reachable_cost` or `$/MWh`. Do not quote `pct_of_total_welfare`,
and quote `pct_of_tail_cost` only alongside the unreachable share.** `gaps()`
returns all three with this reasoning in its docstring.

The ceiling (W0) is still real and still worth reporting: risk-aware clearing
reallocates adjustment cost and cannot improve adequacy. But "the denominator
grows with tail-heaviness" and "the method does not scale with tail-heaviness"
are different claims, and only the first is true.

### Recommended framing

1. Lead with W7: the gain scales 2–4× with supply tail-heaviness, confirmed on
   four independent mechanisms.
2. Report W0 immediately after, as scope rather than apology: the gain is
   concentrated in adjustment cost because adequacy is physically fixed. This
   pre-empts the obvious referee question instead of inviting it.
3. Quote `$/MWh` and `% of reachable tail cost`.
4. Keep W4 (VOLL 200–5000) as the standing rebuttal to "it is a VOLL artefact".

---

## 2026-11-17 — Tail-welfare experiments (W0-W6) and a structural ceiling on them

> **Superseded in part by the W7 entry above.** The ceiling result stands; the
> conclusion that the advantage does not grow with tail-heaviness does not —
> that was a denominator artefact, corrected above.

### Headline, stated plainly

**In the paper's formulation, risk-aware clearing cannot improve reliability.
It reallocates adjustment cost.** Unserved energy is *identical* under the
risk-aware, risk-neutral and deterministic policies — not approximately, but to
solver precision (max relative difference `4.7e-08`). Since unserved energy is
**73–82% of conditional tail cost**, this caps the achievable tail-welfare gain
before any clearing happens.

The proof is one line. Real-time capacity `ḡ(ω)` is exogenous, so

    Σ_j D_j(ω) ≤ min( Σ_i ḡ_i(ω), Σ_j d̄_j(ω) )

and because VOLL greatly exceeds every marginal cost the planner always serves
the physical maximum. How much load is shed in scenario ω is fixed by the
scenario, not by the day-ahead decision. **No scenario family changes this** —
it is a property of the formulation, so searching for a distribution that makes
the gap large is searching for something that does not exist.

This is almost certainly why referees are unconvinced: they expect a reliability
gain the model structurally cannot produce. Reporting the ceiling converts the
objection into a contribution.

### What the method *does* do, and it is statistically solid

Pilot at S=100, 5 paired seeds, ε=0.3, β=0.95, CLARABEL:

| metric | gain | 95% CI (paired) | p |
|---|---|---|---|
| % of tail cost | **+2.41%** | [+0.76, +4.07] | 0.016 |
| % of total welfare | +0.74% | [−0.09, +1.58] | 0.068 |
| $/MWh served | +5.71 | [−0.43, +11.86] | 0.061 |

100% of seeds improved. The decomposition shows precisely what is bought:

    upward adjustment   −3780      ← pre-positioning avoids expensive ramps
    load adjustment     −1893
    downward adjustment  +422      ← the premium paid to back down
    energy                  −1
    unserved                 0     ← untouchable, as above

### The denominator is doing a lot of damage

Expected social surplus is dominated by the gross consumer value `α^d · D ≈ 9e5`,
which barely moves between policies. **A gain quoted as a percentage of total
welfare is divided by that near-constant and reads ~3.3× smaller than a gain
quoted against the cost the policy can actually move.** Both are reported in
`gaps()`; the paper should lead with tail cost or $/MWh and state the
relationship, not quote the diluted figure.

This compounds the S5 finding below: the headline $/MWh figure is proportional
to `delta_scale`, so a single number is a restatement of that constant. Quote
the slope.

### Mechanism is confirmed, monotone and causal

The day-ahead position is priced *only* through the deviation terms, so the
benefit must scale with `α⁺`. It does, cleanly (W2, 3 seeds, pilot):

| `α⁺/c` | 0 | 1 | 3–4 |
|---|---|---|---|
| gain (% of tail cost) | +1.17 | +2.45 | +7.45 |

`zavala_funcs.py` hard-codes `α⁺ = α⁻ = mc/10`, which is economically backwards:
starting an uncommitted unit costs far more than backing a committed one down.
Paper Eq. (3) already defines the two separately, so **asymmetry is within the
existing theory** and is simply not being used. `tail_model.py` supports it;
passing `dgp == dgn` recovers the current model exactly.

### The one extension that breaks the ceiling

Adding a start-up constraint `G_i(ω) − g_i ≤ R_i` makes the day-ahead schedule
physically binding, and then unserved energy *does* respond to policy:

| `R_i`/nameplate | Δ CVaR unserved | gain (% tail cost) |
|---|---|---|
| none (paper's model) | **−0.00 MW** | 2.21 |
| 0.30, 0.15 | **−0.00 MW** | 0.6–1.3 |
| 0.05 / 0.02 / 0.00 | −2.5 / −7.2 / −7.5 MW | 2.3 |

This is an **extension beyond Eq. (63)–(65)** and is run and reported
separately (`--ramp-limit-frac`, the `_rampext` output). It does not obviously
dominate the asymmetry channel at these settings.

Also note the invariance is a *consequence of VOLL dominating*: at VOLL=200 the
peaker ramp premium becomes comparable to VOLL, shedding becomes a genuine
choice, and `Δ unserved = −37.6 MW`. The ceiling is a high-VOLL phenomenon.

### Scenario family

`tail_scenarios.py` replaces assumed shocks with estimable dependence: a
one-factor Gaussian copula over Beta capacity-factor marginals (spatial
correlation), a two-component regime mixture for the latent weather factor
(wide-area low-renewable events), demand loading on `−z` (the same weather that
kills wind raises load), and correlated thermal forced outages with a
common-mode block failure. Realised `corr(demand, renewables) = −0.71`,
net-load p95/p50 = 1.66. No inflated VOLL is required, and W4 sweeps VOLL over
200–5000 to demonstrate it.

### Files

- `tail_scenarios.py` — dependence-structured instance generator
- `tail_model.py` — vectorised model with asymmetric `α±`; common recourse
  evaluator; `assert_matches_reference` checks it against `zavala_cvar`
- `tail_experiments.py` — W0–W6
- `run_tail_experiments.py` — CLI, records git commit and solver provenance
- `plot_tail_figures.py` — six conference-ready PDF/PNG figure pairs
- `slurm_tail.sh` — OSCAR batch job; pushes results back

### A correctness fix folded in

`zavala_rt_energy_only` scored the deterministic baseline with the objective
`(mc + δ)|G − g_da|`, which omits the energy cost and the load's value entirely
— it minimises schedule *change*, not cost. Every policy is now scored through
one `evaluate_da`, making the three-way comparison apples-to-apples.

### Status / not yet done

- Numbers above are a **local pilot** (CLARABEL, S=80–100, 3–6 seeds). The
  cluster run at S=500 with 30 paired seeds under Gurobi is pending; CIs will
  tighten materially.
- `--check-reference` has not yet run (needs jax + Gurobi, cluster only).
- Consider whether the paper's strongest claim is tail welfare at all: the
  pricing and decentralisation results (T1–T11, D1–D5) all verify with
  residuals at `1e-13`, which is a far stronger position than a 2% welfare gap.

---

## 2026-11-04 — Decentralisation & insurance experiments (D1-D5), and a sign error in Eq. (111)-(112)

### Headline

Implemented Sections 8.3-8.7 (the agents' problems, the insurance market, the
transfer constructions) and built five experiments. **All checks pass** at
S=30 locally under CLARABEL. In the process, found and numerically confirmed a
**sign error in Eq. (111)-(112)**.

### The sign error

Section 8.4 states the agent's per-scenario payoff is `u_k + r_k`, so the loss
entering the CVaR constraint Eq. (97)/(101) is

    -u_k - r_k = L_k - r_k.

Flattening *that* requires

    r_k(w) = +L_k(w) - ell(w)/N - delta_k        [corrected]

giving insured loss `L_k - r_k = ell(w)/N + delta_k`. But Eq. (111)-(112) as
printed give

    r_k(w) = -L_k(w) + ell(w)/N + delta_k        [as printed]

which flattens `L_k + r_k` instead. Against the actual CVaR constraint it does
not hedge the agent at all — it roughly **doubles** its exposure.

**Both signs satisfy market clearing Eq. (95)**, which is why the error survives
that check: `sum_k r_k = -ell + ell = 0` either way.

**Corroboration inside the paper.** Eq. (122) in Theorem 9 already uses the
corrected sign, `r_k = L_k - t_k - w_k(ell - t*)`. With `w_k = 1/N` and
`sum_k t_k = t*` it reduces exactly to the corrected equal-share rule. So
Eq. (111)-(112) are the inconsistent pair, not Eq. (122).

**Numerical evidence** (s_real10_mix, S=30, seed 2025, beta=0.95), maximum
relative (UM) gap over all 11 agents — 0 means the planner's allocation is
agent-optimal:

| eps | corrected sign | Eq. (111)-(112) as printed |
|-----|----------------|----------------------------|
| 0.1 | 7.6e-11        | 9.0e-01                    |
| 0.3 | 4.9e-10        | 1.6e+00                    |
| 0.5 | 0.0            | 9.9e-01                    |
| 1.0 | 0.0            | --                         |

At `eps = 0` the gap is exactly 0 under both signs, because insurance is
redundant there (Remark 10). That is also the control establishing that the
agent model itself is correct: it reproduces risk-neutral (UM) exactly.

### Experiment design

| ID | Question | Verifies |
|----|----------|----------|
| D1 | Does the decentralised market reproduce the planner's allocation? | Thm 6, 7, 8; Eq. (94), (109) |
| D2 | What breaks without the insurance market? | Section 8.2.2, Eq. (94) |
| D3 | What do the equilibrium transfers look like? | Remarks 7, 8, 9; Eq. (113) |
| D4 | Do heterogeneous confidence levels still clear? | Thm 9, Eq. (121)-(123) |
| D5 | How do transfers scale with the risk weight? | Remark 10 |

**Testing methodology.** Definition 3's (UM) makes insurance a *choice*
variable, so each agent optimises over physical position **and** `r` at prices
`q`; the test is whether its optimal objective equals the objective at
(planner allocation, constructed transfer). Real-time optima are degenerate —
several allocations tie — so allocation distance is reported as a diagnostic
only and the objective value is the criterion.

### Results at S=30 (local smoke run; cluster run at S=500 pending)

- **D1.** (UM) gap 7.6e-11. Eq. (109) `gamma_k = gamma` holds to 5.2e-10:
  every agent's CVaR multiplier equals the planner's. Market clears to 2.9e-10.
- **D2.** Without insurance Eq. (94) **fails**, as Section 8.2.2 predicts:
  `max_k max_w |gamma_k - gamma| = 8.0e-02`. Per-agent tail-scenario counts
  range from 2 to 7 against the planner's 2 — agents disagree about where the
  tail *is*, which is exactly the obstruction.
- **D3.** Eq. (113) holds exactly: slope 1 to 2.2e-16, affine residual 1.5e-11.
  Economically, **the unreliable units buy cover and the reliable units sell
  it**: unreliable generators pay 13,198 more in premium than they expect back,
  reliable generators collect +12,826. Tail premium `gamma . r` is positive for
  every unreliable unit and negative for every reliable one.
- **D4.** Theorem 9 verified — the construction the manuscript marks as not yet
  proof-checked. With beta drawn from {0.90, 0.95, 0.99} and beta_min = 0.90,
  the 10 agents in `S_>` hold a constant insured loss across the tail to
  2.3e-13 (Eq. 121); the equal-share rule gives them a spread of 4.0e+03, i.e.
  it violates Eq. (121) exactly as Theorem 9 predicts.
- **D5.** At eps = 0 the tail premium vanishes (1.7e-09 relative) and insurance
  is redundant. As eps rises the day-ahead price climbs steeply
  (206.9 -> 295.1 -> 473.9 -> 638.0 -> 1000.0) while `E_rho[Pi]` stays near 200.

### Modelling observations worth a sentence in the paper

1. **The equal-share rule pools gross welfare, not just cost.** The load holds
   most of the surplus, so its `L` is large and negative and it becomes the
   dominant risk-bearer. The free constants `delta_k` shift the level but not
   the shape. A cost-only or capacity-weighted pooling rule would distribute
   differently; Remark 9 already notes `delta_k` is free, but the *shape* is
   not.
2. **`p0` tracks `E_rhotilde[Pi]`, not `E_rho[Pi]`.** At eps=0.1,
   `p0 = 295.09` against `E_rhotilde[Pi] = 294.84` but `E_rho[Pi] = 205.38`.
   The risk-neutral consistency property holds only under the tilted measure.
   This is worth stating explicitly, since `price_distortion()` still measures
   against the raw measure (flagged in the previous entry).
3. **The repo's planner leaves day-ahead quantities free**, whereas Eq. (12)/
   (18)/(65) impose `x, y >= 0`. The solutions seen so far satisfy it anyway;
   `build_planner` now warns if a run ever violates it.

### Files

- `zavala_decentralized.py` — agent problems, transfer constructions, utilities.
  `transfers_equal_share(..., as_printed=True)` reproduces Eq. (111)-(112)
  verbatim for the comparison above.
- `run_decentralized_experiments.py` — D1-D5 driver; writes JSON + log.
- `slurm_decentralized.sh` — cluster script: Gurobi invariant check, then D1-D5
  at eps=0.1, then D1/D3 at eps in {0.3, 0.5, 1.0}, all at S=500.

### Not yet done

- Cluster run at S=500 with Gurobi (local runs used CLARABEL at S=30; the local
  Gurobi licence expired 2026-03-21).
- Remark 11's jointly-heterogeneous `(eps_k, beta_k)` case is deferred in the
  paper and not implemented.

---

## 2026-11-04 — Fix CVaR real-time price identification (paper Eq. 67 / 107)

### Summary

`zavala_cvar` computed real-time prices with the wrong probability weight in the
denominator. The optimisation model itself was correct; only the dual readout was
wrong. Fixed, and verified numerically against the paper's KKT identities.

### The defect

Paper Eq. (67) defines the CVaR-tilted probability weight

    rho_tilde_w = (1 - eps) * rho_w + gamma_w

and Eq. (107) identifies the prices as

    p_0 = lambda                      (day-ahead)
    p_w = nu_w / rho_tilde_w          (real-time, scenario w)
    q_w = rho_tilde_w                 (Arrow-Debreu / insurance)

where `eps` is the risk weight (`lambda_cvar` in code), `nu_w` is the multiplier on
the real-time balance constraint, and `gamma_w` the multiplier on the CVaR shortfall
constraint Eq. (62).

`zavala_funcs.py` divided by `(rho_w + gamma_w)` — omitting the `(1 - eps)` factor:

```python
# before
Pi = jnp.array([real_time_balance[p].dual_value / (probs[p] + cvar_link[p].dual_value) ...])
# after
rho_tilde = (1.0 - eps) * probs_np + gamma
Pi = jnp.array(nu / rho_tilde)
```

**Origin.** That denominator is correct for the *un-normalised* objective
`E[loss] + eps*CVaR`. Commit `251f501` ("regularized cvar objective") changed the
objective to the normalised form of paper Eq. (58),
`(1 - eps)*E[loss] + eps*CVaR`, but left the price formula untouched.

### Magnitude of the error

For any scenario with `gamma_w = 0` (all non-tail scenarios) the old price was too
low by exactly a factor of `(1 - eps)`, i.e. a relative error of exactly `eps`.
Measured on a 40-scenario `s_real10_mix` instance (seed 2025, beta=0.95):

| eps | bias in E[Pi] (old - new), $/MWh | max per-scenario relative error |
|-----|----------------------------------|---------------------------------|
| 0.1 | -6.07                            | 10.0 %                          |
| 0.3 | -14.78                           | 30.0 %                          |
| 0.5 | -23.91                           | 50.0 %                          |
| 1.0 | -85.85                           | 100.0 %                         |

The old formula **systematically under-reported CVaR real-time prices.**

### What this does and does not invalidate

- **Not affected — all welfare results.** `compute_social_surplus` reads primal
  allocations only. The LP solved was always the correct one, so dispatch,
  expected welfare and tail welfare are sound. That covers Table 1, Table 3,
  Table 4, Table 6, and Figures 1-3.
- **Affected — every reported price under CVaR clearing.** In particular Table 5
  (`E[RT price]` = 989.96 for CVaR) and the $100/MWh day-ahead premium discussion
  in Appendix C, which is stated relative to that number. These need re-running.

### Verification

New script `verify_cvar_pricing.py` checks the planner's KKT identities directly.
Run locally with CLARABEL (Gurobi licence on this machine expired 2026-03-21);
reproduce on the cluster with `--solver GUROBI`. Full output in
`logs/verify_cvar_pricing_local.log`.

Checks, all passing on `s_real10_mix`, S=40, G=10, beta=0.95, eps in {0.1,0.3,0.5,1.0}:

| Check | Identity | Result |
|-------|----------|--------|
| Eq. (69) | `sum_w gamma_w = eps` | max abs dev 2.1e-09 |
| Eq. (67) | `sum_w rho_tilde_w = 1` | max abs dev 2.1e-09 |
| Eq. (70) | `0 <= gamma_w <= eps*rho_w/(1-beta)` | satisfied |
| Eq. (107) | `rho_tilde_w > 0` so `p_w` well defined | satisfied |
| Remark 10 | `eps = 0` reproduces risk-neutral `zavala` | `|dpi| = 3.7e-07`, `max|dPi| = 3.4e-05`, `max|gamma| = 2.3e-09` |

This also **confirms the CVXPY sign convention**: `cvar_link[p].dual_value` is
exactly the paper's `gamma_w`, with no rescaling needed.

Note: only 2 of 40 scenarios carry tail mass (`gamma_w > 0`) at beta=0.95, as
expected — `(1-beta)*S = 2`.

### Two inconsistencies in the paper (not code)

The same stale normalisation appears in the manuscript, in two places that
contradict Eq. (58), (67) and (107):

1. **p. 19:** "Moreover, `sum_w rho_tilde_w = 1 + eps`, so the normalised weights
   `rho_tilde_w/(1+eps)` define a probability measure." Under Eq. (58) the correct
   sum is `(1-eps) + eps = 1`; no normalisation is needed. Stated correctly on
   p. 21, p. 25 and p. 27 (Lemma 5 proof).
2. **p. 26**, Walras' law proof: "since `p_0 = lambda/(1+eps)`" contradicts
   Eq. (107)'s `p_0 := lambda`.

Both read as leftovers from the pre-normalisation objective — the same root cause
as the code defect.

### Open modelling question

`price_distortion()` computes `|pi - sum_w rho_w * Pi_w|` using the **raw**
probabilities. Under the risk-aware model the natural analogue is the tilted
measure, `|pi - sum_w rho_tilde_w * Pi_w|`. The paper does not state which it
intends (Section 9 reports welfare, not distortion). Flagged; not changed.

### Other changes in this commit

- `generate_instance(...)` now takes `input_scenario` as a parameter, defaulting
  to **`s_real10_mix`** — the setup in Section 9.1 / Table 2. It was hard-coded to
  `s_htoy_mix_v2` (the 2-generator toy), so the repo as committed did not reproduce
  the paper's synthetic experiment.
- `zavala(...)` and `zavala_cvar(...)` take an optional `solver` (default unchanged:
  Gurobi) and `verbose` flag, so invariants can be checked without a Gurobi licence.
  The model is untouched.
- Both now raise on non-optimal solver status instead of silently returning `None`
  duals.

### Still unimplemented

The insurance market (Section 8.3-8.5): nothing computes `q_w`, `r^c` or `r^g`.
`q_w = rho_tilde_w` is now available for free from the fix above. Section 9 makes
no insurance claims, so this is not a gap in the current results.

---

## 2026-11-05 — Pre-flight verification of the OSCAR decentralisation run

Audit of `slurm_decentralized.sh` before submission. Three defects found and fixed;
no change to any experiment's logic or results.

### 1. Result logs were silently gitignored (blocking)

`.gitignore` inherited a blanket `*.log` rule from the GitHub Python template.
Both `logs/decentralized_<tag>.log` (the full paper-ready narrative log) and
`logs/verify_cvar_pricing_gurobi.log` matched it, so the documented
round-trip (`git add logs outputs && git push`) would have pushed the JSON
results but **silently dropped every human-readable log**. Verified with
`git check-ignore`. Fixed with explicit negations; re-verified all four
artifact paths now stage.

### 2. GPU partition requested for a pure-CPU workload

All three SLURM scripts requested `--partition=gpu --gres=gpu:nvidia_rtx_a5000:1`.
The workload is cvxpy + Gurobi linear programming, which is CPU-only; `jax` is
used solely for PRNG-keyed instance generation and `requirements.txt` pins the
CPU `jaxlib` wheel. Confirmed by grep: no `cuda`/`device` reference anywhere in
`zavala_funcs.py`, `zavala_real_data.py`, `run_decentralized_experiments.py`,
`zavala_decentralized.py`. Switched to `--partition=batch` (OSCAR's default
general-purpose CPU partition). Queue time only — no effect on results.

### 3. `logs/` not created by the job script

The script `mkdir -p outputs/logs outputs/results` but then piped the Gurobi
sanity check through `tee logs/...`. Under `set -euo pipefail` a missing `logs/`
would abort the job at step one. Added `logs` to the `mkdir`, and added
`outputs/logs/.gitkeep` so the SBATCH `--output` directory exists at submission
time (SLURM resolves `--output` *before* the script body runs).

### 4. Added a Gurobi licence preflight

A trivial one-variable LP now runs before any experiment, so an expired or
missing licence fails in seconds with an actionable message instead of a cryptic
solver error partway through D3.

### Runtime and scale validation

| S | configuration | wall time | result |
|---|---|---|---|
| 30 | D1, CLARABEL | 4.9 s | pass |
| 80 | D1, CLARABEL | 5.4 s | pass |
| 200 | **D1–D5, CLARABEL** | **81.8 s** | **ALL CHECKS PASSED** |
| 40 | D1–D5 + eps sweep {0.3,0.5,1.0} | ~60 s | ALL CHECKS PASSED |

The S=200 run is the important one: all identities that were previously only
confirmed at the S=30 smoke scale continue to hold an order of magnitude
larger. The `--time=04:00:00` limit is generous by a wide margin.

Every command in `slurm_decentralized.sh` was dry-run locally with its exact
flag set (CLARABEL substituted for GUROBI, reduced S). All exited 0.

**Caveat carried forward:** every number to date is CLARABEL, not Gurobi,
because the local Gurobi licence expired 2026-03-21. The pass/fail identities
are structural and solver-independent, but the economic magnitudes in D3 and D5
should be quoted from the OSCAR Gurobi run, not from these.

---

## 2026-11-05 — Theory audit + IM3 GO-WEST economic experiments (E1–E6)

Two deliverables: a full adjudication of the manuscript's theory
(`docs/THEORY_ERRATA.md`, 10 items + 5 empirical ones), and a new experiment
suite on real Western Interconnection data that tests what the theory *buys*
rather than only that its equations hold.

### New files

| File | Purpose |
|---|---|
| `verify_theory.py` | Adjudicates T1–T8. Each test can only pass if the corrected statement is right. |
| `im3_data.py` | IM3 GO-WEST instance builder: real marginal costs, conditional scenario sampling, fleet sizing. |
| `im3_experiments.py` | E1–E6 plus a recourse evaluator and CVaR/ρ_RA helpers. |
| `run_im3_experiments.py` | Driver. |
| `slurm_im3.sh` | OSCAR job: theory → D1–D5 → E1–E6 → seasonal robustness. |
| `docs/THEORY_ERRATA.md` | The errata list with the numbers behind each verdict. |

### Theory findings

Confirmed errors **T1** (`Σρ̃ = 1`, not `1+ε`; contradicted three times by the
paper itself), **T2** (`p₀ = λ`, not `λ/(1+ε)`), **T3**/**T4** (sign of
Eq. 111–113). Two *gaps* rather than errors: **T9**, the agent's insured program
is exactly flat in `r` at `q = ρ̃` so the argmax is a flat set and the LP is
numerically unbounded; **T10**, the agent's optimum exists only because `α^± > 0`,
which the paper never states.

Two retractions/corrections of our own earlier claims:

* **Theorem 9 "verified" was overstated.** D4 checked only Eq. (95) and Eq. (121),
  which hold for *any* `{t_k}` and so cannot discriminate. The (UM) test now lives
  in `verify_theory.py::test_T5` and passes (1.3e−05, solver tolerance).
* **The suspected `t*` circularity in Eq. (122) is not an error.** Both readings
  give identical objectives (1.1e−10) because a constant shift of `r` is
  objective-neutral — which holds *precisely because* `Σρ̃ = 1` (T1).

**Remark 11 verified** and promotable from a deferred remark: with ε,β
anti-correlated so `β_P` falls outside the agents' own pool, `ε_min = 0.2`,
`φ_min = 4.5`, `β_P = 0.957447` against `{0.90, 0.99}`; the resulting `q` is
feasible for every agent (max violation 2.8e−09).

### Data-layer corrections (all affect the published tables)

* **Marginal costs were randomised** (`Uniform[35,55]`) despite real `Heat_Rate`
  and `Fuel` columns. Now `mc = Heat_Rate × fuel_price + VOM`, giving
  nuclear $13.70 < NG $21.22 < coal $23.27. **Caveat: 270 of 280 gas heat rates
  are non-physical (mean 4.02 MMBtu/MWh) and are floored.**
* **Scenarios were consecutive hours**, conflating the diurnal cycle (solar
  0→13.8→0 GW within a day) with forecast uncertainty. Now conditioned on a
  delivery hour and sampled across days.
* **The published instances sit in permanent scarcity**: full WECC load (mean
  82 GW) against a few-bus fleet, so price pins to VOLL. Load *level* is now
  scaled to a 15% reserve margin; *shape* untouched. Market prices at
  **$21.92/MWh** with a sensible merit order.
* **`delta_scale` is now a parameter of `zavala_cvar`** (default 0.1, reproducing
  all earlier results exactly). It was hard-coded as `mc/10` with the comment
  "just a heuristic".

### Results (local smoke, CLARABEL, S=50–80 — magnitudes pending OSCAR/Gurobi)

**E1 — value of the missing insurance market.** $27,839/h = **1.03 $/MWh**
= $244M/yr, 0.11% of risk-adjusted welfare. Decomposes into E[ℓ] +55,765 (worse)
against CVaR −222,916 (better). Without insurance the agents' tail sets range
over 3–60 scenarios against the planner's 3; `max|γ_k − γ| = 0.10` where Eq. (94)
requires 0.

**E2 — incidence.** The **load is the residual risk bearer** (π = +1.55M/h):
its loss spikes exactly where the market prices heavily, so it is the natural
buyer of cover. Renewables are mildly negative (they sell cover). π is *exactly*
zero for coal and nuclear — a property of the dataset, not the theory: thermal
capacity is scenario-invariant, so an infra-marginal unit has constant `L` and
`Σ(q−ρ)L = L·0 = 0`. Forced outages would change this.

**E3 — decarbonisation.** Scaling renewables *on top of* an unchanged thermal
fleet leaves the insurance value flat (1.31 → 1.25 $/MWh), because firm capacity
still covers peak and the tail never moves. Redesigned as substitution at fixed
total capacity (thermal retires as renewables enter): **still flat**
(1.3075 → 1.1988 $/MWh from 15% to 75% renewable share). Reported as a negative
result.

**E4 — out-of-sample.** Clearing on one scenario set and evaluating the day-ahead
commitment on a disjoint set from the same conditional pool:
tail loss **−211,356 ± 14,134, improved in 5/5 replications**; expected loss
+38,005 (worse); planner objective −36,803, 5/5. Risk-aware clearing genuinely
reduces held-out tail loss rather than fitting the in-sample tail.

**E5 — paired statistics.** Repairs Tables 1 and 3.

| method | mean | unpaired ± | paired diff | paired ± | wins |
|---|---|---|---|---|---|
| risk-neutral | −19,646,193 | 416,371 | — | — | — |
| CVaR ε=0.1 | −19,707,113 | 425,760 | −60,920 | 29,956 | 5/5 |
| CVaR ε=0.3 | −19,884,763 | 433,269 | −238,570 | 22,833 | 5/5 |
| CVaR ε=1.0 | −20,324,748 | 379,285 | −678,555 | 44,455 | 5/5 |

The unpaired interval is **14× wider** than the paired one. At ε=0.1 the effect is
invisible unpaired and clearly significant paired — exactly Table 1's situation.

**E6 — what the headline number depends on.** The insurance value is linear in
the deviation bid price to **0.003%**:

| `delta_scale` | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| $/MWh | 0.1931 | 0.4827 | 0.9655 | 1.9309 | 3.8621 |

**value = 9.655 × `delta_scale` $/MWh = 0.9655% of the load's deviation bid price.**
The paper should quote this slope, not a point value.

### Honest caveats

* All magnitudes above are CLARABEL at S=50–80 (local Gurobi licence expired
  2026-03-21). Pass/fail identities are structural; **economic magnitudes will
  move** on the S=500 Gurobi run.
* E3 is a negative result and is reported as one.
* The single inelastic load at VOLL means essentially all tail risk lands on it;
  a richer demand side would redistribute the incidence in E2.
