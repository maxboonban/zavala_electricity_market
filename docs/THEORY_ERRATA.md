# Theory errata — `Risk-Aware Market Clearing in Electricity Markets`

Every item below was adjudicated numerically by `verify_theory.py`, which is
written so that a test can only pass if the corrected statement is right and the
printed one wrong. Run it with:

```bash
python verify_theory.py --solver GUROBI --scenarios 200 --eps 0.3
```

Status at 2026-11-05: **ALL THEORY CHECKS PASSED** (CLARABEL, S=50, ε=0.3).

| # | Location | Severity | Status |
|---|---|---|---|
| T1 | p. 19, after Eq. (70) | **Error** | confirmed, fix below |
| T2 | p. 26, Walras' law proof | **Error** | confirmed, fix below |
| T3 | Eq. (111)–(112) | **Error** | confirmed, fix below |
| T4 | Eq. (113) | **Error** | follows from T3 |
| T5 | Eq. (122), `t* := Σ t_k` | *investigated* | **not an error** |
| T6 | Eq. (68) and (86) | Typo | confirmed |
| T7 | Lemma 5 proof | Edge case | needs an ε>0 caveat |
| T8 | Remark 11 | *deferred in paper* | **verified correct** |
| T9 | Eq. (96)–(103) | **Gap** | agent's program is ill-posed as written |
| T10 | Eq. (63)–(65) | **Gap** | agent's optimum need not exist |

---

## T1 — `Σ_ω ρ̃_ω = 1`, not `1 + ε`

**Printed (p. 19):** "Moreover, `Σ_ω ρ̃_ω = 1 + ε`, so the normalised weights
`ρ̃_ω/(1+ε)` define a probability measure…"

**Correct:** `Σ_ω ρ̃_ω = Σ_ω[(1−ε)ρ_ω + γ_ω] = (1−ε)·1 + ε = 1`, using Eq. (69).
`ρ̃` is *already* a probability measure; no normalisation is needed or correct.

The manuscript contradicts itself here three times over — it computes the sum
correctly everywhere else:

* p. 21: "`Σ_ω ρ̃^c_{i,ω} = (1−ε) + ε = 1`, so the agent's tilted weights also form a probability measure"
* p. 25, Theorem 6 proof: "Since `Σ_ω ρ̃*_ω = (1−ε) + ε = 1`…"
* Remark 8: "Because `Σ_ω q*_ω = 1`, the insurance prices form a probability measure"

Since `ρ̃` (planner) and `ρ̃^c` (agent) have *identical* definitions, p. 19 is the
outlier. **Measured: `Σρ̃ = 1.000000007` at ε = 0.3.**

**Fix:** delete the normaliser. The sentence should read "Moreover `Σ_ω ρ̃_ω = 1`,
so `ρ̃` is itself a probability measure that places additional mass on tail
scenarios (those with `γ_ω > 0`)."

## T2 — `p₀* = λ*`, not `λ*/(1+ε)`

**Printed (p. 26, proof of Walras' law):** "since `p₀* = λ*/(1+ε)`".

**Correct:** `p₀* = λ*`. This is what Eq. (107) *defines*, and what the Theorem 6
(UM) proof *derives* ("these coincide if and only if `λ* = p₀*`"). The `(1+ε)`
is residue from the same mistake as T1.

**Consequence:** this one reached the code. `zavala_cvar` previously divided the
real-time price by `ρ_ω + γ_ω` instead of `ρ̃_ω = (1−ε)ρ_ω + γ_ω`, under-reporting
prices by exactly a factor `(1−ε)` off-tail. Fixed in commit `e6f4609`;
**Table 5's price rows and the $100/MWh premium claim in Appendix C were computed
with the buggy formula and must be regenerated.**

## T3 — sign error in Eq. (111)–(112)

**Printed:** `r^c*_{iω} := −L^c_{iω} + ℓ*(ω)/(n+m) + δ^c_i`

**Correct:** `r^c*_{iω} := +L^c_{iω} − ℓ*(ω)/(n+m) − δ^c_i`

Eq. (97) writes the CVaR shortfall constraint as `ζ ≥ −u_i − r_iω − t`, i.e. the
insured loss is `L − r`; p. 23 confirms the payoff is `u + r`. But Eq. (113)
flattens `L + r`. Only one sign can be right, and it is not the printed one: the
printed transfer roughly *doubles* the agent's exposure instead of hedging it.

**Both signs satisfy market clearing Eq. (95)** (`Σr = −ℓ + ℓ = 0` either way),
which is exactly why the error survived the check the paper performs.

Max relative (UM) gap across all agents — 0 means the planner's allocation is
agent-optimal at the posted prices:

| ε | as printed | corrected |
|---|---|---|
| 0.1 | 9.0e−01 | 7.6e−11 |
| 0.3 | 1.98e+00 | 2.4e−05 |
| 0.5 | 9.9e−01 | 0.0 |

The residuals on the corrected side are solver tolerance; the two signs separate
by **8.2 × 10⁴**. The control that makes this conclusive: at **ε = 0 both signs
give exactly 0**, because insurance is redundant there (Remark 10) — so the
agent model is right and the failure is specific to ε > 0.

**Eq. (122) in Theorem 9 already uses the corrected sign** (`r_k = L_k − t_k − …`),
and reduces to the corrected rule when `w_k = 1/N`. Two independent confirmations.

## T4 — Eq. (113) restatement

**Printed:** `L^c_{iω} + r^c*_{iω} = ℓ*(ω)/(n+m) + δ^c_i`

**Correct:** `L^c_{iω} − r^c*_{iω} = ℓ*(ω)/(n+m) + δ^c_i`

Direct consequence of T3, and consistent with Eq. (124) in Theorem 9, which
already writes the insured loss as `L_kω − r*_kω`.

## T5 — `t* := Σ_k t^c*_k` is **not** circular *(suspected error, cleared)*

Eq. (122) sets `r_k = L_k − t_k − w_k(ℓ*(ω) − t*)` with `t* := Σ_k t_k`, while
`t_k` also names each agent's own VaR variable. We expected self-consistency to
force `t* = VaR_{β_min}(ℓ*)` — the *planner's* VaR — and tested both readings.

**They give identical objectives** (difference 1.1e−10), because a constant shift
of `r_k` is exactly objective-neutral:

```
d/dc [ (1−ε)E[u+r+c] − q·(r+c) − ε·CVaR(L−r−c) ] = (1−ε) − Σ_ω q_ω + ε = 1 − Σ_ω q_ω = 0
```

which holds **precisely because `Σ ρ̃ = 1`** (T1). So the `{t_k}` are free
constants, `t* := Σ_k t_k` is a harmless normalisation, and Theorem 9's
construction is sound as printed. *Had the paper's `1 + ε` been right, this
invariance would fail* — a pleasing internal cross-check.

**Note on our own earlier claim:** the first experiment round reported "Theorem 9
verified". That was overstated. D4 checked only Eq. (95) and Eq. (121), both of
which hold for *any* `{t_k}` and so cannot discriminate. The (UM) test that
actually verifies Theorem 9 is in `verify_theory.py::test_T5`; it passes
(max relative gap 1.3e−05, solver tolerance).

## T6 — Lagrangian typo in Eq. (68) and Eq. (86)

Printed: `… + Σ_ω γ_ω − ε t + Σ_ω(γ_ω + η_ω − ερ_ω/(1−β)) u_ω`.
The first term needs the multiplier: it should be `t(Σ_ω γ_ω − ε)`, which is what
makes `∂_t L = 0 ⟹ Σγ_ω = ε` (Eq. 69). Same typo in the agent Lagrangian (86).

## T7 — Lemma 5 divides by ε

The proof defines `Q*(ω) := γ†_ω/ε`, undefined at ε = 0. Harmless but needs a
caveat: at ε = 0, `ρ_RA = E` and `Q̂* = P` trivially, which is Remark 10.

## T8 — Remark 11 is correct *(the paper defers it; we verified it)*

Remark 11 claims that with `φ_k := ε_k β_k/(1−β_k)`, heterogeneous `(ε_k, β_k)`
reduce to a planner at `ε_P = ε_min`, `β_P = φ_min/(ε_min + φ_min)`. The paper
defers this "to the extended version". It is right, both algebraically and
numerically.

Algebra: `γ_k ≥ 0 ⟹ q ≥ (1−ε_k)ρ`, binding at `ε_min`; and
`γ_k ≤ ε_kρ/(1−β_k) ⟹ q ≤ ρ[(1−ε_k) + ε_k/(1−β_k)] = (1+φ_k)ρ`, binding at `φ_min`.
Setting `ε_P = ε_min` matches the lower bound, and `φ_P = φ_min` solves to
`β_P = φ_min/(ε_min + φ_min)`.

Numerically, with ε and β *anti-correlated* so that `β_P` falls strictly outside
the agents' own `{β_k}` (a draw the test enforces, so it cannot pass by
coincidence): `ε_min = 0.2`, `φ_min = 4.5`, **`β_P = 0.957447`** against a pool
of `{0.90, 0.99}`. The `q` produced by that planner satisfies **every** agent's
bounds (max violation 2.8e−09) and sums to 1.

**This is promotable from a deferred remark to a stated result.**

## T9 — the agent's insured program is ill-posed at equilibrium prices

At `q = ρ̃` the agent's objective is **exactly flat in `r`**: Eq. (108) is an
identity, not a condition that pins `r`. The agent's LP therefore has an
unbounded recession ray, and any numerical noise in `γ` makes it strictly
improving — solvers return `unbounded`.

The paper half-acknowledges this ("for any choice of insurance transfers
consistent with complementary slackness"), but it has a real consequence:
**(UM) cannot be stated as "the agent's problem has the planner's allocation as
its unique argmax"**, because the argmax is a large flat set. It must be stated
as "the constructed allocation attains the supremum". Any numerical verification
needs a box on `r` (we use 50× the largest loss and assert it never binds).

## T10 — the agent's optimum need not exist without strictly positive `α^±`

Eq. (65) bounds the day-ahead quantity only by `x_i ≥ 0`; Eq. (63) bounds the
*realised* quantity `x_i + x⁺ − x⁻`. So `x_i → ∞` with `x⁻ → ∞` is feasible, and
its objective slope is `p₀ − E_ρ̃[p_ω] − α`. Since equilibrium forces
`p₀ = E_ρ̃[p_ω]` (we measure 295.09 vs 294.84), the slope is `−α`: the program is
bounded **only because `α > 0`**. With `α = 0` the agent LP is unbounded and
(UM) is vacuous.

Theorem 6's "the KKT conditions are necessary and sufficient" presumes a bounded
optimum. The paper should either state `α^± > 0` as a standing assumption or
impose an explicit upper bound on day-ahead quantities.

*(The repo's planner also leaves day-ahead variables free where Eq. (12)/(18)/(65)
impose `x, y ≥ 0`. Solutions have satisfied it so far; `build_planner` warns.)*

---

## Not a theory error, but fatal to the empirical section

### S1 — Tables 1 and 3 report differences smaller than their own error bars

Table 1, 5 wind: Stochastic `2.5836 ± 0.0064` vs CVaR `2.5874 ± 0.0064`. The
difference is **0.0038 against a ±0.0064 error bar**. As printed, the headline
claim ("the CVaR-augmented formulation achieves the highest tail welfare") is not
statistically supported. Same in Table 3 (`2.58362 ± 0.00643` vs `2.58487 ± 0.00611`).

The comparison is **paired** — same instance, same scenarios, same draw — so the
paired interval is the correct summary and is far tighter. Reporting unpaired
marginal error bars for a paired design throws away most of the power. `E5` in
`run_im3_experiments.py` reports both side by side.

The *pattern* in Table 1 is real and is the interesting result: the CVaR − Stochastic
gap grows monotonically with wind penetration (0.0038 → 0.0087 → 0.0094 → 0.0102).
That is worth stating — with paired intervals.

### S2 — the published instances sit in permanent deep scarcity

`zavala_real_data.py` pairs the **full WECC system load** (mean 82 GW, peak
141 GW) with a fleet of 3 solar + N wind + 4 thermal buses (≈ tens of GW). Load
exceeds capacity in essentially every scenario, so the price pins to VOLL
throughout and the clearing rule has little left to decide. The reported welfare
(≈ 2.6 × 10⁷ against a served-load value of ≈ 1.15 × 10⁸) is consistent with only
about a quarter of load being served.

`im3_data.py` rescales the load *level* to the selected fleet at a 15% planning
reserve margin, leaving the load *shape* — where the uncertainty lives — untouched.
The resulting market prices at **$21.92/MWh** with a sensible merit order
(nuclear $13.70 < NG $21.22 < coal $23.27) rather than at VOLL.

### S3 — marginal costs were randomised despite real heat rates being present

`zavala_real_data.py` draws thermal marginal costs from `Uniform[35, 55]` while
`thermal_gens.csv` carries a per-unit `Heat_Rate` and `Fuel`. `im3_data.py`
builds `mc = Heat_Rate × fuel_price + VOM` instead.

**Caveat we must state in the paper:** the dataset's gas heat rates are not
physical (mean 4.02, min 0.62 MMBtu/MWh; a real CCGT is ≈ 6.4). We floor each
unit at a technology-specific physical minimum; **270 of 280 units are floored**.

### S4 — scenarios were consecutive hours

Taking 500 consecutive hours as 500 scenarios conflates the diurnal cycle with
forecast uncertainty: system solar runs 0 → 13.8 → 0 GW within one day, so such
an instance asks "what time is it?" rather than "what will the wind do at 18:00
tomorrow?". `im3_data.py` conditions on a delivery hour and samples across days.

### S5 — the headline economic magnitude is proportional to an undocumented constant

The deviation bid prices `α^±` of Eq. (3) were hard-coded in `zavala_funcs.py` as
`mc/10` and `VOLL/10`, with the comment *"just a heuristic"*. The value of
risk-aware clearing is **almost exactly linear** in them:

| `delta_scale` | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| insurance value ($/MWh) | 0.2610 | 0.6525 | 1.3050 | 2.6100 | 5.2200 |

The slope is constant to within rounding: **value = 13.05 × `delta_scale` $/MWh**.
Any single headline $/MWh figure is therefore a restatement of that constant.
`delta_scale` is now a parameter of `zavala_cvar` (default 0.1, reproducing all
earlier results exactly) and `E6` sweeps it. **The paper should quote the slope,
not a point value.**
