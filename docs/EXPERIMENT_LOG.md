# Experiment & Change Log

Running log of code changes and experimental results, kept in enough detail to
cite directly in the paper. Newest entries first.

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
