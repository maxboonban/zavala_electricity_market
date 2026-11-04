# Experiment & Change Log

Running log of code changes and experimental results, kept in enough detail to
cite directly in the paper. Newest entries first.

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
