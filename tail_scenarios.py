"""Tail-heavy market instances with empirically motivated dependence structure.

Why this file exists
--------------------
`s_real10_mix` in `zavala_funcs.py` applies a common multiplicative shock to the
unreliable units and raises demand in shocked states. That produces a tail, but
it produces it by assumption. A reviewer is entitled to ask why *that* shock
family and not another, and the honest answer is that it was chosen to have a
tail. This module replaces the assumption with a dependence structure that is
estimated routinely in power-systems practice:

  1. **Spatial correlation via a Gaussian copula.** Wind output at sites in one
     region is strongly correlated; so is solar. The standard representation is
     a Gaussian copula over per-site Beta marginals. We use one latent system
     factor `z` plus idiosyncratic noise, which is the one-factor special case.

  2. **Regime switching.** `z` is drawn from a two-component mixture: a calm
     component and a stressed component with a lower mean. This is how
     wide-area low-renewable events ("Dunkelflaute") are usually represented,
     and it is what produces a genuinely heavy left tail in renewable output
     rather than a Gaussian one.

  3. **Tail dependence between demand and renewables.** Demand loads on `-z`:
     the same weather that suppresses wind raises air-conditioning or heating
     load. This is the single most important feature of real adequacy events
     (ERCOT Feb-2021, CAISO Aug-2020) and it is absent from any model that
     draws demand independently.

  4. **Correlated thermal forced outages.** Units fail independently at a base
     rate, but in the stressed regime a common-mode event (gas supply, freeze)
     can remove a block of the fleet at once. Independent outages alone have a
     thin aggregate tail; common-mode outages do not.

None of these require an inflated VOLL to generate scarcity, which matters
because "they just set VOLL high" is the obvious objection to any tail result.
`run_tail_experiments.py` sweeps VOLL explicitly to rebut it.

Asymmetric adjustment costs
---------------------------
The deviation bid prices of paper Eq. (3) are defined separately for upward and
downward deviations (`alpha^+` and `alpha^-`), but `zavala_funcs.py` hard-codes
them equal at `mc/10`. That is economically backwards. Bringing an uncommitted
thermal unit up in real time incurs start-up fuel, a non-firm gas premium and
ten-minute reserve; backing a committed unit down mostly saves fuel. Real
adjustment costs are strongly asymmetric, and since the day-ahead position is
the *only* thing risk-aware clearing can choose, the asymmetry is precisely the
channel through which it creates value. `ramp_premium` parametrises it and the
W2 experiment sweeps it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# marginal helpers
# --------------------------------------------------------------------------

def _beta_ab_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    """Beta(a, b) on [0,1] matching a target mean and standard deviation.

    Parametrising by (mean, sd) rather than (a, b) keeps the calibration
    legible: a reviewer can check "wind capacity factor, mean 0.35, sd 0.22"
    against published statistics, which they cannot do for Beta(1.2, 2.23).
    """
    mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
    max_var = mean * (1.0 - mean)
    var = float(min(sd * sd, 0.98 * max_var))
    nu = max_var / var - 1.0
    return mean * nu, (1.0 - mean) * nu


def _copula_marginal(latent, mean, sd):
    """Map a standard-normal latent array to Beta(mean, sd) marginals."""
    a, b = _beta_ab_from_mean_sd(mean, sd)
    u = stats.norm.cdf(latent)
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return stats.beta.ppf(u, a, b)


# --------------------------------------------------------------------------
# instance generation
# --------------------------------------------------------------------------

def generate_tail_instance(
    seed: int,
    S: int = 500,
    *,
    # --- fleet ---
    n_wind: int = 5,
    n_solar: int = 4,
    n_ccgt: int = 4,
    n_peaker: int = 3,
    mean_demand: float = 1000.0,
    adequacy: float = 1.12,
    renewable_penetration: float = 0.45,
    # --- dependence ---
    rho_wind: float = 0.80,
    rho_solar: float = 0.70,
    rho_cross: float = 0.35,
    # --- regime switching ---
    p_stress: float = 0.12,
    stress_shift: float = -1.9,
    stress_scale: float = 1.05,
    # --- demand coupling ---
    demand_beta: float = 0.60,
    demand_cv: float = 0.08,
    demand_noise: float = 0.030,
    # --- thermal outages ---
    forced_outage_rate: float = 0.05,
    common_mode_prob: float = 0.30,
    common_mode_frac: float = 0.35,
    # --- renewable marginals ---
    wind_cf_mean: float = 0.35,
    wind_cf_sd: float = 0.22,
    solar_cf_mean: float = 0.22,
    solar_cf_sd: float = 0.16,
    # --- economics ---
    voll: float = 1000.0,
    ramp_premium: float = 1.00,
    backdown_discount: float = 0.05,
    load_delta_frac: float = 0.05,
    renewable_delta_floor: float = 0.50,
    # --- optional physical coupling (NOT in the paper's model; see W-ramp) ---
    ramp_limit_frac: float | None = None,
):
    """Build one tail-heavy market instance.

    Returns a dict with `probs, mc, mv, gbar, dbar, dgp, dgn, ddp, ddn` (the
    arrays the model consumes) plus metadata for logging and diagnostics.

    Key parameters
    --------------
    adequacy : firm thermal nameplate divided by mean demand. The sweep in W3
        moves this; the benefit of risk-aware clearing should vanish at both
        ends (abundant capacity is committed anyway; absent capacity cannot be
        committed at all) and peak in between.
    ramp_premium : `alpha^+ / mc` for thermal units -- the cost of bringing a
        unit up that was not scheduled day-ahead, as a multiple of its marginal
        cost. This is the mechanism under test; W2 sweeps it. 0 recovers the
        symmetric convention of `zavala_funcs.py` up to `backdown_discount`.
    ramp_limit_frac : if set, caps real-time upward deviation at this fraction
        of nameplate, i.e. `G_i(w) - g_i <= frac * nameplate_i`. This models
        start-up times making the day-ahead schedule physically binding. It is
        an EXTENSION beyond the paper's Eq. (63)-(65); leave it None to stay
        within verified theory. Reported separately wherever it is used.
    """
    rng = np.random.default_rng(seed)
    n_thermal = n_ccgt + n_peaker
    G = n_wind + n_solar + n_thermal

    # ---- scenario probabilities -------------------------------------------
    # Near-uniform rather than exactly uniform: a Dirichlet with large
    # concentration keeps the measure non-degenerate without letting any single
    # scenario dominate the CVaR tail by probability alone.
    probs = rng.dirichlet(np.full(S, 1500.0 / S))
    probs = probs / probs.sum()

    # ---- latent weather factor (regime-switching mixture) -----------------
    stressed = rng.random(S) < p_stress
    z = np.where(
        stressed,
        rng.normal(stress_shift, stress_scale, S),
        rng.normal(0.0, 1.0, S),
    )

    # ---- renewable capacity factors via one-factor Gaussian copula --------
    # Wind and solar load on the same system factor with different strengths,
    # and additionally share `rho_cross` through a secondary common term, so the
    # two technologies are correlated with each other as well as with demand.
    cross = rng.standard_normal(S)

    def _latent(n, rho, rho_x):
        idio = rng.standard_normal((S, n))
        common = rho * z[:, None] + rho_x * cross[:, None]
        resid = np.sqrt(max(1e-9, 1.0 - rho**2 - rho_x**2))
        return common + resid * idio

    cf_wind = _copula_marginal(_latent(n_wind, rho_wind, rho_cross),
                               wind_cf_mean, wind_cf_sd)
    cf_solar = _copula_marginal(_latent(n_solar, rho_solar, rho_cross),
                                solar_cf_mean, solar_cf_sd)

    # ---- demand: loads on -z, so bad weather raises it --------------------
    d_mult = (1.0
              + demand_beta * (-z) * demand_cv
              + rng.normal(0.0, demand_noise, S))
    demand = mean_demand * np.clip(d_mult, 0.55, 1.45)
    peak_demand = float(demand.max())

    # ---- fleet sizing -----------------------------------------------------
    # Renewable nameplate is set so that expected renewable energy covers
    # `renewable_penetration` of mean demand; thermal nameplate is set by
    # `adequacy`. Both are expressed relative to demand so the sweeps are
    # interpretable independently of absolute system size.
    wind_share, solar_share = 0.65, 0.35
    wind_nameplate_total = (renewable_penetration * wind_share * mean_demand
                            / max(wind_cf_mean, 1e-6))
    solar_nameplate_total = (renewable_penetration * solar_share * mean_demand
                             / max(solar_cf_mean, 1e-6))
    wind_np = np.full(n_wind, wind_nameplate_total / n_wind)
    solar_np = np.full(n_solar, solar_nameplate_total / n_solar)

    thermal_total = adequacy * mean_demand
    # Mid-merit CCGT carries the bulk; peakers are the marginal tail resource.
    ccgt_total = 0.72 * thermal_total
    peaker_total = thermal_total - ccgt_total
    ccgt_np = np.full(n_ccgt, ccgt_total / max(n_ccgt, 1))
    peaker_np = np.full(n_peaker, peaker_total / max(n_peaker, 1))
    thermal_np = np.concatenate([ccgt_np, peaker_np])

    # ---- thermal availability: independent + common-mode outages ----------
    avail = (rng.random((S, n_thermal)) > forced_outage_rate).astype(float)
    cm_hit = stressed & (rng.random(S) < common_mode_prob)
    n_cm = int(round(common_mode_frac * n_thermal))
    if n_cm > 0:
        for s in np.flatnonzero(cm_hit):
            # Common-mode events hit a contiguous block (shared fuel supply or
            # geography) rather than a random subset.
            start = rng.integers(0, n_thermal)
            idx = (start + np.arange(n_cm)) % n_thermal
            avail[s, idx] = 0.0
    thermal_cap = avail * thermal_np[None, :]

    # ---- assemble capacity matrix (S, G) ----------------------------------
    gbar = np.column_stack([
        cf_wind * wind_np[None, :],
        cf_solar * solar_np[None, :],
        thermal_cap,
    ])
    dbar = demand.reshape(S, 1)

    # ---- costs ------------------------------------------------------------
    mc_wind = rng.uniform(0.0, 2.0, n_wind)
    mc_solar = rng.uniform(0.0, 2.0, n_solar)
    mc_ccgt = rng.uniform(28.0, 45.0, n_ccgt)
    mc_peaker = rng.uniform(65.0, 110.0, n_peaker)
    mc = np.concatenate([mc_wind, mc_solar, mc_ccgt, mc_peaker])
    mv = np.array([float(voll)])

    # ---- deviation bid prices, paper Eq. (3), ASYMMETRIC ------------------
    # Renewables: curtailment is nearly free and upward deviation is bounded by
    # the resource itself, so the premium is a floor rather than a multiple of
    # a near-zero marginal cost. The floor is required: theory note T10 shows
    # the agent LP is unbounded when alpha = 0.
    is_thermal = np.zeros(G, dtype=bool)
    is_thermal[n_wind + n_solar:] = True

    dgp = np.where(is_thermal,
                   ramp_premium * mc,
                   np.maximum(0.10 * mc, renewable_delta_floor))
    dgn = np.where(is_thermal,
                   backdown_discount * mc,
                   np.maximum(0.10 * mc, renewable_delta_floor))
    dgp = np.maximum(dgp, renewable_delta_floor)
    dgn = np.maximum(dgn, renewable_delta_floor)

    ddp = np.full(1, load_delta_frac * voll)
    ddn = np.full(1, load_delta_frac * voll)

    ramp_cap = None
    if ramp_limit_frac is not None:
        nameplate = np.concatenate([wind_np, solar_np, thermal_np])
        ramp_cap = ramp_limit_frac * nameplate
        # Renewables and already-synchronised capacity are not start-limited.
        ramp_cap[:n_wind + n_solar] = nameplate[:n_wind + n_solar]

    # ---- diagnostics ------------------------------------------------------
    renewable_mw = gbar[:, :n_wind + n_solar].sum(axis=1)
    thermal_mw = gbar[:, n_wind + n_solar:].sum(axis=1)
    net_load = demand - renewable_mw
    shortfall = np.maximum(net_load - thermal_mw, 0.0)

    return dict(
        probs=probs, mc=mc, mv=mv, gbar=gbar, dbar=dbar,
        dgp=dgp, dgn=dgn, ddp=ddp, ddn=ddn,
        ramp_cap=ramp_cap,
        meta=dict(
            seed=int(seed), S=int(S), G=int(G),
            n_wind=n_wind, n_solar=n_solar, n_ccgt=n_ccgt, n_peaker=n_peaker,
            adequacy=float(adequacy), ramp_premium=float(ramp_premium),
            backdown_discount=float(backdown_discount), voll=float(voll),
            renewable_penetration=float(renewable_penetration),
            ramp_limit_frac=(None if ramp_limit_frac is None
                             else float(ramp_limit_frac)),
            mean_demand=float(mean_demand), peak_demand=peak_demand,
            thermal_nameplate=float(thermal_np.sum()),
            renewable_nameplate=float(wind_np.sum() + solar_np.sum()),
            # Share of probability mass in which physical capacity cannot meet
            # demand at all. Risk-aware clearing cannot help in those scenarios,
            # so this number bounds what any day-ahead policy can achieve.
            p_physical_shortfall=float(probs[shortfall > 1e-6].sum()),
            expected_shortfall_mw=float(probs @ shortfall),
            p_stressed=float(probs[stressed].sum()),
            mean_renewable_mw=float(probs @ renewable_mw),
            mean_net_load=float(probs @ net_load),
            net_load_p95=float(np.quantile(net_load, 0.95)),
            net_load_p50=float(np.quantile(net_load, 0.50)),
        ),
    )


def tail_dependence_summary(inst) -> dict:
    """Diagnostics a reviewer would ask for about the dependence structure.

    Reports the realised correlation between demand and renewable output and
    the ratio of the 95th to the 50th percentile of net load, which is the
    single number that best captures whether the tail is actually heavy.
    """
    meta = inst["meta"]
    nw, ns = meta["n_wind"], meta["n_solar"]
    gbar, dbar = inst["gbar"], inst["dbar"]
    renewable = gbar[:, :nw + ns].sum(axis=1)
    demand = dbar[:, 0]
    net = demand - renewable
    return dict(
        corr_demand_renewable=float(np.corrcoef(demand, renewable)[0, 1]),
        net_load_p95_over_p50=float(np.quantile(net, 0.95) / max(np.quantile(net, 0.50), 1e-9)),
        net_load_p99_over_p50=float(np.quantile(net, 0.99) / max(np.quantile(net, 0.50), 1e-9)),
        renewable_cf_p05=float(np.quantile(renewable, 0.05) / max(meta["renewable_nameplate"], 1e-9)),
        p_physical_shortfall=meta["p_physical_shortfall"],
        expected_shortfall_mw=meta["expected_shortfall_mw"],
    )
