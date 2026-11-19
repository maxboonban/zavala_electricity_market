"""Diagnostic: does the W-suite share the IM3 instance's no-scarcity defect?

    python check_w_suite_scarcity.py          # needs numpy + scipy, no solver

`check_tail_composition.py` showed that the headline IM3 instance is never
scarce (firm thermal exceeds peak demand in every scenario by construction,
because `load_scale = firm/(1+margin)/peak`), so its loss
`cost - VOLL*demand` is a near-perfectly decreasing function of demand and the
beta-tail is the LOW-demand evenings rather than an adequacy event.

The W-suite (`tail_scenarios.py`) is sized differently -- `thermal_total =
adequacy * mean_demand`, not against peak -- and its thermal capacity is
scenario-dependent through independent and common-mode forced outages. So it
may be genuinely scarce. This script checks, without a solver, by building the
same instances `run_tail_experiments.py` builds and asking:

  1. With what probability is physical capacity short of demand?
  2. Is the beta-tail of the deviation-free loss a HIGH-demand / LOW-renewable
     event (scarcity) or a LOW-demand one (the IM3 pathology)?
  3. How much of the variance of the loss is lost-load value rather than the
     served-demand constant that cancels nothing?

Unlike IM3, served demand here is a decision: when capacity is short the model
curtails, so `VOLL * (dbar - served)` is real lost-load value, not an additive
constant in disguise. The deviation-free loss used below is the same quantity
the planner minimises with the day-ahead deviation terms switched off:

    l0(w) = merit_cost(w) - VOLL * served(w)

Written while auditing the W0-W7 tail-welfare suite; see docs/EXPERIMENT_LOG.md.
"""
import numpy as np

from tail_scenarios import generate_tail_instance

# W0 headline configuration, exactly as run_tail_experiments.py sets it.
BASE = dict(ramp_premium=1.0, backdown_discount=0.4, adequacy=1.12,
            voll=1000.0, renewable_penetration=0.45, p_stress=0.12,
            ramp_limit_frac=None)
S = 500
SEEDS = 30
BETA = 0.95


def cvar(x, p, beta):
    o = np.argsort(x)
    xs, ps = x[o], p[o]
    i = min(int(np.searchsorted(np.cumsum(ps), beta, side="left")), xs.size - 1)
    var = xs[i]
    return float(var + (ps @ np.maximum(xs - var, 0.0)) / (1.0 - beta))


def merit_fill(gbar_w, mc, need):
    """Cheapest way to serve `need` MW from the available capacity.

    Returns (cost, served). When capacity is short, serves everything it has.
    """
    order = np.argsort(mc)
    cost = 0.0
    served = 0.0
    for k in order:
        take = min(gbar_w[k], need - served)
        if take <= 0.0:
            continue
        cost += take * mc[k]
        served += take
        if need - served <= 1e-9:
            break
    return cost, served


def pctile(x, mask):
    """Mean percentile rank, within x, of the entries selected by `mask`."""
    return float(np.mean([(x < x[w]).mean() for w in np.flatnonzero(mask)]) * 100)


rows = []
print("=" * 78)
print("W-suite (tail_scenarios.generate_tail_instance) scarcity audit")
print(f"  adequacy={BASE['adequacy']}  voll={BASE['voll']}  "
      f"p_stress={BASE['p_stress']}  S={S}  seeds={SEEDS}  beta={BETA}")
print("=" * 78)

for seed in range(SEEDS):
    inst = generate_tail_instance(seed, S=S, **BASE)
    probs = inst["probs"]
    mc = inst["mc"]
    gbar = inst["gbar"]
    dbar = inst["dbar"][:, 0]
    voll = BASE["voll"]
    meta = inst["meta"]

    nr = meta["n_wind"] + meta["n_solar"]
    ren = gbar[:, :nr].sum(axis=1)
    th = gbar[:, nr:].sum(axis=1)
    cap = gbar.sum(axis=1)

    cost = np.empty(S)
    served = np.empty(S)
    for w in range(S):
        cost[w], served[w] = merit_fill(gbar[w], mc, dbar[w])

    lost = dbar - served
    ell0 = cost - voll * served

    thr = np.sort(ell0)[int(BETA * S)]
    tail = ell0 >= thr

    rows.append(dict(
        seed=seed,
        p_short=float(probs[lost > 1e-6].sum()),
        n_short=int((lost > 1e-6).sum()),
        els_mw=float(probs @ lost),
        tail_lost=float(lost[tail].mean()),
        tail_dpct=pctile(dbar, tail),
        tail_rpct=pctile(ren, tail),
        tail_thpct=pctile(th, tail),
        corr_d=float(np.corrcoef(ell0, dbar)[0, 1]),
        corr_r=float(np.corrcoef(ell0, ren)[0, 1]),
        corr_th=float(np.corrcoef(ell0, th)[0, 1]),
        sd_ell=float(ell0.std()),
        sd_lost=float((voll * lost).std()),
        sd_cost=float(cost.std()),
        sd_servedval=float((voll * served).std()),
        cvar=cvar(ell0, probs, BETA),
        mean=float(probs @ ell0),
    ))

k = lambda f: np.array([r[f] for r in rows])

print("\n--- 1. IS IT EVER SCARCE? ---")
print(f"  P(physical shortfall), mean over seeds   {k('p_short').mean():8.4f}"
      f"   (min {k('p_short').min():.4f}, max {k('p_short').max():.4f})")
print(f"  scenarios short per seed                 {k('n_short').mean():8.1f} of {S}"
      f"   (min {k('n_short').min():.0f}, max {k('n_short').max():.0f})")
print(f"  seeds with ZERO short scenarios          {int((k('n_short') == 0).sum())} of {SEEDS}")
print(f"  expected lost load                       {k('els_mw').mean():8.2f} MW")

print("\n--- 2. WHAT IS IN THE beta-TAIL? ---")
print(f"  mean lost load in tail scenarios         {k('tail_lost').mean():8.2f} MW")
print(f"  tail demand    percentile                {k('tail_dpct').mean():8.1f}th")
print(f"  tail renewable percentile                {k('tail_rpct').mean():8.1f}th")
print(f"  tail thermal   percentile                {k('tail_thpct').mean():8.1f}th")
print(f"  corr(loss, demand)                       {k('corr_d').mean():+8.4f}")
print(f"  corr(loss, renewable)                    {k('corr_r').mean():+8.4f}")
print(f"  corr(loss, available thermal)            {k('corr_th').mean():+8.4f}")

print("\n--- 3. WHERE DOES THE VARIANCE LIVE? ---")
print(f"  std( loss )                              {k('sd_ell').mean():12,.1f}")
print(f"  std( VOLL * lost load )                  {k('sd_lost').mean():12,.1f}"
      f"   <- {100*k('sd_lost').mean()/k('sd_ell').mean():.1f}% of total")
print(f"  std( VOLL * served )                     {k('sd_servedval').mean():12,.1f}"
      f"   <- {100*k('sd_servedval').mean()/k('sd_ell').mean():.1f}%")
print(f"  std( merit cost )                        {k('sd_cost').mean():12,.1f}"
      f"   <- {100*k('sd_cost').mean()/k('sd_ell').mean():.1f}%")

print("\n--- VERDICT ---")
scarce = k('p_short').mean() > 1e-4
hi_demand = k('tail_dpct').mean() > 50.0
if scarce and hi_demand:
    print("  The W-suite does NOT share the IM3 defect: shortfall occurs with")
    print("  positive probability and the beta-tail is a HIGH-demand event.")
elif not scarce:
    print("  Same defect as IM3: capacity is never short, so the tail cannot be")
    print("  an adequacy event.")
else:
    print("  Mixed: shortfall occurs, but the tail is not a high-demand event.")
    print("  Inspect the per-seed table below.")

print("\n--- per-seed detail ---")
print(f"  {'seed':>4} {'n_short':>8} {'P(short)':>9} {'E[LOL]MW':>9} "
      f"{'tailD%':>7} {'tailR%':>7} {'corr_d':>8} {'CVaR':>14} {'mean':>14}")
for r in rows:
    print(f"  {r['seed']:>4} {r['n_short']:>8} {r['p_short']:>9.4f} "
          f"{r['els_mw']:>9.2f} {r['tail_dpct']:>7.1f} {r['tail_rpct']:>7.1f} "
          f"{r['corr_d']:>+8.4f} {r['cvar']:>14,.1f} {r['mean']:>14,.1f}")
