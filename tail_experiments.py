"""Tail-welfare experiments W0-W6 for the risk-aware clearing paper.

The headline finding, established by W0, shapes everything else
-----------------------------------------------------------------
In the paper's formulation the real-time capacity `gbar(w)` is exogenous, so
`sum_j D_j(w) <= min(sum_i gbar_i(w), sum_j dbar_j(w))`. Because VOLL greatly
exceeds every marginal cost, the planner always serves the physical maximum.
**Unserved energy is therefore identical under every day-ahead policy.** It is
fixed by the scenario before any clearing happens.

Since unserved energy is 75-85% of tail cost, this caps what risk-aware
clearing can achieve: it reallocates adjustment costs, it does not improve
adequacy. W0 measures the ceiling directly. Reporting it is better than letting
a referee find it, and it converts the obvious objection into a contribution.

Consequences for how results are reported
-----------------------------------------
1. **Denominator.** Expected social surplus is dominated by the gross consumer
   value `mv . D ~ 9e5`, which barely moves between policies. A gap quoted as a
   percentage of total welfare is divided by that constant and looks ~6x
   smaller than it is. Every experiment here reports the gap against *tail
   cost* and in `$/MWh served`, and reports the total-welfare percentage too so
   the two can be reconciled.

2. **Baseline.** The honest comparison is risk-aware against *risk-neutral
   stochastic*, both two-stage. The deterministic policy is reported as well,
   but beating it is a weak claim since it never sees the scenario set.

Experiments
-----------
W0  Structural ceiling: unserved energy is policy-invariant.
W1  Paired multi-seed headline with confidence intervals.
W2  Sweep the adjustment-cost asymmetry `alpha+/mc` -- the mechanism.
W3  Sweep adequacy -- where in capacity space the benefit lives.
W4  Sweep VOLL -- rebuts "the result is an artefact of a high VOLL".
W5  Tail cost decomposition -- what risk aversion actually buys.
W6  Risk-return frontier in `eps`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from tail_scenarios import generate_tail_instance, tail_dependence_summary
from tail_model import (
    solve_planner, solve_deterministic_da, evaluate_da,
    score, cvar, var_level, tail_decomposition,
)


# --------------------------------------------------------------------------
# core: solve all policies on one instance and score them identically
# --------------------------------------------------------------------------

def run_instance(seed, S, eps, beta, solver, use_ramp=False,
                 include_deterministic=True, **gen_kw):
    """Solve every policy on one instance and score through a common evaluator."""
    inst = generate_tail_instance(seed, S=S, **gen_kw)
    probs = inst["probs"]

    policies = {}
    rn = solve_planner(inst, eps=0.0, beta=beta, solver=solver, use_ramp=use_ramp)
    ra = solve_planner(inst, eps=eps, beta=beta, solver=solver, use_ramp=use_ramp)
    policies["risk_neutral"] = rn
    policies["risk_aware"] = ra
    if include_deterministic:
        policies["deterministic"] = solve_deterministic_da(inst, solver=solver)

    out = {}
    for name, pol in policies.items():
        ev = evaluate_da(inst, pol["g"], pol["d"], solver=solver, use_ramp=use_ramp)
        s = score(probs, ev, beta=beta)
        s["tail"] = tail_decomposition(probs, ev, beta=beta)
        s["tail_cost"] = float(sum(
            s["tail"][k] for k in
            ("energy", "dev_up", "dev_dn", "load_dev", "unserved_cost")))
        s["da_total"] = float(pol["g"].sum())
        s["p0"] = float(pol["p0"])
        s["served_mwh"] = float(probs @ ev["D"].sum(axis=1))
        out[name] = s

    out["_meta"] = inst["meta"]
    out["_dep"] = tail_dependence_summary(inst)
    out["_eps"] = float(eps)
    out["_prices"] = dict(
        p0_risk_aware=float(ra["p0"]), p0_risk_neutral=float(rn["p0"]),
        E_rho_Pi=float(np.nansum(probs * ra["Pi"])),
        E_rhotilde_Pi=float(np.nansum(ra["rho_tilde"] * ra["Pi"])),
        n_priced=int(ra["support"].sum()),
    )
    return out


def gaps(res, ref="risk_neutral", tgt="risk_aware"):
    """Gap metrics for one instance, with every denominator we report."""
    a, b = res[tgt], res[ref]
    d_cvar = a["cvar_welfare"] - b["cvar_welfare"]
    d_exp = a["expected_welfare"] - b["expected_welfare"]
    return dict(
        d_cvar_welfare=float(d_cvar),
        d_expected_welfare=float(d_exp),
        # Honest denominator: the cost the policy can actually influence.
        pct_of_tail_cost=float(100.0 * d_cvar / max(b["tail_cost"], 1e-9)),
        # The conventional but diluted denominator, reported for reconciliation.
        pct_of_total_welfare=float(100.0 * d_cvar / max(abs(b["cvar_welfare"]), 1e-9)),
        # Directly comparable to the S5 slope (value = 13.05 x delta_scale).
        dollars_per_mwh=float(d_cvar / max(b["served_mwh"], 1e-9)),
        d_cvar_unserved=float(a["cvar_unserved"] - b["cvar_unserved"]),
        d_dev_up=float(a["expected_dev_up"] - b["expected_dev_up"]),
        d_dev_dn=float(a["expected_dev_dn"] - b["expected_dev_dn"]),
        d_da_total=float(a["da_total"] - b["da_total"]),
    )


def paired_ci(x, conf=0.95):
    """Mean, CI and paired tests for a vector of per-seed paired differences.

    Paired across a common seed, so the instance-to-instance variance -- which
    dwarfs the treatment effect -- cancels. Reporting unpaired means with SDs
    would hide a real effect inside noise.
    """
    x = np.asarray(x, float).reshape(-1)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, sem=np.nan,
                    t_p=np.nan, wilcoxon_p=np.nan, frac_positive=np.nan)
    m, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    half = stats.t.ppf(0.5 + conf / 2.0, n - 1) * sem if n > 1 else 0.0
    t_p = float(stats.ttest_1samp(x, 0.0).pvalue) if n > 1 and sd > 0 else np.nan
    try:
        w_p = float(stats.wilcoxon(x).pvalue) if n >= 6 and np.any(x != 0) else np.nan
    except ValueError:
        w_p = np.nan
    return dict(n=int(n), mean=m, sd=sd, sem=float(sem),
                lo=float(m - half), hi=float(m + half),
                t_p=t_p, wilcoxon_p=w_p,
                frac_positive=float((x > 0).mean()))


def _sweep(seeds, S, beta, solver, varying, base_kw, eps, use_ramp=False,
           rep=print):
    """Run a paired sweep: for each level, every seed, collect gap metrics."""
    rows = []
    for level_name, level_kw in varying:
        per_seed, meta_acc = [], []
        for sd in seeds:
            kw = dict(base_kw)
            kw.update(level_kw)
            e = kw.pop("_eps", eps)
            try:
                res = run_instance(sd, S, e, beta, solver, use_ramp=use_ramp,
                                   include_deterministic=False, **kw)
            except Exception as exc:                            # noqa: BLE001
                rep(f"    [skip] level={level_name} seed={sd}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            per_seed.append(gaps(res))
            meta_acc.append(res["_meta"])
        if not per_seed:
            continue
        row = dict(level=level_name, n_seeds=len(per_seed))
        for k in per_seed[0]:
            row[k] = paired_ci([g[k] for g in per_seed])
        if meta_acc:
            row["p_physical_shortfall"] = float(
                np.mean([m["p_physical_shortfall"] for m in meta_acc]))
        rows.append(row)
        rep(f"    {level_name:>12}  "
            f"tailcost% {row['pct_of_tail_cost']['mean']:+7.3f} "
            f"[{row['pct_of_tail_cost']['lo']:+.3f},{row['pct_of_tail_cost']['hi']:+.3f}]   "
            f"$/MWh {row['dollars_per_mwh']['mean']:+7.4f}   "
            f"d_unserved {row['d_cvar_unserved']['mean']:+8.3f}")
    return rows


# --------------------------------------------------------------------------
# W0 : the structural ceiling
# --------------------------------------------------------------------------

def W0_structural_ceiling(seeds, S, eps, beta, solver, base_kw, rep=print):
    """Unserved energy is identical under every day-ahead policy.

    Establishes the ceiling on what risk-aware clearing can do, and separates
    the share of tail cost that is reachable from the share that is not.
    """
    rep("\n" + "=" * 78)
    rep("W0  Structural ceiling: unserved energy is policy-invariant")
    rep("=" * 78)
    rep("sum_j D_j(w) <= min(sum_i gbar_i(w), sum_j dbar_j(w)) and VOLL >> mc, so")
    rep("the planner always serves the physical maximum. Shedding is fixed by the")
    rep("scenario, not by the policy. Risk-aware clearing can only move the")
    rep("remaining adjustment cost. This bounds every result that follows.")

    rows = []
    for sd in seeds:
        try:
            res = run_instance(sd, S, eps, beta, solver, **base_kw)
        except Exception as exc:                                # noqa: BLE001
            rep(f"  [skip] seed={sd}: {type(exc).__name__}: {exc}")
            continue
        rn, ra = res["risk_neutral"], res["risk_aware"]
        det = res.get("deterministic")
        t = rn["tail"]
        reachable = t["energy"] + t["dev_up"] + t["dev_dn"] + t["load_dev"]
        rows.append(dict(
            seed=int(sd),
            unserved_rn=rn["cvar_unserved"], unserved_ra=ra["cvar_unserved"],
            unserved_det=(det["cvar_unserved"] if det else np.nan),
            d_unserved=ra["cvar_unserved"] - rn["cvar_unserved"],
            unserved_cost=t["unserved_cost"],
            reachable_cost=float(reachable),
            tail_cost=rn["tail_cost"],
            unreachable_share=float(t["unserved_cost"] / max(rn["tail_cost"], 1e-9)),
        ))

    if not rows:
        return dict(rows=[], summary={})

    d_uns = np.array([r["d_unserved"] for r in rows])
    share = np.array([r["unreachable_share"] for r in rows])
    scale = float(np.mean([r["unserved_rn"] for r in rows]))
    # Relative tolerance: the claim is that the two are the same number, and
    # barrier returns each to ~1e-8 relative. An absolute threshold would be
    # reporting solver precision rather than the invariance.
    max_abs = float(np.abs(d_uns).max())
    rel = max_abs / max(scale, 1e-9)
    summary = dict(
        max_abs_d_unserved=max_abs,
        rel_d_unserved=rel,
        mean_unserved=scale,
        mean_unreachable_share=float(share.mean()),
        mean_reachable_cost=float(np.mean([r["reachable_cost"] for r in rows])),
        mean_tail_cost=float(np.mean([r["tail_cost"] for r in rows])),
        invariant=bool(rel < 1e-6),
    )
    rep(f"\n  max |CVaR[unserved]_aware - CVaR[unserved]_neutral| = "
        f"{max_abs:.3e} MW   on a base of {scale:.2f} MW  (relative {rel:.2e})")
    rep(f"  -> unserved energy is {'INVARIANT' if summary['invariant'] else 'NOT invariant'} "
        f"to the day-ahead policy")
    rep(f"  share of tail cost that is unreachable (unserved x VOLL): "
        f"{100*summary['mean_unreachable_share']:.1f}%")
    rep(f"  reachable tail cost (energy + deviations): "
        f"{summary['mean_reachable_cost']:.1f} of {summary['mean_tail_cost']:.1f}")
    amp = 1.0 / max(1.0 - summary["mean_unreachable_share"], 1e-9)
    rep("\n  Read this as the ceiling: a gap quoted as X% of tail cost is")
    rep(f"  {amp:.2f}x that as a share of the cost the policy can actually move.")
    summary["reachable_amplification"] = float(amp)
    return dict(rows=rows, summary=summary)


# --------------------------------------------------------------------------
# W1 : paired headline
# --------------------------------------------------------------------------

def W1_headline(seeds, S, eps, beta, solver, base_kw, rep=print):
    rep("\n" + "=" * 78)
    rep(f"W1  Paired headline, {len(seeds)} seeds, eps={eps}, beta={beta}")
    rep("=" * 78)
    rep("Differences are paired on the seed, so instance-to-instance variance")
    rep("cancels. Both comparisons are reported: against the risk-neutral")
    rep("stochastic planner (the honest one) and against the deterministic")
    rep("policy (easier, but it never sees the scenario set).")

    vs_rn, vs_det, per_seed = [], [], []
    for sd in seeds:
        try:
            res = run_instance(sd, S, eps, beta, solver, **base_kw)
        except Exception as exc:                                # noqa: BLE001
            rep(f"  [skip] seed={sd}: {type(exc).__name__}: {exc}")
            continue
        vs_rn.append(gaps(res, "risk_neutral", "risk_aware"))
        if "deterministic" in res:
            vs_det.append(gaps(res, "deterministic", "risk_aware"))
        per_seed.append(dict(
            seed=int(sd),
            cvar_rn=res["risk_neutral"]["cvar_welfare"],
            cvar_ra=res["risk_aware"]["cvar_welfare"],
            cvar_det=res.get("deterministic", {}).get("cvar_welfare", np.nan),
            exp_rn=res["risk_neutral"]["expected_welfare"],
            exp_ra=res["risk_aware"]["expected_welfare"],
            tail_cost_rn=res["risk_neutral"]["tail_cost"],
            prices=res["_prices"], dep=res["_dep"],
        ))

    def _summarise(rows, label):
        if not rows:
            return {}
        out = {k: paired_ci([r[k] for r in rows]) for k in rows[0]}
        rep(f"\n  --- risk-aware vs {label} (n={out['d_cvar_welfare']['n']}) ---")
        for key, unit in (("pct_of_tail_cost", "% of tail cost"),
                          ("pct_of_total_welfare", "% of total welfare"),
                          ("dollars_per_mwh", "$/MWh served"),
                          ("d_cvar_welfare", "absolute $")):
            c = out[key]
            rep(f"    CVaR welfare gain, {unit:<20} "
                f"{c['mean']:+10.4f}  95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]"
                f"   p={c['t_p']:.2e}" if np.isfinite(c["t_p"])
                else f"    CVaR welfare gain, {unit:<20} {c['mean']:+10.4f}")
        c = out["d_expected_welfare"]
        rep(f"    Expected-welfare change          {c['mean']:+10.2f}"
            f"  95% CI [{c['lo']:+.2f}, {c['hi']:+.2f}]"
            "   (negative is the price paid for tail protection)")
        c = out["pct_of_tail_cost"]
        rep(f"    seeds improved: {100*c['frac_positive']:.0f}%   "
            f"Wilcoxon p={c['wilcoxon_p']:.2e}" if np.isfinite(c["wilcoxon_p"])
            else f"    seeds improved: {100*c['frac_positive']:.0f}%")
        return out

    return dict(
        vs_risk_neutral=_summarise(vs_rn, "risk-neutral stochastic"),
        vs_deterministic=_summarise(vs_det, "deterministic"),
        per_seed=per_seed,
    )


# --------------------------------------------------------------------------
# W2-W4 : sweeps
# --------------------------------------------------------------------------

def W2_asymmetry(seeds, S, eps, beta, solver, base_kw, levels, rep=print):
    rep("\n" + "=" * 78)
    rep("W2  Adjustment-cost asymmetry sweep (alpha+ / mc)")
    rep("=" * 78)
    rep("The day-ahead position is the only thing risk aversion chooses, and it")
    rep("is priced only through the deviation terms, so the benefit must scale")
    rep("with alpha+. A monotone response is evidence the mechanism is the one")
    rep("claimed rather than a numerical accident.")
    return _sweep(seeds, S, beta, solver,
                  [(f"{v:g}", dict(ramp_premium=v)) for v in levels],
                  base_kw, eps, rep=rep)


def W3_adequacy(seeds, S, eps, beta, solver, base_kw, levels, rep=print):
    rep("\n" + "=" * 78)
    rep("W3  Adequacy sweep (thermal nameplate / mean demand)")
    rep("=" * 78)
    rep("Expect the benefit to vanish at both ends: with abundant capacity the")
    rep("risk-neutral planner already commits enough, and with too little no")
    rep("day-ahead choice can conjure capacity that does not exist.")
    return _sweep(seeds, S, beta, solver,
                  [(f"{v:g}", dict(adequacy=v)) for v in levels],
                  base_kw, eps, rep=rep)


def W4_voll(seeds, S, eps, beta, solver, base_kw, levels, rep=print):
    rep("\n" + "=" * 78)
    rep("W4  VOLL sweep -- rebuttal to 'the tail result is a high-VOLL artefact'")
    rep("=" * 78)
    rep("VOLL scales the unserved-energy term, which W0 shows is identical under")
    rep("every policy. If the gap in $/MWh is roughly flat in VOLL, the result")
    rep("does not come from VOLL. Note the load's deviation price is tied to")
    rep("VOLL by load_delta_frac, so a mild slope is expected.")
    return _sweep(seeds, S, beta, solver,
                  [(f"{v:g}", dict(voll=v)) for v in levels],
                  base_kw, eps, rep=rep)


# --------------------------------------------------------------------------
# W5 : decomposition
# --------------------------------------------------------------------------

def W5_decomposition(seeds, S, eps, beta, solver, base_kw, rep=print):
    rep("\n" + "=" * 78)
    rep("W5  Tail cost decomposition: what risk aversion actually buys")
    rep("=" * 78)

    acc = {}
    for sd in seeds:
        try:
            res = run_instance(sd, S, eps, beta, solver, **base_kw)
        except Exception as exc:                                # noqa: BLE001
            rep(f"  [skip] seed={sd}: {type(exc).__name__}: {exc}")
            continue
        for pol in ("risk_neutral", "risk_aware", "deterministic"):
            if pol in res:
                acc.setdefault(pol, []).append(res[pol]["tail"])

    keys = ("energy", "dev_up", "dev_dn", "load_dev", "unserved_cost")
    means = {p: {k: float(np.mean([t[k] for t in rows])) for k in keys}
             for p, rows in acc.items()}
    rep(f"\n  {'component':>14} " + "".join(f"{p:>16}" for p in means))
    for k in keys:
        rep(f"  {k:>14} " + "".join(f"{means[p][k]:>16.1f}" for p in means))
    if "risk_neutral" in means and "risk_aware" in means:
        rep("\n  change, risk-aware minus risk-neutral:")
        for k in keys:
            d = means["risk_aware"][k] - means["risk_neutral"][k]
            rep(f"    {k:>14} {d:>+14.1f}")
    return dict(means=means)


# --------------------------------------------------------------------------
# W6 : risk-return frontier
# --------------------------------------------------------------------------

def W6_frontier(seeds, S, beta, solver, base_kw, eps_list, rep=print):
    rep("\n" + "=" * 78)
    rep("W6  Risk-return frontier in eps")
    rep("=" * 78)
    rep("Expected welfare traded against tail welfare. eps must stay strictly")
    rep("below 1: at eps=1 the tilted measure collapses onto the tail and the")
    rep("real-time prices of Eq. (107) cease to exist off it (theory note T11).")

    rows = []
    for e in eps_list:
        if e >= 1.0:
            rep(f"    [skip] eps={e}: Eq. (107) requires eps < 1 (T11)")
            continue
        exp_, cv_, d_exp, d_cv, p0s = [], [], [], [], []
        for sd in seeds:
            try:
                res = run_instance(sd, S, e, beta, solver,
                                   include_deterministic=False, **base_kw)
            except Exception as exc:                            # noqa: BLE001
                rep(f"    [skip] eps={e} seed={sd}: {type(exc).__name__}: {exc}")
                continue
            g = gaps(res)
            exp_.append(res["risk_aware"]["expected_welfare"])
            cv_.append(res["risk_aware"]["cvar_welfare"])
            d_exp.append(g["d_expected_welfare"])
            d_cv.append(g["d_cvar_welfare"])
            p0s.append(res["_prices"]["p0_risk_aware"])
        if not exp_:
            continue
        rows.append(dict(eps=float(e),
                         expected_welfare=paired_ci(exp_),
                         cvar_welfare=paired_ci(cv_),
                         d_expected=paired_ci(d_exp),
                         d_cvar=paired_ci(d_cv),
                         p0=paired_ci(p0s)))
        rep(f"    eps={e:<5} E[W] {np.mean(exp_):12.1f}   "
            f"CVaR[W] {np.mean(cv_):12.1f}   "
            f"dE {np.mean(d_exp):+9.1f}   dCVaR {np.mean(d_cv):+9.1f}   "
            f"p0 {np.mean(p0s):8.2f}")
    return rows
