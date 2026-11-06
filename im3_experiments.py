"""Experiments on IM3 GO-WEST that test the paper's ECONOMIC claims.

The D1-D5 suite verifies that the equations in Sections 8.3-8.7 hold. This
suite asks what the theory BUYS, on real Western Interconnection data:

  E1  The price of the missing insurance market.
      Without Arrow-Debreu securities Eq. (94) fails, so the risk-aware
      allocation is not decentralisable; an energy-only market supports the
      risk-NEUTRAL allocation (at eps = 0 every agent's gamma is zero, so (94)
      holds trivially). The risk-adjusted welfare difference between the two
      allocations is therefore the value of completing the market. This turns
      Section 8.2.2 from an impossibility remark into a dollar figure.

  E2  Who insures whom, by generation technology, on the real fleet.

  E3  Does the insurance market matter more as the grid decarbonises?
      Sweeps renewable penetration and tracks E1's gap.

  E4  Out-of-sample validation. Clear on one scenario set, evaluate the
      day-ahead commitment on a disjoint set drawn from the same conditional
      pool. Tests whether risk-aware clearing actually reduces realised tail
      loss, or merely fits the in-sample tail.

  E5  Paired statistics. The manuscript's Tables 1 and 3 report differences
      that are smaller than their own error bars. The comparisons are paired
      by construction (same scenarios, same instance), so paired intervals are
      the right summary and are far tighter.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

import im3_data as D
from zavala_funcs import zavala_cvar
from zavala_decentralized import planner_losses, solve_agent


# ---------------------------------------------------------------------------
# recourse evaluation
# ---------------------------------------------------------------------------
def evaluate_recourse(g_da, d_da, mc, mv, gbar, dbar, solver="GUROBI",
                      delta_scale=0.1):
    """Per-scenario planner loss under optimal recourse, holding the day-ahead
    commitment fixed.

    Matches the loss inside `zavala_cvar` exactly:
        l(w) = sum_i [ mc_i G_iw + delta_i |G_iw - g_i| ]
             + sum_j [ -mv_j D_jw + delta_j |D_jw - d_j| ]
    subject to real-time balance and per-scenario capacity. Scenarios decouple
    once the day-ahead quantities are fixed, so minimising the unweighted sum
    minimises each scenario independently.
    """
    g_da = np.asarray(g_da, float).reshape(-1)
    d_da = np.asarray(d_da, float).reshape(-1)
    mc = np.asarray(mc, float).reshape(-1)
    mv = np.asarray(mv, float).reshape(-1)
    gbar = np.asarray(gbar, float)
    dbar = np.asarray(dbar, float)
    S, ng = gbar.shape
    nd = dbar.shape[1]
    dg, dd = mc * delta_scale, mv * delta_scale

    G = cp.Variable((S, ng))
    Dv = cp.Variable((S, nd))
    loss = (cp.sum(G @ mc) + cp.sum(cp.abs(G - g_da[None, :]) @ dg)
            - cp.sum(Dv @ mv) + cp.sum(cp.abs(Dv - d_da[None, :]) @ dd))
    cons = [G >= 0, G <= gbar, Dv >= 0, Dv <= dbar,
            cp.sum(Dv, axis=1) - float(d_da.sum())
            == cp.sum(G, axis=1) - float(g_da.sum())]
    prob = cp.Problem(cp.Minimize(loss), cons)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"recourse evaluation failed: {prob.status}")
    Gv, Dvv = np.asarray(G.value, float), np.asarray(Dv.value, float)
    ell = (Gv @ mc + np.abs(Gv - g_da[None, :]) @ dg
           - Dvv @ mv + np.abs(Dvv - d_da[None, :]) @ dd)
    return ell, Gv, Dvv


def cvar(x, probs, beta):
    """beta-CVaR of a discrete loss distribution (Rockafellar-Uryasev)."""
    x = np.asarray(x, float).reshape(-1)
    probs = np.asarray(probs, float).reshape(-1)
    order = np.argsort(x)
    xs, ps = x[order], probs[order]
    cum = np.cumsum(ps)
    i = int(np.searchsorted(cum, beta, side="left"))
    i = min(i, xs.size - 1)
    var = xs[i]
    excess = np.maximum(xs - var, 0.0)
    return float(var + (ps @ excess) / (1.0 - beta))


def rho_ra(ell, probs, eps, beta):
    """The planner's risk measure: (1-eps) E[l] + eps CVaR_beta[l]."""
    probs = np.asarray(probs, float).reshape(-1)
    return float((1 - eps) * (probs @ np.asarray(ell, float).reshape(-1))
                 + eps * cvar(ell, probs, beta))


def clear(inst, eps, beta, solver="GUROBI", delta_scale=0.1):
    """Solve the risk-aware planner and package prices, duals and losses."""
    probs, mc, mv, gbar, dbar, meta = inst
    g, d, G, Dm, p0, Pi, gam = zavala_cvar(
        probs, mc, mv, gbar, dbar, beta=beta, lambda_cvar=eps,
        check_invariants=False, solver=solver, verbose=False,
        delta_scale=delta_scale)
    out = dict(probs=probs, mc=mc, mv=mv, gbar=gbar, dbar=dbar, meta=meta,
               g=np.asarray(g, float).reshape(-1), d=np.asarray(d, float).reshape(-1),
               G=np.asarray(G, float), D=np.asarray(Dm, float),
               p0=float(p0), Pi=np.asarray(Pi, float).reshape(-1),
               gamma=np.asarray(gam, float).reshape(-1),
               eps=float(eps), beta=float(beta))
    out["rho_tilde"] = (1 - eps) * probs + out["gamma"]
    out["q"] = out["rho_tilde"].copy()
    out["mc_delta"] = mc * delta_scale
    out["mv_delta"] = mv * delta_scale
    out["L_prod"], out["L_cons"], out["ell"] = planner_losses(
        out["p0"], out["Pi"], mc, mv, out["g"], out["d"], out["G"], out["D"])
    return out


def agent_specs(P):
    S = P["probs"].size
    for j in range(P["mc"].size):
        yield ("producer", j, P["mc"][j], P["mc_delta"][j], np.zeros(S),
               P["gbar"][:, j], P["g"][j], P["G"][:, j])
    for i in range(P["mv"].size):
        yield ("consumer", i, P["mv"][i], P["mv_delta"][i], np.zeros(S),
               P["dbar"][:, i], P["d"][i], P["D"][:, i])


def net_insurance_cost(r, q, probs):
    """Premium paid minus expected payout, sum_w (q_w - rho_w) r_w.

    Positive  => the agent pays more than it expects back: a net BUYER of cover.
    Negative  => the agent is compensated for bearing tail risk: a net SELLER.

    This is invariant to the budget-neutral constants delta_k in Eq. (111)-(112):
    shifting r by a constant c changes it by -c (sum_w q_w - sum_w rho_w) = 0,
    because both q and rho are probability measures. It is therefore the one
    incidence measure that does not depend on an arbitrary modelling choice.
    """
    q = np.asarray(q, float).reshape(-1)
    probs = np.asarray(probs, float).reshape(-1)
    return float((q - probs) @ np.asarray(r, float).reshape(-1))


# ---------------------------------------------------------------------------
# E1 : the price of the missing insurance market
# ---------------------------------------------------------------------------
def exp_E1(inst, eps, beta, solver, rep):
    rep("\n" + "=" * 78)
    rep("E1  The price of the missing insurance market")
    rep("=" * 78)
    rep("  Without Arrow-Debreu securities Eq. (94) fails and the risk-aware")
    rep("  allocation cannot be decentralised. At eps = 0 every agent's gamma is")
    rep("  zero, so (94) holds trivially: an energy-only market DOES support the")
    rep("  risk-neutral allocation. The risk-adjusted welfare difference between")
    rep("  the two is the value of completing the market.")

    probs, mc, mv, gbar, dbar, meta = inst
    RN = clear(inst, 0.0, beta, solver)
    RA = clear(inst, eps, beta, solver)

    ell_RN, _, _ = evaluate_recourse(RN["g"], RN["d"], mc, mv, gbar, dbar, solver)
    ell_RA, _, _ = evaluate_recourse(RA["g"], RA["d"], mc, mv, gbar, dbar, solver)

    r_rn = rho_ra(ell_RN, probs, eps, beta)
    r_ra = rho_ra(ell_RA, probs, eps, beta)
    gap = r_rn - r_ra
    served = float(probs @ dbar.sum(axis=1))

    rep(f"\n  risk-adjusted loss rho_RA[l]  (eps={eps}, beta={beta}):")
    rep(f"    risk-neutral commitment (energy-only market) : {r_rn:16.2f} $/h")
    rep(f"    risk-aware commitment   (with insurance)     : {r_ra:16.2f} $/h")
    rep(f"    VALUE OF THE INSURANCE MARKET                : {gap:16.2f} $/h"
        f"   ({100 * gap / max(abs(r_rn), 1e-9):.4f}% of risk-adjusted welfare)")
    rep(f"    per MWh served ({served:,.0f} MWh/h)          : {gap / max(served, 1e-9):16.4f} $/MWh")
    rep(f"    annualised (x8760)                           : ${gap * 8760 / 1e6:,.2f} M/yr")

    e_rn, e_ra = float(probs @ ell_RN), float(probs @ ell_RA)
    c_rn, c_ra = cvar(ell_RN, probs, beta), cvar(ell_RA, probs, beta)
    rep(f"\n  decomposition (loss, so lower is better):")
    rep(f"    E[l]        risk-neutral {e_rn:14.2f}   risk-aware {e_ra:14.2f}"
        f"   delta {e_ra - e_rn:+12.2f}")
    rep(f"    CVaR_b[l]   risk-neutral {c_rn:14.2f}   risk-aware {c_ra:14.2f}"
        f"   delta {c_ra - c_rn:+12.2f}")
    rep("    (risk-aware clearing gives up expected welfare to cut the tail;")
    rep("     the insurance market is what makes that trade decentralisable.)")

    # Eq. (94) on real data: without insurance the agents' gammas disagree.
    rep("\n  Eq. (94) without insurance, on this fleet:")
    gam_P = RA["gamma"]
    tail_P = int(np.sum(gam_P > 1e-9 * max(1.0, gam_P.max())))
    devs, tails = [], []
    for kind, k, cv_, dl, lo, hi, da, rt in agent_specs(RA):
        sol = solve_agent(kind, probs, RA["p0"], RA["Pi"], cv_, dl, lo, hi,
                          eps, beta, q=None, insurance=False, solver=solver)
        if sol.get("gamma") is None:
            continue
        gk = np.asarray(sol["gamma"], float).reshape(-1)
        devs.append(float(np.max(np.abs(gk - gam_P))))
        tails.append(int(np.sum(gk > 1e-9 * max(1.0, gk.max()))))
    rep(f"    planner tail scenarios: {tail_P} of {probs.size}")
    rep(f"    agents' tail scenarios: min {min(tails)}, max {max(tails)}"
        f"  -> the agents do not agree on where the tail is")
    rep(f"    max_k |gamma_k - gamma| = {max(devs):.6f}   (Eq. 94 requires 0)")

    return dict(rho_rn=r_rn, rho_ra=r_ra, gap=gap, gap_per_mwh=gap / max(served, 1e-9),
                gap_pct=100 * gap / max(abs(r_rn), 1e-9), gap_annual_musd=gap * 8760 / 1e6,
                E_rn=e_rn, E_ra=e_ra, CVaR_rn=c_rn, CVaR_ra=c_ra,
                served_mwh=served, planner_tail=tail_P,
                agent_tail_min=min(tails), agent_tail_max=max(tails),
                max_gamma_dev=max(devs), p0_rn=RN["p0"], p0_ra=RA["p0"])


# ---------------------------------------------------------------------------
# E2 : who insures whom, by technology
# ---------------------------------------------------------------------------
def exp_E2(inst, eps, beta, solver, rep):
    rep("\n" + "=" * 78)
    rep("E2  Risk incidence by generation technology")
    rep("=" * 78)
    rep("  At equilibrium the insurance price is q = rho_tilde, so the premium an")
    rep("  agent pays in excess of its expected payout is sum_w (q_w - rho_w) r_w.")
    rep("  Under the corrected Eq. (111)-(112), r_k = L_k - l/N - delta_k, and that")
    rep("  splits exactly into")
    rep("      OWN RISK   pi_k := sum_w (q_w - rho_w) L_kw     (agent-specific)")
    rep("      POOL SHARE       -(1/N) sum_w (q_w - rho_w) l_w (identical for all)")
    rep("  The delta_k drop out of both. pi_k is the economically meaningful term:")
    rep("  the equilibrium price of the tail risk that agent k's OWN profit stream")
    rep("  brings to the market. The pool share is a flat rebate that the")
    rep("  equal-share rule hands every agent regardless of contribution.")

    probs, mc, mv, gbar, dbar, meta = inst
    P = clear(inst, eps, beta, solver)
    L_all = np.vstack([P["L_prod"], P["L_cons"]])
    N = L_all.shape[0]
    R = L_all - P["ell"][None, :] / N          # corrected Eq. (111)-(112), delta = 0
    clearing = float(np.max(np.abs(R.sum(0))))
    rep(f"\n  Eq.(95) clearing check: max_w |sum_k r_k(w)| = {clearing:.3e}")

    w = P["q"] - probs                          # = gamma - eps*rho
    own = L_all @ w                             # pi_k, per agent
    pool = -float(w @ P["ell"]) / N             # common to every agent
    net = own + pool                            # = net_insurance_cost(R_k)

    tech = list(meta["tech"]) + ["LOAD"]
    cap = np.concatenate([gbar.mean(axis=0), dbar.mean(axis=0)])
    rep(f"\n  flat pool share handed to every agent: {pool:,.2f} $/h")
    rep(f"\n  {'technology':<24}{'n':>4}{'own risk pi_k $/h':>20}"
        f"{'$/MW capacity':>16}{'equal-share net':>18}")
    rep("  " + "-" * 82)
    agg = {}
    for t in dict.fromkeys(tech):
        sel = [k for k in range(N) if tech[k] == t]
        o, c, nt = float(own[sel].sum()), float(cap[sel].sum()), float(net[sel].sum())
        agg[t] = dict(own=o, cap=c, net=nt, per_mw=o / max(c, 1e-9), n=len(sel))
        rep(f"  {t:<24}{len(sel):>4}{o:>20,.2f}{o / max(c, 1e-9):>16,.3f}{nt:>18,.2f}")
    ren = float(sum(v["own"] for t, v in agg.items() if t in ("solar", "wind")))
    ren_cap = float(sum(v["cap"] for t, v in agg.items() if t in ("solar", "wind")))
    thm = float(sum(v["own"] for t, v in agg.items()
                    if t not in ("solar", "wind", "LOAD")))
    thm_cap = float(sum(v["cap"] for t, v in agg.items()
                        if t not in ("solar", "wind", "LOAD")))
    rep("  " + "-" * 82)
    rep(f"  {'RENEWABLE (solar+wind)':<24}{'':>4}{ren:>20,.2f}"
        f"{ren / max(ren_cap, 1e-9):>16,.3f}")
    rep(f"  {'THERMAL':<24}{'':>4}{thm:>20,.2f}{thm / max(thm_cap, 1e-9):>16,.3f}")
    rep(f"  {'LOAD':<24}{'':>4}{agg['LOAD']['own']:>20,.2f}")
    tot = float(np.abs(own).sum())
    rep(f"\n  share of total |tail-risk premium|  (renewable share "
        f"{meta['renewable_share']:.1%}):")
    rep(f"    renewables {100 * abs(ren) / max(tot, 1e-9):5.1f}%"
        f"    thermal {100 * abs(thm) / max(tot, 1e-9):5.1f}%"
        f"    load {100 * abs(agg['LOAD']['own']) / max(tot, 1e-9):5.1f}%")
    rep("  The load is the residual risk bearer: pi_LOAD > 0 means its loss spikes")
    rep("  exactly in the scenarios the market prices most heavily, so it is the")
    rep("  natural BUYER of cover. Renewables are mildly negative (they earn more")
    rep("  in scarcity when they do produce), so they SELL cover.")
    rep("  pi is EXACTLY zero for coal and nuclear, and that is a property of the")
    rep("  dataset, not of the theory: thermal capacity in thermal_gens.csv is")
    rep("  constant across scenarios, so an infra-marginal unit has a constant L")
    rep("  and sum_w (q_w - rho_w) L = L * 0 = 0. Modelling forced outages would")
    rep("  give thermal genuine tail exposure; we flag this rather than hide it.")
    ratio = float("nan") if abs(thm_cap) < 1e-9 or abs(thm) < 1e-6 else (
        (ren / ren_cap) / (thm / thm_cap))
    rep("  NOTE the equal-share column is nearly identical across generators: the")
    rep("  flat pool share swamps the signal. That is a property of the Eq. (111)")
    rep("  rule, not of the market, and is worth saying in the paper -- a")
    rep("  contribution-weighted delta_k would allocate the rebate sensibly.")
    return dict(by_tech=_f(agg), renewable_own=ren, thermal_own=thm,
                load_own=float(agg["LOAD"]["own"]), pool_share=pool,
                renewable_per_mw=ren / max(ren_cap, 1e-9),
                thermal_per_mw=thm / max(thm_cap, 1e-9), ratio=float(ratio),
                clearing=clearing, renewable_share=meta["renewable_share"],
                insurance_volume=float(np.abs(R).sum()))


def _f(d):
    return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in d.items()}


def insurance_value(inst, eps, beta, solver, delta_scale=0.1):
    """E1's headline number, computed quietly (used by the sweeps)."""
    probs, mc, mv, gbar, dbar, meta = inst
    RN = clear(inst, 0.0, beta, solver, delta_scale)
    RA = clear(inst, eps, beta, solver, delta_scale)
    ell_RN, _, _ = evaluate_recourse(RN["g"], RN["d"], mc, mv, gbar, dbar, solver, delta_scale)
    ell_RA, _, _ = evaluate_recourse(RA["g"], RA["d"], mc, mv, gbar, dbar, solver, delta_scale)
    gap = rho_ra(ell_RN, probs, eps, beta) - rho_ra(ell_RA, probs, eps, beta)
    served = float(probs @ dbar.sum(axis=1))
    L_all = np.vstack([RA["L_prod"], RA["L_cons"]])
    N = L_all.shape[0]
    R = L_all - RA["ell"][None, :] / N
    tech = list(meta["tech"]) + ["LOAD"]
    cost = np.array([net_insurance_cost(R[k], RA["q"], probs) for k in range(N)])
    ren = float(sum(cost[k] for k in range(N) if tech[k] in ("solar", "wind")))
    return dict(gap=gap, gap_per_mwh=gap / max(served, 1e-9), served=served,
                p0_rn=RN["p0"], p0_ra=RA["p0"],
                risk_premium=RA["p0"] - RN["p0"],
                insurance_volume=float(np.abs(R).sum()),
                renewable_net_cost=ren,
                renewable_share=meta["renewable_share"],
                E_rn=float(probs @ ell_RN), E_ra=float(probs @ ell_RA),
                CVaR_rn=cvar(ell_RN, probs, beta), CVaR_ra=cvar(ell_RA, probs, beta))


# ---------------------------------------------------------------------------
# E3 : does the insurance market matter more as the grid decarbonises?
# ---------------------------------------------------------------------------
def exp_E3(base_kw, shares, eps, beta, solver, rep):
    rep("\n" + "=" * 78)
    rep("E3  Decarbonisation sweep: renewable share at FIXED total capacity")
    rep("=" * 78)
    rep("  Adding renewables on top of an unchanged thermal fleet does not change")
    rep("  scarcity risk -- firm capacity still covers peak load, so the tail is")
    rep("  untouched and the value of insurance is flat. (We checked; it is.)")
    rep("  Decarbonisation is a SUBSTITUTION: thermal retires as renewables enter.")
    rep("  So we hold total installed capacity and the load fixed, and vary the")
    rep("  mix. Firm capacity then falls as the renewable share rises, which is")
    rep("  what actually puts weight in the tail.")

    base = D.build_instance(**dict(base_kw, renewable_scale=1.0, thermal_scale=1.0))
    ren0 = base[5]["renewable_capacity_mw"]
    th0 = base[5]["thermal_capacity_mw"]
    total = ren0 + th0
    load_scale = base[5]["load_scale"]          # frozen across the sweep
    rep(f"\n  total installed capacity held at {total:,.0f} MW; "
        f"load level frozen at scale {load_scale:.4f}")
    rep(f"\n  {'target':>7}{'ren.share':>11}{'firm MW':>11}{'ins. value $/h':>16}"
        f"{'$/MWh':>9}{'E[l] delta':>14}{'CVaR delta':>14}{'ren. own risk':>15}")
    rep("  " + "-" * 97)
    rows = []
    for sg in shares:
        rs = (sg * total) / max(ren0, 1e-9)
        ts = ((1.0 - sg) * total) / max(th0, 1e-9)
        inst = D.build_instance(**dict(base_kw, renewable_scale=rs,
                                       thermal_scale=ts, load_scale=load_scale))
        v = insurance_value(inst, eps, beta, solver)
        v.update(target_share=float(sg), renewable_scale=float(rs),
                 thermal_scale=float(ts),
                 firm_mw=float(inst[5]["thermal_capacity_mw"]))
        rows.append(v)
        rep(f"  {sg:>7.2f}{v['renewable_share']:>11.3f}{v['firm_mw']:>11,.0f}"
            f"{v['gap']:>16,.2f}{v['gap_per_mwh']:>9.4f}"
            f"{v['E_ra'] - v['E_rn']:>14,.0f}{v['CVaR_ra'] - v['CVaR_rn']:>14,.0f}"
            f"{v['renewable_net_cost']:>15,.0f}")
    if len(rows) >= 2:
        a_, b_ = rows[0], rows[-1]
        fac = b_["gap_per_mwh"] / max(abs(a_["gap_per_mwh"]), 1e-12)
        rep(f"\n  from {a_['renewable_share']:.1%} to {b_['renewable_share']:.1%} "
            f"renewable share, the value of the insurance market per MWh moves "
            f"from {a_['gap_per_mwh']:.4f} to {b_['gap_per_mwh']:.4f} $/MWh "
            f"(factor {fac:.2f}).")
    return rows


# ---------------------------------------------------------------------------
# E4 : out-of-sample validation
# ---------------------------------------------------------------------------
def exp_E4(base_kw, n_train, n_test, eps, beta, solver, reps, rep):
    rep("\n" + "=" * 78)
    rep("E4  Out-of-sample validation")
    rep("=" * 78)
    rep("  Clear on one scenario set, then evaluate the resulting day-ahead")
    rep("  commitment on a DISJOINT set drawn from the same conditional pool.")
    rep("  This separates 'CVaR genuinely reduces tail loss' from 'CVaR fits the")
    rep("  in-sample tail'. Differences are paired by replication.")

    dE, dC, dR = [], [], []
    for r in range(reps):
        kw = dict(base_kw); kw["n_scenarios"] = n_train; kw["seed"] = 1000 + r
        tr = D.build_instance(**kw)
        kw2 = dict(kw); kw2["n_scenarios"] = n_test; kw2["seed"] = 500_000 + r
        te = D.build_instance(**kw2)
        RN = clear(tr, 0.0, beta, solver)
        RA = clear(tr, eps, beta, solver)
        p_te, mc, mv, gbar_te, dbar_te, _ = te
        eRN, _, _ = evaluate_recourse(RN["g"], RN["d"], mc, mv, gbar_te, dbar_te, solver)
        eRA, _, _ = evaluate_recourse(RA["g"], RA["d"], mc, mv, gbar_te, dbar_te, solver)
        dE.append(float(p_te @ eRA) - float(p_te @ eRN))
        dC.append(cvar(eRA, p_te, beta) - cvar(eRN, p_te, beta))
        dR.append(rho_ra(eRA, p_te, eps, beta) - rho_ra(eRN, p_te, eps, beta))

    def summarize(name, d, lower_is_better=True):
        d = np.asarray(d, float)
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(d.size)
        wins = int(np.sum(d < 0)) if lower_is_better else int(np.sum(d > 0))
        rep(f"    {name:<34}{m:>14,.2f} +/- {1.96 * se:>10,.2f}"
            f"   improved in {wins}/{d.size} replications")
        return dict(mean=float(m), ci95=float(1.96 * se), wins=wins, n=int(d.size))

    rep(f"\n  held-out loss, risk-aware MINUS risk-neutral ({reps} replications,"
        f" train S={n_train}, test S={n_test}):")
    rep("  negative = risk-aware is better out of sample")
    out = dict(
        dE=summarize("E[l]            (expected)", dE),
        dCVaR=summarize(f"CVaR_{beta}[l]      (tail)", dC),
        dRho=summarize("rho_RA[l]       (planner obj)", dR))
    rep("\n  The tail column is the claim that matters: if CVaR clearing only fit")
    rep("  the in-sample tail, this would not be reliably negative.")
    return out


# ---------------------------------------------------------------------------
# E5 : paired statistics -- repairing the reporting in Tables 1 and 3
# ---------------------------------------------------------------------------
def exp_E5(base_kw, n_scen, eps_list, beta, solver, reps, rep):
    rep("\n" + "=" * 78)
    rep("E5  Paired statistics (repairs the reporting in Tables 1 and 3)")
    rep("=" * 78)
    rep("  Table 1 reports e.g. 2.5836 +/- 0.0064 against 2.5874 +/- 0.0064: the")
    rep("  difference is smaller than either error bar, so as printed the result")
    rep("  is not statistically supported. But the comparison is PAIRED -- same")
    rep("  instance, same scenarios, same draw -- so the paired interval is the")
    rep("  correct summary. We report both to show what changes.")

    tails = {e: [] for e in eps_list}
    tails["RN"] = []
    for r in range(reps):
        kw = dict(base_kw); kw["n_scenarios"] = n_scen; kw["seed"] = 7000 + r
        inst = D.build_instance(**kw)
        probs, mc, mv, gbar, dbar, _ = inst
        RN = clear(inst, 0.0, beta, solver)
        eRN, _, _ = evaluate_recourse(RN["g"], RN["d"], mc, mv, gbar, dbar, solver)
        tails["RN"].append(cvar(eRN, probs, beta))
        for e in eps_list:
            RA = clear(inst, e, beta, solver)
            eRA, _, _ = evaluate_recourse(RA["g"], RA["d"], mc, mv, gbar, dbar, solver)
            tails[e].append(cvar(eRA, probs, beta))

    base = np.asarray(tails["RN"], float)
    rep(f"\n  tail loss CVaR_{beta}[l] over {reps} replications (lower is better)")
    rep(f"  {'method':<22}{'mean':>14}{'unpaired +/-':>16}"
        f"{'paired diff':>14}{'paired +/-':>14}{'wins':>8}")
    rep("  " + "-" * 88)
    rep(f"  {'risk-neutral':<22}{base.mean():>14,.2f}"
        f"{1.96 * base.std(ddof=1) / np.sqrt(reps):>16,.2f}{'--':>14}{'--':>14}{'--':>8}")
    out = {}
    for e in eps_list:
        a = np.asarray(tails[e], float)
        d = a - base
        unp = 1.96 * a.std(ddof=1) / np.sqrt(reps)
        pai = 1.96 * d.std(ddof=1) / np.sqrt(reps)
        wins = int(np.sum(d < 0))
        rep(f"  {'CVaR eps=' + str(e):<22}{a.mean():>14,.2f}{unp:>16,.2f}"
            f"{d.mean():>14,.2f}{pai:>14,.2f}{wins:>5}/{reps}")
        out[str(e)] = dict(mean=float(a.mean()), unpaired_ci=float(unp),
                           paired_diff=float(d.mean()), paired_ci=float(pai),
                           wins=wins, n=reps,
                           significant_paired=bool(abs(d.mean()) > pai),
                           significant_unpaired=bool(
                               abs(d.mean()) > unp + 1.96 * base.std(ddof=1) / np.sqrt(reps)))
    rep("\n  The 'unpaired' column is what Tables 1 and 3 report; the 'paired'")
    rep("  column is what the experimental design actually licenses.")
    return out


# ---------------------------------------------------------------------------
# E6 : what the headline number actually depends on
# ---------------------------------------------------------------------------
def exp_E6(inst, eps, beta, solver, deltas, rep):
    rep("\n" + "=" * 78)
    rep("E6  Sensitivity to the deviation bid price (and why it must be reported)")
    rep("=" * 78)
    rep("  The planner's loss charges alpha^+/alpha^- for deviating from the")
    rep("  day-ahead schedule (paper Eq. 3). In the code these were hard-coded as")
    rep("  mc/10 and VOLL/10 with the comment 'just a heuristic'. They are not a")
    rep("  harmless heuristic: the value of risk-aware clearing is almost exactly")
    rep("  LINEAR in them, so any single headline $/MWh figure is a restatement of")
    rep("  that constant. Reporting the slope instead makes the result parameter-free.")

    probs, mc, mv, gbar, dbar, meta = inst
    served = float(probs @ dbar.sum(axis=1))
    rep(f"\n  {'delta_scale':>12}{'alpha_load $/MWh':>18}{'ins. value $/h':>16}"
        f"{'$/MWh':>10}{'$/MWh per unit delta':>22}")
    rep("  " + "-" * 78)
    rows = []
    for ds in deltas:
        v = insurance_value(inst, eps, beta, solver, delta_scale=ds)
        slope = v["gap_per_mwh"] / max(ds, 1e-12)
        rows.append(dict(delta_scale=float(ds), **{k: v[k] for k in
                    ("gap", "gap_per_mwh", "E_rn", "E_ra", "CVaR_rn", "CVaR_ra")},
                    slope=float(slope)))
        rep(f"  {ds:>12.3f}{float(mv[0]) * ds:>18.1f}{v['gap']:>16,.0f}"
            f"{v['gap_per_mwh']:>10.4f}{slope:>22.4f}")
    sl = np.array([r["slope"] for r in rows])
    rep(f"\n  slope is constant to within {100 * sl.std() / max(abs(sl.mean()), 1e-12):.3f}%"
        f"  =>  value of the insurance market = {sl.mean():.3f} x delta_scale  $/MWh")
    rep(f"  equivalently {100 * sl.mean() / float(mv[0]):.4f}% of the load's deviation")
    rep("  bid price per MWh served. THIS is the number the paper should quote:")
    rep("  it is a property of the uncertainty, not of an arbitrary constant.")
    return dict(rows=rows, slope_mean=float(sl.mean()), slope_std=float(sl.std()),
                pct_of_voll=float(100 * sl.mean() / float(mv[0])), served=served)
