#!/usr/bin/env python
"""Decentralisation and insurance-transfer experiments (paper Sections 8.3-8.7).

Experiments
-----------
D1  Decentralisation holds with insurance.   Theorems 6, 7, 8
D2  Decentralisation fails without it.       Section 8.2.2, Eq. (94)
D3  Anatomy of the equilibrium transfers.    Remarks 7, 8, 9
D4  Heterogeneous confidence levels.         Theorem 9, Eq. (121)-(123)
D5  Transfers as the risk weight varies.     Remark 10

Usage
-----
    python run_decentralized_experiments.py --experiments D1 D2 D3 D4 D5 \
        --scenarios 500 --eps 0.1 --beta 0.95 --seed 2025 \
        --solver GUROBI --out outputs/results --tag paper
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import cvxpy as cp
from jax import random

from zavala_funcs import zavala_cvar, generate_instance
from zavala_decentralized import (
    planner_losses, transfers_equal_share, transfers_heterogeneous,
    solve_agent, objective_at, producer_utility, consumer_utility, _var_level,
)

TOL = 1e-5


class Reporter:
    def __init__(self, logfile=None):
        self.lines, self.logfile, self.results, self.fails = [], logfile, {}, 0

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))
        if self.logfile:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")

    def check(self, label, ok, detail=""):
        ok = bool(ok)
        if not ok:
            self.fails += 1
        self(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        return ok


def build_planner(args, rep):
    """Solve the risk-aware planner problem and return everything downstream needs."""
    key = random.key(args.seed)
    probs, mc, mv, gbar, dbar = generate_instance(
        key, num_scenarios=args.scenarios, num_g=10, num_d=10, r=args.seed,
        input_scenario=args.scenario,
    )
    probs = np.asarray(probs, float).reshape(-1)
    mc = np.asarray(mc, float).reshape(-1)
    mv = np.asarray(mv, float).reshape(-1)
    gbar = np.asarray(gbar, float)
    dbar = np.asarray(dbar, float)

    rep(f"Planner: S={probs.size}, producers={mc.size}, consumers={mv.size}, "
        f"eps={args.eps}, beta={args.beta}, scenario={args.scenario}, seed={args.seed}")

    g, d, G, D, pi, Pi, gamma = zavala_cvar(
        probs, mc, mv, gbar, dbar, beta=args.beta, lambda_cvar=args.eps,
        check_invariants=True, solver=args.solver_obj, verbose=False,
    )
    g = np.asarray(g, float).reshape(-1)
    d = np.asarray(d, float).reshape(-1)
    G = np.asarray(G, float)
    D = np.asarray(D, float)
    Pi = np.asarray(Pi, float).reshape(-1)
    gamma = np.asarray(gamma, float).reshape(-1)
    rho_tilde = (1.0 - args.eps) * probs + gamma          # Eq. (67)
    q = rho_tilde.copy()                                   # Eq. (107): q_w = rho_tilde_w
    p0 = float(pi)

    rep(f"  p0 = {p0:.6f},  E_rho[Pi] = {float(probs @ Pi):.6f},  "
        f"E_rhotilde[Pi] = {float(rho_tilde @ Pi):.6f}")
    rep(f"  tail scenarios with gamma_w > 0: {int((gamma > 1e-9).sum())} of {probs.size} "
        f"(expected ~(1-beta)S = {(1-args.beta)*probs.size:.0f})")

    # The paper's feasible set has x, y >= 0 (Eq. 12/18/65); the repo leaves the
    # day-ahead variables free. Report any violation rather than hiding it.
    neg_g, neg_d = int((g < -1e-6).sum()), int((d < -1e-6).sum())
    if neg_g or neg_d:
        rep(f"  [WARN] planner day-ahead allocation violates x,y >= 0 "
            f"({neg_g} producers, {neg_d} consumers negative). The repo's LP leaves "
            f"day-ahead variables free; the paper's Eq. (12)/(18) does not.")

    L_prod, L_cons, ell = planner_losses(p0, Pi, mc, mv, g, d, G, D)

    # Eq. (110) gate: per-agent losses must sum to the aggregate planner loss.
    agg = L_prod.sum(axis=0) + L_cons.sum(axis=0)
    rep.check("Eq.(110)  sum_k L_k(w) = ell(w)  (price terms cancel)",
              np.max(np.abs(agg - ell)) < 1e-4 * max(1.0, float(np.max(np.abs(ell)))),
              f"max|diff| = {float(np.max(np.abs(agg - ell))):.3e}")

    return dict(probs=probs, mc=mc, mv=mv, gbar=gbar, dbar=dbar, g=g, d=d, G=G, D=D,
                p0=p0, Pi=Pi, gamma=gamma, rho_tilde=rho_tilde, q=q,
                L_prod=L_prod, L_cons=L_cons, ell=ell,
                mc_delta=mc / 10.0, mv_delta=mv / 10.0)


def _agent_specs(P):
    """Yield (kind, idx, cost_or_value, delta, cap_lo, cap_hi, da, rt) per agent."""
    S = P["probs"].size
    for j in range(P["mc"].size):
        yield ("producer", j, P["mc"][j], P["mc_delta"][j],
               np.zeros(S), P["gbar"][:, j], P["g"][j], P["G"][:, j])
    for i in range(P["mv"].size):
        yield ("consumer", i, P["mv"][i], P["mv_delta"][i],
               np.zeros(S), P["dbar"][:, i], P["d"][i], P["D"][:, i])


# ---------------------------------------------------------------------------
# D1 : decentralisation with insurance
# ---------------------------------------------------------------------------

def exp_D1(P, args, rep):
    rep("\n" + "=" * 78)
    rep("D1  Decentralisation with insurance  (Theorems 6, 7, 8)")
    rep("=" * 78)
    rep("Definition 3 (UM) requires (x*, X*, r*) to solve the agent's problem, with")
    rep("insurance a CHOICE variable. So the test is: let each agent optimise freely")
    rep("over physical position AND insurance at prices q, then check its optimal")
    rep("objective equals the objective at (planner allocation, constructed r).")
    rep("Degenerate real-time optima make raw allocation distance a poor criterion,")
    rep("so the objective value is the primary test and allocations are diagnostic.")

    r_prod, r_cons = transfers_equal_share(P["L_prod"], P["L_cons"], P["ell"])
    rep.check("Eq.(95)  insurance market clears state-by-state",
              np.max(np.abs(r_prod.sum(axis=0) + r_cons.sum(axis=0))) < 1e-6
              * max(1.0, float(np.max(np.abs(P["ell"])))),
              f"max|sum_k r_k(w)| = {float(np.max(np.abs(r_prod.sum(0) + r_cons.sum(0)))):.3e}")

    rows, obj_gaps, da_gaps, gamma_gaps = [], [], [], []
    for kind, idx, cv, dl, lo, hi, da, rt in _agent_specs(P):
        r_k = (r_prod[idx] if kind == "producer" else r_cons[idx])
        free = solve_agent(kind, P["probs"], P["p0"], P["Pi"], cv, dl, lo, hi,
                           args.eps, args.beta, q=P["q"], r_fixed=None,
                           solver=args.solver_obj)
        if free["status"] not in ("optimal", "optimal_inaccurate"):
            rep(f"  [{kind} {idx}] solver status: {free['status']}")
            rows.append(dict(kind=kind, idx=idx, status=free["status"]))
            continue
        base = objective_at(kind, P["probs"], P["p0"], P["Pi"], cv, dl,
                            args.eps, args.beta, da, rt, q=P["q"], r=r_k)
        scale = max(1.0, abs(free["objective"]))
        obj_gap = (free["objective"] - base) / scale
        obj_gaps.append(obj_gap)
        da_gaps.append(abs(free["da"] - da))
        g_gap = np.nan
        if free["gamma"] is not None:
            g_gap = float(np.max(np.abs(free["gamma"] - P["gamma"])))
            gamma_gaps.append(g_gap)
        rows.append(dict(kind=kind, idx=idx, status=free["status"],
                         obj_free=free["objective"], obj_at_planner=base,
                         rel_obj_gap=obj_gap, da_gap=abs(free["da"] - da),
                         max_gamma_dev=g_gap))

    if obj_gaps:
        rep.check("(UM) planner allocation + constructed transfer is agent-optimal",
                  max(obj_gaps) < 1e-5,
                  f"max relative objective gap = {max(obj_gaps):.3e}")
        # Diagnostic only: once the objectives tie, any tied allocation is an
        # equally valid agent optimum, so a nonzero distance is expected.
        rep(f"  [info] max|DA diff| vs planner = {max(da_gaps):.3e} "
            f"(alternative optima; (UM) is settled by the objective test above)")
    if gamma_gaps:
        rep.check("Eq.(94)/(109)  agents' CVaR multipliers equal the planner's",
                  max(gamma_gaps) < 1e-5,
                  f"max|gamma_k - gamma| = {max(gamma_gaps):.3e}")

    # Document the sign discrepancy in Eq. (111)-(112) as printed.
    rep("\n  Control: the same test with Eq. (111)-(112) exactly as printed")
    rp_pr, rc_pr = transfers_equal_share(P["L_prod"], P["L_cons"], P["ell"],
                                         as_printed=True)
    printed_gaps = []
    for kind, idx, cv, dl, lo, hi, da, rt in _agent_specs(P):
        r_k = (rp_pr[idx] if kind == "producer" else rc_pr[idx])
        free = solve_agent(kind, P["probs"], P["p0"], P["Pi"], cv, dl, lo, hi,
                           args.eps, args.beta, q=P["q"], r_fixed=None,
                           solver=args.solver_obj)
        if free["status"] not in ("optimal", "optimal_inaccurate"):
            continue
        base = objective_at(kind, P["probs"], P["p0"], P["Pi"], cv, dl,
                            args.eps, args.beta, da, rt, q=P["q"], r=r_k)
        printed_gaps.append((free["objective"] - base) / max(1.0, abs(free["objective"])))
    if printed_gaps:
        rep(f"  max relative (UM) gap with the printed sign = {max(printed_gaps):.3e}")
        rep("  (a large gap here is the evidence that Eq. 111-112 need r -> -r;")
        rep("   Eq. 122 in Theorem 9 already uses the corrected sign)")

    return dict(rows=rows,
                max_um_gap=(max(obj_gaps) if obj_gaps else None),
                max_um_gap_printed_sign=(max(printed_gaps) if printed_gaps else None),
                r_prod=r_prod.tolist() if args.dump_transfers else None,
                r_cons=r_cons.tolist() if args.dump_transfers else None)


# ---------------------------------------------------------------------------
# D2 : decentralisation without insurance
# ---------------------------------------------------------------------------

def exp_D2(P, args, rep):
    rep("\n" + "=" * 78)
    rep("D2  Decentralisation WITHOUT insurance  (Section 8.2.2, Eq. 94)")
    rep("=" * 78)
    rep("Same prices, no insurance market. Eq. (94) requires every agent's CVaR")
    rep("multiplier to equal the planner's; with heterogeneous exposures it cannot.")
    rep("We measure the resulting dispersion and the allocation error.")

    rows, dev_alloc, dev_gamma = [], [], []
    for kind, idx, cv, dl, lo, hi, da, rt in _agent_specs(P):
        sol = solve_agent(kind, P["probs"], P["p0"], P["Pi"], cv, dl, lo, hi,
                          args.eps, args.beta, insurance=False, solver=args.solver_obj)
        if sol["status"] not in ("optimal", "optimal_inaccurate"):
            rep(f"  [{kind} {idx}] status: {sol['status']}")
            rows.append(dict(kind=kind, idx=idx, status=sol["status"]))
            continue
        v_planner = objective_at(kind, P["probs"], P["p0"], P["Pi"], cv, dl,
                                 args.eps, args.beta, da, rt, insurance=False)
        alloc_gap = max(abs(sol["da"] - da), float(np.max(np.abs(sol["rt"] - rt))))
        g_dev = (np.nan if sol["gamma"] is None
                 else float(np.max(np.abs(sol["gamma"] - P["gamma"]))))
        dev_alloc.append(alloc_gap)
        if not np.isnan(g_dev):
            dev_gamma.append(g_dev)
        rows.append(dict(kind=kind, idx=idx, status=sol["status"],
                         obj_agent=sol["objective"], obj_at_planner=v_planner,
                         alloc_gap=alloc_gap, max_gamma_dev=g_dev,
                         tail_count=(None if sol["gamma"] is None
                                     else int((sol["gamma"] > 1e-9).sum()))))

    if dev_gamma:
        rep(f"  max_k max_w |gamma_k(w) - gamma(w)| = {max(dev_gamma):.6e}")
        rep(f"  mean over agents                    = {float(np.mean(dev_gamma)):.6e}")
        rep(f"  >> Eq. (94) {'HOLDS' if max(dev_gamma) < 1e-6 else 'FAILS'} without insurance")
    if dev_alloc:
        rep(f"  max_k |allocation_k - planner_k|    = {max(dev_alloc):.6e}")

    tails = [r.get("tail_count") for r in rows if r.get("tail_count") is not None]
    if tails:
        rep(f"  per-agent tail-scenario counts: min={min(tails)}, max={max(tails)}, "
            f"planner={int((P['gamma'] > 1e-9).sum())}")
        rep("  (differing tail sets across agents is precisely the obstruction in 8.2.2)")
    return dict(rows=rows)


# ---------------------------------------------------------------------------
# D3 : anatomy of the transfers
# ---------------------------------------------------------------------------

def exp_D3(P, args, rep):
    rep("\n" + "=" * 78)
    rep("D3  Anatomy of the equilibrium transfers  (Remarks 7, 8, 9)")
    rep("=" * 78)

    r_prod, r_cons = transfers_equal_share(P["L_prod"], P["L_cons"], P["ell"])
    probs, q, ell = P["probs"], P["q"], P["ell"]
    N = P["mc"].size + P["mv"].size

    rep("\n  Eq.(113), sign-corrected: insured loss L_k - r_k must be ell(w)/N + const")
    ins_prod = P["L_prod"] - r_prod
    ins_cons = P["L_cons"] - r_cons
    ins = np.vstack([ins_prod, ins_cons])
    target = ell / N
    slopes, resid = [], []
    for k in range(ins.shape[0]):
        A = np.vstack([target, np.ones_like(target)]).T
        coef, *_ = np.linalg.lstsq(A, ins[k], rcond=None)
        slopes.append(coef[0])
        resid.append(float(np.max(np.abs(ins[k] - (coef[0] * target + coef[1])))))
    rep.check("Eq.(113)  slope = 1 for every agent (common tail set)",
              max(abs(s - 1.0) for s in slopes) < 1e-6,
              f"max|slope - 1| = {max(abs(s - 1.0) for s in slopes):.3e}")
    rep.check("Eq.(113)  insured loss is exactly affine in ell(w)",
              max(resid) < 1e-6 * max(1.0, float(np.max(np.abs(ell)))),
              f"max residual = {max(resid):.3e}")

    rep("\n  Premium decomposition (Remark 8): q = (1-eps)rho + gamma, so")
    rep("  premium = (1-eps) * E_rho[payout] + tail premium gamma . r")
    rep(f"\n  {'agent':>12} {'premium':>14} {'E_rho[payout]':>15} {'tail prem':>13} "
        f"{'net':>12} {'|r|_max':>12}")
    rows = []
    names = ([f"gen_{j}" for j in range(P["mc"].size)]
             + [f"load_{i}" for i in range(P["mv"].size)])
    R = np.vstack([r_prod, r_cons])
    for k, nm in enumerate(names):
        prem = float(q @ R[k])
        epay = float(probs @ R[k])
        tail = float(P["gamma"] @ R[k])
        rows.append(dict(agent=nm, premium=prem, expected_payout=epay,
                         tail_premium=tail, net=epay - prem,
                         max_abs=float(np.max(np.abs(R[k])))))
        rep(f"  {nm:>12} {prem:>14.3f} {epay:>15.3f} {tail:>13.3f} "
            f"{epay - prem:>12.3f} {float(np.max(np.abs(R[k]))):>12.3f}")
    rep(f"  {'TOTAL':>12} {float(q @ R.sum(0)):>14.3e} {float(probs @ R.sum(0)):>15.3e}")
    rep("  (totals are ~0 by Eq. 95. net = E_rho[payout] - premium: NEGATIVE means the")
    rep("   agent pays more than it expects back, i.e. it is a net BUYER of cover;")
    rep("   POSITIVE means it is a net SELLER, compensated for bearing tail risk.)")

    # Who insures whom, by technology (s_real10_mix: gens 0-5 unreliable, 6-9 reliable)
    if P["mc"].size == 10:
        unrel = slice(0, 6)
        rel = slice(6, 10)
        rep(f"\n  By technology: unreliable gens net = "
            f"{float(probs @ r_prod[unrel].sum(0)) - float(q @ r_prod[unrel].sum(0)):+.3f}, "
            f"reliable gens net = "
            f"{float(probs @ r_prod[rel].sum(0)) - float(q @ r_prod[rel].sum(0)):+.3f}, "
            f"loads net = "
            f"{float(probs @ r_cons.sum(0)) - float(q @ r_cons.sum(0)):+.3f}")

    # Risk reduction actually delivered
    rep("\n  Risk reduction: CVaR of each agent's loss, uninsured vs insured")
    L_all = np.vstack([P["L_prod"], P["L_cons"]])
    red = []
    for k, nm in enumerate(names):
        c_un = _cvar(L_all[k], probs, args.beta)
        c_in = _cvar(L_all[k] - R[k], probs, args.beta)
        red.append(dict(agent=nm, cvar_uninsured=c_un, cvar_insured=c_in,
                        reduction=c_un - c_in))
    for r_ in red:
        rep(f"  {r_['agent']:>12}  CVaR {r_['cvar_uninsured']:>14.3f} -> "
            f"{r_['cvar_insured']:>14.3f}   ({r_['reduction']:+.3f})")

    return dict(premiums=rows, risk_reduction=red, slopes=slopes)


def _cvar(loss, probs, beta):
    t = _var_level(loss, probs, beta)
    return float(t + (1.0 / (1.0 - beta)) * (np.asarray(probs, float)
                                             @ np.maximum(np.asarray(loss, float) - t, 0.0)))


# ---------------------------------------------------------------------------
# D4 : heterogeneous confidence levels
# ---------------------------------------------------------------------------

def exp_D4(P, args, rep):
    rep("\n" + "=" * 78)
    rep("D4  Heterogeneous confidence levels  (Theorem 9, Eq. 121-123)")
    rep("=" * 78)
    rep("Agents get different beta_k; the planner must use beta_min (Eq. 117-120).")
    rep("Agents in S_> (beta_k > beta_min) need a CONSTANT insured loss across the")
    rep("tail (Eq. 121), which the equal-share construction cannot deliver. This")
    rep("verifies the construction the manuscript flags as not yet proof-checked.")

    m, n = P["mc"].size, P["mv"].size
    rng = np.random.default_rng(args.seed)
    betas_pool = np.array(args.het_betas, dtype=float)
    beta_prod = rng.choice(betas_pool, size=m)
    beta_cons = rng.choice(betas_pool, size=n)
    beta_prod[0] = betas_pool.min()          # guarantee S_min is non-empty
    beta_min = float(min(beta_prod.min(), beta_cons.min()))
    rep(f"  beta_producers = {np.round(beta_prod, 3).tolist()}")
    rep(f"  beta_consumers = {np.round(beta_cons, 3).tolist()}")
    rep(f"  beta_min = {beta_min}  (planner must clear at this level)")

    if abs(beta_min - args.beta) > 1e-12:
        rep(f"  [NOTE] planner solution in hand used beta={args.beta}; re-solving at beta_min")
        g, d, G, D, pi, Pi, gamma = zavala_cvar(
            P["probs"], P["mc"], P["mv"], P["gbar"], P["dbar"],
            beta=beta_min, lambda_cvar=args.eps, check_invariants=True,
            solver=args.solver_obj, verbose=False)
        Q = dict(P)
        Q.update(g=np.asarray(g, float).reshape(-1), d=np.asarray(d, float).reshape(-1),
                 G=np.asarray(G, float), D=np.asarray(D, float), p0=float(pi),
                 Pi=np.asarray(Pi, float).reshape(-1),
                 gamma=np.asarray(gamma, float).reshape(-1))
        Q["rho_tilde"] = (1 - args.eps) * Q["probs"] + Q["gamma"]
        Q["q"] = Q["rho_tilde"].copy()
        Q["L_prod"], Q["L_cons"], Q["ell"] = planner_losses(
            Q["p0"], Q["Pi"], Q["mc"], Q["mv"], Q["g"], Q["d"], Q["G"], Q["D"])
    else:
        Q = P

    r_prod, r_cons, info = transfers_heterogeneous(
        Q["L_prod"], Q["L_cons"], Q["ell"], beta_prod, beta_cons,
        probs=Q["probs"], beta_min=beta_min)

    rep.check("Eq.(95)  heterogeneous transfers clear state-by-state",
              np.max(np.abs(r_prod.sum(0) + r_cons.sum(0))) < 1e-6
              * max(1.0, float(np.max(np.abs(Q["ell"])))),
              f"max|sum_k r_k| = {float(np.max(np.abs(r_prod.sum(0) + r_cons.sum(0)))):.3e}")

    # Eq. (121): agents in S_> hold a constant insured loss on the tail
    L_all = np.vstack([Q["L_cons"], Q["L_prod"]])
    R_all = np.vstack([r_cons, r_prod])
    ins = L_all - R_all
    tail_idx = np.argsort(Q["ell"])[::-1][:max(1, int(round((1 - beta_min) * Q["probs"].size)))]
    spreads = []
    for k in info["S_gt"]:
        spreads.append(float(ins[k][tail_idx].max() - ins[k][tail_idx].min()))
    if spreads:
        rep.check("Eq.(121)  S_> agents have constant insured loss across the tail",
                  max(spreads) < 1e-6 * max(1.0, float(np.max(np.abs(Q["ell"])))),
                  f"max spread over tail = {max(spreads):.3e}  ({len(info['S_gt'])} agents in S_>)")
    else:
        rep("  (all agents share beta_min; S_> is empty for this draw)")

    # contrast: the equal-share construction must violate Eq. (121) for S_> agents
    r_p_eq, r_c_eq = transfers_equal_share(Q["L_prod"], Q["L_cons"], Q["ell"])
    ins_eq = np.vstack([Q["L_cons"], Q["L_prod"]]) - np.vstack([r_c_eq, r_p_eq])
    eq_spreads = [float(ins_eq[k][tail_idx].max() - ins_eq[k][tail_idx].min())
                  for k in info["S_gt"]]
    if eq_spreads:
        rep(f"  contrast: equal-share construction gives S_> agents a tail spread of "
            f"{max(eq_spreads):.3e} (non-constant => violates Eq. 121, as Theorem 9 predicts)")

    rep(f"  risk-bearing weights w_k: {np.round(info['w'], 4).tolist()}")
    rep(f"  |S_min| = {info['n_min']}, agents in S_>: {info['S_gt'].tolist()}")
    return dict(beta_prod=beta_prod.tolist(), beta_cons=beta_cons.tolist(),
                beta_min=beta_min, w=info["w"].tolist(),
                S_gt=info["S_gt"].tolist(),
                tail_spread_hetero=(max(spreads) if spreads else None),
                tail_spread_equalshare=(max(eq_spreads) if eq_spreads else None))


# ---------------------------------------------------------------------------
# D5 : transfers across the risk weight
# ---------------------------------------------------------------------------

def exp_D5(P, args, rep):
    rep("\n" + "=" * 78)
    rep("D5  Transfers as the risk weight varies  (Remark 10)")
    rep("=" * 78)
    rep("At eps = 0 insurance is priced at q = rho, so premium = expected payout and")
    rep("every agent is indifferent: the market is redundant. As eps grows, q tilts")
    rep("toward the tail and the premium separates from the expected payout.")
    rep(f"\n  {'eps':>6} {'sum|gamma|':>12} {'max tail prem':>15} "
        f"{'mean|r|':>12} {'E[Pi]':>12} {'p0':>12}")

    rows = []
    for eps in args.eps_sweep:
        g, d, G, D, pi, Pi, gamma = zavala_cvar(
            P["probs"], P["mc"], P["mv"], P["gbar"], P["dbar"],
            beta=args.beta, lambda_cvar=eps, check_invariants=True,
            solver=args.solver_obj, verbose=False)
        g = np.asarray(g, float).reshape(-1); d = np.asarray(d, float).reshape(-1)
        G = np.asarray(G, float); D = np.asarray(D, float)
        Pi = np.asarray(Pi, float).reshape(-1); gamma = np.asarray(gamma, float).reshape(-1)
        q = (1 - eps) * P["probs"] + gamma
        Lp, Lc, ell = planner_losses(float(pi), Pi, P["mc"], P["mv"], g, d, G, D)
        rp, rc = transfers_equal_share(Lp, Lc, ell)
        R = np.vstack([rp, rc])
        tail_prem = [float(gamma @ R[k]) for k in range(R.shape[0])]
        row = dict(eps=eps, sum_abs_gamma=float(np.abs(gamma).sum()),
                   max_tail_premium=float(np.max(np.abs(tail_prem))),
                   mean_abs_r=float(np.mean(np.abs(R))),
                   E_Pi=float(P["probs"] @ Pi), p0=float(pi))
        rows.append(row)
        rep(f"  {eps:>6.2f} {row['sum_abs_gamma']:>12.6f} {row['max_tail_premium']:>15.3f} "
            f"{row['mean_abs_r']:>12.3f} {row['E_Pi']:>12.3f} {row['p0']:>12.3f}")

    zero = [r for r in rows if abs(r["eps"]) < 1e-12]
    if zero:
        # Scale-relative: transfers are O(1e4), so an absolute threshold would
        # only be measuring solver tolerance on gamma.
        rel = zero[0]["max_tail_premium"] / max(1.0, zero[0]["mean_abs_r"])
        rep.check("Remark 10  eps=0 makes insurance redundant (tail premium = 0)",
                  rel < 1e-6,
                  f"max tail premium / mean|r| at eps=0 = {rel:.3e} "
                  f"(absolute {zero[0]['max_tail_premium']:.3e})")
    return dict(rows=rows)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["D1", "D2", "D3", "D4", "D5"],
                    choices=["D1", "D2", "D3", "D4", "D5"])
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.1, help="risk weight (paper eps_cvar)")
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--scenario", default="s_real10_mix")
    ap.add_argument("--solver", default="GUROBI")
    ap.add_argument("--eps-sweep", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--het-betas", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--tag", default="dec")
    ap.add_argument("--dump-transfers", action="store_true",
                    help="include full transfer matrices in the JSON (large)")
    args = ap.parse_args()
    args.solver_obj = getattr(cp, args.solver)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    logfile = Path("logs") / f"decentralized_{args.tag}.log"
    if logfile.exists():
        logfile.unlink()
    rep = Reporter(str(logfile))

    rep("=" * 78)
    rep("Decentralisation and insurance-transfer experiments")
    rep("=" * 78)
    rep(f"args: {vars({k: v for k, v in vars(args).items() if k != 'solver_obj'}) if False else {k: v for k, v in vars(args).items() if k != 'solver_obj'}}")

    P = build_planner(args, rep)
    results = {}
    table = dict(D1=exp_D1, D2=exp_D2, D3=exp_D3, D4=exp_D4, D5=exp_D5)
    for name in args.experiments:
        results[name] = table[name](P, args, rep)

    rep("\n" + "=" * 78)
    rep(f"{rep.fails} check(s) FAILED" if rep.fails else "ALL CHECKS PASSED")
    rep("=" * 78)

    meta = {k: v for k, v in vars(args).items() if k != "solver_obj"}
    payload = dict(meta=meta, planner=dict(
        p0=P["p0"], E_Pi=float(P["probs"] @ P["Pi"]),
        E_Pi_tilted=float(P["rho_tilde"] @ P["Pi"]),
        n_tail=int((P["gamma"] > 1e-9).sum()),
    ), results=results, n_failed_checks=rep.fails)
    jpath = out / f"decentralized_{args.tag}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    rep(f"\nResults written to {jpath}")
    rep(f"Log written to {logfile}")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
