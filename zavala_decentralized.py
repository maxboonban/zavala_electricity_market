#!/usr/bin/env python
"""Decentralised (agent-level) problems and the insurance market.

Implements Sections 8.3-8.7 of "Risk-Aware Market Clearing in Electricity
Markets": the individual risk-aware agent problems with and without Arrow-Debreu
insurance, the explicit transfer constructions, and R-SCER verification.

Equation numbers below refer to the paper.

Model mapping (paper <-> code)
------------------------------
  producers j in [m]  <->  generators, cost c_j = mc_g_i[j], caps g_i_bar[w][j]
  consumers i in [n]  <->  loads,      value v_i = mv_d_j[i], caps d_j_bar[w][i]
  deviation penalties  alpha^+ = alpha^- = cost/10   (the repo's Delta heuristic)
  eps = lambda_cvar,  rho = probs,  Omega = scenarios

Agent utilities, Eq. (3) and (4), with y^+ - y^- = G_w - g by Observation 1:

  producer j:  u_j(w) = (p0 - c_j) g_j + (p_w - c_j)(G_wj - g_j) - D_j |G_wj - g_j|
  consumer i:  u_i(w) = (v_i - p0) d_i + (v_i - p_w)(D_wi - d_i) - D_i |D_wi - d_i|
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# Utilities evaluated at a given allocation
# ---------------------------------------------------------------------------

def producer_utility(p0, p_w, c, delta, g_da, G_rt):
    """u_j(w) for one producer, Eq. (4). G_rt shape (S,). Returns shape (S,)."""
    p_w = np.asarray(p_w, float).reshape(-1)
    G_rt = np.asarray(G_rt, float).reshape(-1)
    dev = G_rt - float(g_da)
    return (float(p0) - c) * float(g_da) + (p_w - c) * dev - delta * np.abs(dev)


def consumer_utility(p0, p_w, v, delta, d_da, D_rt):
    """u_i(w) for one consumer, Eq. (3). Returns shape (S,)."""
    p_w = np.asarray(p_w, float).reshape(-1)
    D_rt = np.asarray(D_rt, float).reshape(-1)
    dev = D_rt - float(d_da)
    return (v - float(p0)) * float(d_da) + (v - p_w) * dev - delta * np.abs(dev)


def planner_losses(p0, Pi, mc_g_i, mv_d_j, g_da, d_da, G_rt, D_rt,
                   mc_delta=None, mv_delta=None):
    """Per-agent per-scenario losses L = -u at the planner's allocation.

    Returns (L_prod, L_cons, ell) with shapes (m,S), (n,S), (S,).
    Eq. (110) asserts sum_j L_prod + sum_i L_cons = ell (price terms cancel).
    """
    mc = np.asarray(mc_g_i, float).reshape(-1)
    mv = np.asarray(mv_d_j, float).reshape(-1)
    if mc_delta is None:
        mc_delta = mc / 10.0
    if mv_delta is None:
        mv_delta = mv / 10.0
    mc_delta = np.asarray(mc_delta, float).reshape(-1)
    mv_delta = np.asarray(mv_delta, float).reshape(-1)

    g_da = np.asarray(g_da, float).reshape(-1)
    d_da = np.asarray(d_da, float).reshape(-1)
    G_rt = np.asarray(G_rt, float)
    D_rt = np.asarray(D_rt, float)

    m, n = mc.size, mv.size
    L_prod = np.stack([
        -producer_utility(p0, Pi, mc[j], mc_delta[j], g_da[j], G_rt[:, j])
        for j in range(m)
    ])
    L_cons = np.stack([
        -consumer_utility(p0, Pi, mv[i], mv_delta[i], d_da[i], D_rt[:, i])
        for i in range(n)
    ])

    # ell(w) = -W2 : the planner's per-scenario loss, built from primitives only
    dG = G_rt - g_da[None, :]
    dD = D_rt - d_da[None, :]
    gen_cost = G_rt @ mc + np.abs(dG) @ mc_delta
    load_cost = -(D_rt @ mv) + np.abs(dD) @ mv_delta
    ell = gen_cost + load_cost
    return L_prod, L_cons, ell


# ---------------------------------------------------------------------------
# Transfer constructions
# ---------------------------------------------------------------------------

def transfers_equal_share(L_prod, L_cons, ell, delta_prod=None, delta_cons=None,
                          as_printed=False):
    """Remark 7: equal-share comonotone construction, SIGN-CORRECTED.

        r_k(w) = +L_k(w) - ell(w)/(n+m) - delta_k,   sum_k delta_k = 0

    so that the agent's insured loss is

        L_k(w) - r_k(w) = ell(w)/(n+m) + delta_k,

    affine increasing in the aggregate loss with the same positive coefficient
    for every agent: all agents then share one tail set.

    Why the sign differs from Eq. (111)-(112) as printed
    ----------------------------------------------------
    The agent's per-scenario payoff is u_k + r_k (Section 8.4 text), so the loss
    entering the CVaR constraint Eq. (97)/(101) is  -u_k - r_k = L_k - r_k.
    Flattening *that* requires r_k = L_k - ell/N - delta_k. Eq. (111)-(112) as
    printed give r_k = -L_k + ell/N + delta_k, which flattens L_k + r_k instead
    and in fact doubles the agent's exposure; numerically it breaks (UM).

    Eq. (122) in Theorem 9 already uses the corrected sign
    (r = L - t_k - w_k(ell - t*)), so Eq. (111)-(112) are the inconsistent pair.
    Both signs satisfy market clearing Eq. (95), which is why the error survives
    that check.

    Pass as_printed=True to reproduce Eq. (111)-(112) verbatim (used to document
    the discrepancy).
    """
    m, S = L_prod.shape
    n = L_cons.shape[0]
    N = n + m
    if delta_prod is None:
        delta_prod = np.zeros(m)
    if delta_cons is None:
        delta_cons = np.zeros(n)
    dp = np.asarray(delta_prod, float).reshape(m)
    dc = np.asarray(delta_cons, float).reshape(n)
    if abs(dp.sum() + dc.sum()) > 1e-9:
        raise ValueError("delta constants must sum to zero (Remark 7)")

    if as_printed:                                   # Eq. (111)-(112) verbatim
        r_prod = -L_prod + ell[None, :] / N + dp[:, None]
        r_cons = -L_cons + ell[None, :] / N + dc[:, None]
    else:                                            # sign-corrected
        r_prod = L_prod - ell[None, :] / N - dp[:, None]
        r_cons = L_cons - ell[None, :] / N - dc[:, None]
    return r_prod, r_cons


def transfers_heterogeneous(L_prod, L_cons, ell, beta_prod, beta_cons,
                            t_star=None, probs=None, beta_min=None):
    """Theorem 9 / Eq. (122)-(123): construction under heterogeneous beta.

        r_k(w) = L_k(w) - t_k - w_k (ell(w) - t*)
        w_k = 1/|S_min| if beta_k == beta_min else 0

    Agents with beta_k > beta_min (the set S_>) get w_k = 0, hence a *constant*
    insured loss across the whole tail, as Eq. (121) requires. Only agents at
    beta_min bear residual aggregate risk.

    NOTE: the sign convention here follows Eq. (122) as printed. `t_k` is each
    agent's VaR-level variable; absent the agents' own solves we use the
    per-agent CVaR threshold implied by the planner's tail, which is what the
    theorem's construction presumes. Returns (r_prod, r_cons, info).
    """
    m, S = L_prod.shape
    n = L_cons.shape[0]
    beta_p = np.asarray(beta_prod, float).reshape(m)
    beta_c = np.asarray(beta_cons, float).reshape(n)
    all_beta = np.concatenate([beta_c, beta_p])
    if beta_min is None:
        beta_min = float(all_beta.min())

    # S_min membership
    in_min = np.isclose(all_beta, beta_min)
    n_min = int(in_min.sum())
    if n_min == 0:
        raise ValueError("no agent attains beta_min")
    w_all = np.where(in_min, 1.0 / n_min, 0.0)
    w_cons, w_prod = w_all[:n], w_all[n:]

    L_all = np.vstack([L_cons, L_prod])
    if t_star is None:
        if probs is None:
            raise ValueError("need probs to compute VaR thresholds")
        t_k = np.array([_var_level(L_all[k], probs, beta_min) for k in range(n + m)])
        t_star_val = float(t_k.sum())
    else:
        t_k = np.asarray(t_star, float).reshape(n + m)
        t_star_val = float(t_k.sum())

    r_all = L_all - t_k[:, None] - w_all[:, None] * (ell[None, :] - t_star_val)
    info = dict(beta_min=beta_min, n_min=n_min, w=w_all, t_k=t_k, t_star=t_star_val,
                S_gt=np.where(~in_min)[0])
    return r_all[n:], r_all[:n], info


def _var_level(loss, probs, beta):
    """beta-VaR of a discrete loss distribution."""
    loss = np.asarray(loss, float).reshape(-1)
    probs = np.asarray(probs, float).reshape(-1)
    order = np.argsort(loss)
    cum = np.cumsum(probs[order])
    idx = int(np.searchsorted(cum, beta, side="left"))
    idx = min(idx, loss.size - 1)
    return float(loss[order][idx])


# ---------------------------------------------------------------------------
# Individual agent problems
# ---------------------------------------------------------------------------

def solve_agent(kind, probs, p0, Pi, cost_or_value, delta, cap_lo, cap_hi,
                eps, beta, q=None, r_fixed=None, insurance=True,
                solver=None, verbose=False, r_bound=None):
    """Solve one agent's risk-aware problem.

    kind : "producer" (Eq. 100-103) or "consumer" (Eq. 96-99).
    insurance=False gives the uninsured problem, Eq. (78)-(81) / (82)-(85).
    r_fixed : if given, insurance holdings are held at these values rather than
              optimised (used to test that the constructed transfer is optimal).

    Returns dict with status, objective, day-ahead and real-time allocation,
    the CVaR multiplier gamma (dual of the shortfall constraint) and the
    agent's tilted weight rho_tilde = (1-eps) rho + gamma.
    """
    probs = np.asarray(probs, float).reshape(-1)
    Pi = np.asarray(Pi, float).reshape(-1)
    S = probs.size
    cap_lo = np.asarray(cap_lo, float).reshape(-1)
    cap_hi = np.asarray(cap_hi, float).reshape(-1)

    a = cp.Variable(nonneg=True)              # day-ahead quantity (x_i or y_j)
    Up = cp.Variable(S, nonneg=True)          # positive deviation
    Un = cp.Variable(S, nonneg=True)          # negative deviation
    t = cp.Variable()                         # CVaR threshold t_k
    zeta = cp.Variable(S, nonneg=True)        # CVaR slack

    total = a + Up - Un                       # realised second-stage quantity
    dev = Up - Un

    if kind == "producer":
        c = float(cost_or_value)
        u = (p0 - c) * a + cp.multiply(Pi - c, dev) - delta * (Up + Un)
    elif kind == "consumer":
        v = float(cost_or_value)
        u = (v - p0) * a + cp.multiply(v - Pi, dev) - delta * (Up + Un)
    else:
        raise ValueError(kind)

    cons = [total >= cap_lo, total <= cap_hi]

    q_vec = None if q is None else np.asarray(q, float).reshape(-1)
    r_var = None
    if insurance:
        if q_vec is None:
            raise ValueError("insurance=True requires insurance prices q")
        if r_fixed is not None:
            r_val = np.asarray(r_fixed, float).reshape(-1)
            premium = float(q_vec @ r_val)
            payoff = u + r_val
        else:
            r_var = cp.Variable(S)
            premium = q_vec @ r_var
            payoff = u + r_var
            # At q = rho_tilde the objective is EXACTLY flat in r (Eq. 108 is an
            # identity, not a pin), so the LP has an unbounded recession ray and
            # any numerical noise in gamma makes it strictly improving. A box
            # makes the problem well posed; callers must check it is not active.
            if r_bound is not None:
                cons += [r_var <= float(r_bound), r_var >= -float(r_bound)]
    else:
        premium = 0.0
        payoff = u

    # Eq. (97)/(101): zeta_w >= -payoff_w - t
    cons_shortfall = (zeta >= -payoff - t)
    cons.append(cons_shortfall)

    obj = ((1.0 - eps) * (probs @ payoff) - premium
           - eps * (t + (1.0 / (1.0 - beta)) * (probs @ zeta)))

    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.GUROBI if solver is None else solver, verbose=verbose)
    except Exception as e:                                  # noqa: BLE001
        return dict(status=f"error: {type(e).__name__}: {e}", objective=np.nan)

    out = dict(status=prob.status, objective=float(prob.value) if prob.value is not None else np.nan)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return out

    gamma = None
    if cons_shortfall.dual_value is not None:
        gamma = np.asarray(cons_shortfall.dual_value, float).reshape(-1)

    if r_var is not None and r_bound is not None and r_var.value is not None:
        out["r_bound_active"] = bool(
            np.max(np.abs(np.asarray(r_var.value, float))) > 0.999 * float(r_bound))
    out.update(
        da=float(a.value),
        rt=np.asarray(total.value, float).reshape(-1),
        up=np.asarray(Up.value, float).reshape(-1),
        un=np.asarray(Un.value, float).reshape(-1),
        t=float(t.value),
        gamma=gamma,
        rho_tilde=None if gamma is None else (1.0 - eps) * probs + gamma,
    )
    if not insurance:
        out["r"], out["premium"] = None, 0.0
    elif r_fixed is not None:
        r_sol = np.asarray(r_fixed, float).reshape(-1)
        out["r"], out["premium"] = r_sol, float(q_vec @ r_sol)
    else:
        r_sol = np.asarray(r_var.value, float).reshape(-1)
        out["r"], out["premium"] = r_sol, float(q_vec @ r_sol)
    # expected payout under the physical measure, for the premium decomposition
    if out["r"] is not None:
        out["expected_payout"] = float(probs @ out["r"])
        out["tail_premium"] = out["premium"] - out["expected_payout"]
    return out


def objective_at(kind, probs, p0, Pi, cost_or_value, delta, eps, beta,
                 da, rt, q=None, r=None, insurance=True):
    """Evaluate an agent's objective at a *given* allocation (no optimisation).

    Used to test (UM): the planner's allocation must attain the agent's optimal
    objective value. Robust to degenerate optima, unlike comparing allocations.
    """
    probs = np.asarray(probs, float).reshape(-1)
    Pi = np.asarray(Pi, float).reshape(-1)
    rt = np.asarray(rt, float).reshape(-1)
    if kind == "producer":
        u = producer_utility(p0, Pi, cost_or_value, delta, da, rt)
    else:
        u = consumer_utility(p0, Pi, cost_or_value, delta, da, rt)
    payoff = u + (np.asarray(r, float).reshape(-1) if (insurance and r is not None) else 0.0)
    premium = float(np.asarray(q, float).reshape(-1) @ np.asarray(r, float).reshape(-1)) \
        if (insurance and r is not None) else 0.0
    # CVaR of the loss -payoff at level beta, via Rockafellar-Uryasev
    t_opt = _var_level(-payoff, probs, beta)
    cvar = t_opt + (1.0 / (1.0 - beta)) * float(probs @ np.maximum(-payoff - t_opt, 0.0))
    return float((1.0 - eps) * (probs @ payoff) - premium - eps * cvar)
