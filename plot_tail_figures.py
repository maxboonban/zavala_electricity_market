#!/usr/bin/env python
"""Conference-ready figures for the tail-welfare experiments.

    python plot_tail_figures.py --result outputs/results/tail_main.json \
        --outdir figures/tail

Emits PDF (for LaTeX) and PNG (for slides) at publication sizes. Figures are
sized for a two-column IEEE/ACM layout: single column 3.5in, double 7.16in.
Fonts are set so that text in the figure matches ~8pt body text at final size,
which is the usual reason a figure gets rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Okabe-Ito: colourblind-safe and distinguishable in greyscale.
C = dict(
    neutral="#0072B2", aware="#D55E00", det="#009E73",
    accent="#CC79A7", grey="#666666", light="#BBBBBB",
    energy="#56B4E9", up="#E69F00", dn="#009E73",
    load="#CC79A7", unserved="#999999",
)

COL1, COL2 = 3.5, 7.16


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
    })


def save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf / .png")


def _ci(d):
    """(mean, lo, hi) from a paired_ci dict, NaN-safe."""
    if not isinstance(d, dict):
        return np.nan, np.nan, np.nan
    g = lambda k: float(d[k]) if d.get(k) is not None else np.nan  # noqa: E731
    return g("mean"), g("lo"), g("hi")


def _levels(rows):
    return [float(r["level"]) for r in rows]


# --------------------------------------------------------------------------
# Fig 1 : the structural ceiling (W0) -- the most important figure
# --------------------------------------------------------------------------

def fig_ceiling(res, outdir):
    w0 = res.get("W0")
    if not w0 or not w0.get("rows"):
        return
    s, rows = w0["summary"], w0["rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.3))

    # left: tail cost split into reachable and unreachable
    reach = s["mean_reachable_cost"]
    unreach = s["mean_tail_cost"] - reach
    ax1.barh([0], [unreach], height=0.45, color=C["unserved"],
             edgecolor="white", lw=0.8,
             label=f"unserved $\\times$ VOLL ({100*s['mean_unreachable_share']:.0f}%)")
    ax1.barh([0], [reach], left=[unreach], height=0.45, color=C["aware"],
             edgecolor="white", lw=0.8,
             label=f"energy + adjustment ({100*(1-s['mean_unreachable_share']):.0f}%)")
    ax1.set_yticks([])
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_xlabel("Conditional tail cost (\\$)")
    ax1.set_title("(a) What the day-ahead policy can reach", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax1.grid(axis="y", visible=False)

    # right: unserved energy under each policy, per seed
    un_rn = np.array([r["unserved_rn"] for r in rows])
    un_ra = np.array([r["unserved_ra"] for r in rows])
    idx = np.arange(len(rows))
    ax2.plot(idx, un_rn, "o-", color=C["neutral"], label="risk-neutral", zorder=3)
    ax2.plot(idx, un_ra, "s--", color=C["aware"], label="risk-aware", zorder=4)
    ax2.set_xlabel("Instance (seed index)")
    ax2.set_ylabel("CVaR$_{95}$ unserved (MW)")
    ax2.set_title("(b) Identical under both policies", loc="left")
    ax2.legend(loc="best")
    rel = s.get("rel_d_unserved", np.nan)
    ax2.annotate(f"max relative difference {rel:.1e}",
                 xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                 fontsize=7, color=C["grey"])
    save(fig, outdir, "fig1_structural_ceiling")


# --------------------------------------------------------------------------
# Fig 2 : headline with paired CIs (W1)
# --------------------------------------------------------------------------

def fig_headline(res, outdir):
    w1 = res.get("W1")
    if not w1 or not w1.get("per_seed"):
        return
    ps = w1["per_seed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.5))

    # left: per-seed paired CVaR welfare
    rn = np.array([p["cvar_rn"] for p in ps])
    ra = np.array([p["cvar_ra"] for p in ps])
    idx = np.arange(len(ps))
    for i in idx:
        ax1.plot([0, 1], [rn[i], ra[i]], "-", color=C["light"], lw=0.7, zorder=1)
    ax1.scatter(np.zeros_like(rn), rn, s=16, color=C["neutral"],
                label="risk-neutral", zorder=3)
    ax1.scatter(np.ones_like(ra), ra, s=16, color=C["aware"], marker="s",
                label="risk-aware", zorder=3)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["risk-neutral", "risk-aware"])
    ax1.set_xlim(-0.35, 1.35)
    ax1.set_ylabel("CVaR$_{95}$ welfare (\\$)")
    n_up = int((ra > rn).sum())
    ax1.set_title(f"(a) Paired by instance: {n_up}/{len(ps)} improve", loc="left")
    ax1.legend(loc="best")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax1.grid(axis="x", visible=False)

    # right: gap under each denominator
    vs = w1.get("vs_risk_neutral", {})
    keys = [("pct_of_tail_cost", "% of tail cost"),
            ("pct_of_total_welfare", "% of total welfare"),
            ("dollars_per_mwh", "\\$/MWh served")]
    ys, ms, los, his, labels = [], [], [], [], []
    for i, (k, lab) in enumerate(keys):
        m, lo, hi = _ci(vs.get(k))
        if not np.isfinite(m):
            continue
        ys.append(i); ms.append(m); los.append(lo); his.append(hi); labels.append(lab)
    ys = np.array(ys, float)
    ax2.axvline(0, color=C["grey"], lw=0.7, ls=":")
    ax2.errorbar(ms, ys, xerr=[np.array(ms) - np.array(los),
                               np.array(his) - np.array(ms)],
                 fmt="o", color=C["aware"], capsize=3, lw=1.2, markersize=5)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels)
    ax2.set_ylim(-0.6, len(labels) - 0.4)
    ax2.set_xlabel("Tail-welfare gain (95\\% CI, paired)")
    ax2.set_title("(b) The denominator matters", loc="left")
    ax2.grid(axis="y", visible=False)
    save(fig, outdir, "fig2_headline")


# --------------------------------------------------------------------------
# Fig 3 : mechanism and regime sweeps (W2, W3, W4)
# --------------------------------------------------------------------------

def fig_sweeps(res, outdir):
    specs = [("W2", r"Adjustment asymmetry $\alpha^+/c$", "(a) Mechanism"),
             ("W3", "Adequacy (capacity / mean demand)", "(b) Capacity regime"),
             ("W4", "VOLL (\\$/MWh)", "(c) VOLL robustness")]
    have = [s for s in specs if res.get(s[0])]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(COL2, 2.2), sharey=False)
    if len(have) == 1:
        axes = [axes]

    for ax, (key, xlab, title) in zip(axes, have):
        rows = res[key]
        x = _levels(rows)
        m, lo, hi = zip(*[_ci(r["pct_of_tail_cost"]) for r in rows])
        m, lo, hi = np.array(m), np.array(lo), np.array(hi)
        ax.axhline(0, color=C["grey"], lw=0.7, ls=":")
        ax.fill_between(x, lo, hi, color=C["aware"], alpha=0.16, lw=0)
        ax.plot(x, m, "o-", color=C["aware"])
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left")
        if key == "W4":
            ax.set_xscale("log")
        if ax is axes[0]:
            ax.set_ylabel("Tail-welfare gain\n(\\% of tail cost)")
    save(fig, outdir, "fig3_sweeps")


def fig_voll_dollars(res, outdir):
    """VOLL robustness in $/MWh -- the direct rebuttal figure."""
    rows = res.get("W4")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    x = _levels(rows)
    m, lo, hi = zip(*[_ci(r["dollars_per_mwh"]) for r in rows])
    m, lo, hi = np.array(m), np.array(lo), np.array(hi)
    ax.axhline(0, color=C["grey"], lw=0.7, ls=":")
    ax.fill_between(x, lo, hi, color=C["accent"], alpha=0.16, lw=0)
    ax.plot(x, m, "o-", color=C["accent"])
    ax.set_xscale("log")
    ax.set_xlabel("VOLL (\\$/MWh)")
    ax.set_ylabel("Tail-welfare gain (\\$/MWh served)")
    ax.set_title("Gain is not a VOLL artefact", loc="left")
    save(fig, outdir, "fig4_voll_robustness")


# --------------------------------------------------------------------------
# Fig 5 : tail decomposition (W5)
# --------------------------------------------------------------------------

def fig_decomposition(res, outdir):
    w5 = res.get("W5")
    if not w5 or not w5.get("means"):
        return
    means = w5["means"]
    order = [p for p in ("deterministic", "risk_neutral", "risk_aware") if p in means]
    comps = [("energy", "energy", C["energy"]),
             ("dev_up", "upward adjustment", C["up"]),
             ("dev_dn", "downward adjustment", C["dn"]),
             ("load_dev", "load adjustment", C["load"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.4))
    xs = np.arange(len(order))
    bottom = np.zeros(len(order))
    for k, lab, col in comps:
        v = np.array([means[p][k] for p in order])
        ax1.bar(xs, v, bottom=bottom, color=col, label=lab,
                edgecolor="white", lw=0.6, width=0.62)
        bottom += v
    ax1.set_xticks(xs)
    ax1.set_xticklabels([p.replace("_", "-") for p in order], rotation=12)
    ax1.set_ylabel("Conditional tail cost (\\$)")
    ax1.set_title("(a) Reachable tail cost", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax1.grid(axis="x", visible=False)

    if "risk_neutral" in means and "risk_aware" in means:
        d = [means["risk_aware"][k] - means["risk_neutral"][k] for k, _, _ in comps]
        cols = [C["aware"] if v < 0 else C["neutral"] for v in d]
        ax2.barh(np.arange(len(d)), d, color=cols, edgecolor="white",
                 lw=0.6, height=0.6)
        ax2.axvline(0, color=C["grey"], lw=0.7)
        ax2.set_yticks(np.arange(len(d)))
        ax2.set_yticklabels([lab for _, lab, _ in comps])
        ax2.set_xlabel("Change vs risk-neutral (\\$)")
        ax2.set_title("(b) What risk aversion buys", loc="left")
        ax2.grid(axis="y", visible=False)
    save(fig, outdir, "fig5_decomposition")


# --------------------------------------------------------------------------
# Fig 6 : risk-return frontier (W6)
# --------------------------------------------------------------------------

def fig_frontier(res, outdir):
    rows = res.get("W6")
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.4))
    eps = np.array([r["eps"] for r in rows])
    e_m = np.array([_ci(r["expected_welfare"])[0] for r in rows])
    c_m = np.array([_ci(r["cvar_welfare"])[0] for r in rows])
    c_lo = np.array([_ci(r["cvar_welfare"])[1] for r in rows])
    c_hi = np.array([_ci(r["cvar_welfare"])[2] for r in rows])

    sc = ax1.scatter(e_m, c_m, c=eps, cmap="viridis", s=34, zorder=3,
                     edgecolor="white", lw=0.5)
    ax1.plot(e_m, c_m, "-", color=C["light"], lw=0.9, zorder=2)
    cb = fig.colorbar(sc, ax=ax1, pad=0.02)
    cb.set_label(r"risk weight $\varepsilon$")
    cb.ax.tick_params(labelsize=7)
    ax1.set_xlabel("Expected welfare (\\$)")
    ax1.set_ylabel("CVaR$_{95}$ welfare (\\$)")
    ax1.set_title("(a) Risk-return frontier", loc="left")
    for f in (ax1.xaxis, ax1.yaxis):
        f.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

    ax2.fill_between(eps, c_lo, c_hi, color=C["aware"], alpha=0.16, lw=0)
    ax2.plot(eps, c_m, "o-", color=C["aware"])
    ax2.set_xlabel(r"risk weight $\varepsilon$")
    ax2.set_ylabel("CVaR$_{95}$ welfare (\\$)")
    ax2.set_title(r"(b) Tail welfare in $\varepsilon$", loc="left")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    # Eq. (107) loses meaning at eps = 1 (theory note T11).
    ax2.axvline(1.0, color=C["grey"], ls="--", lw=0.8)
    ax2.annotate(r"$\varepsilon<1$ (T11)", xy=(0.98, 0.04),
                 xycoords="axes fraction", ha="right", fontsize=7,
                 color=C["grey"])
    save(fig, outdir, "fig6_frontier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--outdir", default="figures/tail")
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        payload = json.load(f)
    res = payload.get("results", payload)

    style()
    print(f"figures from {args.result}:")
    fig_ceiling(res, args.outdir)
    fig_headline(res, args.outdir)
    fig_sweeps(res, args.outdir)
    fig_voll_dollars(res, args.outdir)
    fig_decomposition(res, args.outdir)
    fig_frontier(res, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
