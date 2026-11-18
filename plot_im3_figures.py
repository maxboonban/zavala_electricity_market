#!/usr/bin/env python
"""Conference-ready figures for the D/E suites (IM3 GO-WEST + decentralisation).

    python plot_im3_figures.py --tag im3_S500_9253564 --outdir figures/im3

Reads the JSONs written by run_im3_experiments.py and emits one PDF+PNG pair
per figure, sized for an IEEE two-column template.

Palette is the Okabe-Ito colourblind-safe set, matching plot_tail_figures.py so
the two figure families read as one. The three categorical hues used here were
checked for CVD separation (worst adjacent pair dE = 11.0 deutan, normal-vision
25.8) and for >= 3:1 contrast against the page.
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

# --- house style -------------------------------------------------------------
C = dict(
    blue="#0072B2",      # primary / "risk-aware"
    verm="#D55E00",      # contrast / "worse"
    green="#009E73",     # third categorical / "better"
    grey="#666666",
    light="#BBBBBB",
    rule="#DDDDDD",
)
COL1, COL2 = 3.5, 7.16   # IEEE single- and double-column widths, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.color": C["rule"],
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

_k = FuncFormatter(lambda v, _: f"{v/1e3:,.0f}k" if abs(v) >= 1e3 else f"{v:,.0f}")


def save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=400)
    plt.close(fig)
    print(f"  wrote {outdir/name}.pdf / .png")


def _tidy(ax, axis="y"):
    ax.grid(axis=axis, alpha=0.9)
    ax.set_axisbelow(True)


# =============================================================================
# Fig 1 (E5) -- paired vs unpaired reporting. The methodological headline.
# =============================================================================
def fig_paired(E5, outdir):
    eps = sorted(E5.keys(), key=float)
    diff = np.array([E5[e]["paired_diff"] for e in eps])
    pci = np.array([E5[e]["paired_ci"] for e in eps])
    uci = np.array([E5[e]["unpaired_ci"] for e in eps])
    wins = [E5[e]["wins"] for e in eps]
    n = E5[eps[0]]["n"]
    y = np.arange(len(eps))

    fig, ax = plt.subplots(figsize=(COL2, 2.4))

    # What the unpaired reporting implies: an interval wide enough to straddle 0.
    ax.barh(y, 2 * uci, left=diff - uci, height=0.52,
            color=C["verm"], alpha=0.22, edgecolor=C["verm"], linewidth=0.8,
            label=f"unpaired $\\pm$ (as printed in Tables 1, 3)")
    # What the paired design licenses.
    ax.errorbar(diff, y, xerr=pci, fmt="o", color=C["blue"], markersize=5,
                capsize=2.5, elinewidth=1.6, markeredgecolor="white",
                markeredgewidth=0.7, zorder=3,
                label="paired difference $\\pm$ 95% CI")
    ax.axvline(0, color=C["grey"], linewidth=0.9, zorder=1)

    # Wins go in a margin column at a fixed x, so they never collide with a bar
    # whose extent changes with eps.
    from matplotlib.transforms import blended_transform_factory
    tf = blended_transform_factory(ax.transAxes, ax.transData)
    for i, w in enumerate(wins):
        ax.text(1.015, i, f"{w}/{n}", transform=tf, va="center", ha="left",
                fontsize=7.2, color="#222222")
    ax.text(1.015, -0.75, "wins", transform=tf, va="center", ha="left",
            fontsize=7, color=C["grey"], style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([f"$\\varepsilon$ = {e}" for e in eps])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_k)
    lo = (diff - uci).min()
    hi = (diff + uci).max()
    ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_xlabel(r"change in tail loss CVaR$_{0.95}[\ell]$ vs risk-neutral  (\$/h)"
                  "\n" r"$\leftarrow$ risk-aware better")
    ax.set_title("Every CVaR level beats risk-neutral in all "
                 f"{n} replications;\nonly the paired interval shows it",
                 loc="left", fontweight="bold", pad=18)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              borderaxespad=0.2, handlelength=1.4, columnspacing=1.2)
    _tidy(ax, "x")
    save(fig, outdir, "fig1_paired_vs_unpaired")


# =============================================================================
# Fig 2 (E4) -- out-of-sample validation.
# =============================================================================
def fig_oos(E4, cfg, outdir):
    rows = [("dCVaR", r"tail   CVaR$_{0.95}[\ell]$"),
            ("dRho",  r"planner objective   $\rho_{RA}[\ell]$"),
            ("dE",    r"expected   $E[\ell]$")]
    vals = [E4[k]["mean"] for k, _ in rows]
    cis = [E4[k]["ci95"] for k, _ in rows]
    wins = [E4[k]["wins"] for k, _ in rows]
    n = E4[rows[0][0]]["n"]
    y = np.arange(len(rows))
    cols = [C["blue"] if v < 0 else C["verm"] for v in vals]

    fig, ax = plt.subplots(figsize=(COL2, 2.0))
    ax.barh(y, vals, height=0.5, color=cols, alpha=0.85, zorder=2)
    ax.errorbar(vals, y, xerr=cis, fmt="none", ecolor="#333333",
                capsize=2.5, elinewidth=1.1, zorder=3)
    ax.axvline(0, color=C["grey"], linewidth=0.9, zorder=1)

    for i, (v, c, w) in enumerate(zip(vals, cis, wins)):
        off = -1 if v < 0 else 1
        ax.text(v + off * (c + 9000), i,
                f"{v:+,.0f} $\\pm$ {c:,.0f}   ({w}/{n})",
                va="center", ha="right" if v < 0 else "left", fontsize=7.2,
                color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(_k)
    ax.set_xlim(min(vals) * 1.95, max(vals) * 2.6)
    ax.set_xlabel(r"held-out loss, risk-aware $-$ risk-neutral  (\$/h)"
                  "\n" r"$\leftarrow$ risk-aware better")
    ax.set_title(f"Out of sample on disjoint scenarios: the tail improves in "
                 f"{E4['dCVaR']['wins']}/{n} replications\n"
                 f"(train S={cfg.get('train')}, test S={cfg.get('test')}, "
                 "drawn from the same conditional pool)",
                 loc="left", fontweight="bold")
    _tidy(ax, "x")
    save(fig, outdir, "fig2_out_of_sample")


# =============================================================================
# Fig 3 (E6) -- linearity in the deviation bid price.
# =============================================================================
def fig_delta(E6, outdir):
    rows = sorted(E6["rows"], key=lambda r: r["delta_scale"])
    d = np.array([r["delta_scale"] for r in rows])
    v = np.array([r["gap_per_mwh"] for r in rows])
    slope = E6["slope_mean"]

    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    xs = np.linspace(0, d.max() * 1.06, 50)
    ax.plot(xs, slope * xs, color=C["grey"], linewidth=1.0, linestyle="--",
            zorder=1, label=f"${slope:.3f}\\times\\delta$  (fit)")
    ax.plot(d, v, "o", color=C["blue"], markersize=5.5, zorder=3,
            markeredgecolor="white", markeredgewidth=0.7, label="measured")
    ax.set_xlabel(r"deviation bid price scale  $\delta$  (fraction of VOLL)")
    ax.set_ylabel(r"value of insurance market" "\n" r"(\$/MWh served)")
    ax.set_xlim(0, d.max() * 1.06)
    ax.set_ylim(0, v.max() * 1.1)
    ax.set_title("The headline $/MWh is linear in an\narbitrary constant"
                 " — quote the slope", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.06,
            f"slope constant to\n{E6['slope_std']/slope*100:.1e}%  over 20$\\times$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=C["grey"])
    _tidy(ax)
    save(fig, outdir, "fig3_delta_linearity")


# =============================================================================
# Fig 4 (E3) -- decarbonisation at fixed total capacity.
# =============================================================================
def fig_decarb(E3, outdir):
    rows = sorted(E3, key=lambda r: r["renewable_share"])
    sh = np.array([r["renewable_share"] for r in rows]) * 100
    val = np.array([r["gap_per_mwh"] for r in rows])
    firm = np.array([r["firm_mw"] for r in rows]) / 1e3
    dE = np.array([r["E_ra"] - r["E_rn"] for r in rows])
    dC = np.array([r["CVaR_ra"] - r["CVaR_rn"] for r in rows])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 2.1))

    a1.plot(sh, firm, "o-", color=C["grey"], markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6)
    a1.set_ylabel("firm capacity (GW)")
    a1.set_title("Substitution shrinks\nfirm capacity", loc="left")

    a2.plot(sh, val, "o-", color=C["blue"], markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6)
    a2.set_ylabel(r"insurance value (\$/MWh)")
    a2.set_ylim(0, val.max() * 1.18)
    a2.set_title("but its value is\nnearly flat", loc="left")

    a3.plot(sh, dC / 1e3, "o-", color=C["blue"], markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6, label=r"CVaR$_{0.95}$")
    a3.plot(sh, dE / 1e3, "s--", color=C["verm"], markersize=4,
            markeredgecolor="white", markeredgewidth=0.6, label=r"$E[\ell]$")
    a3.axhline(0, color=C["grey"], linewidth=0.8)
    a3.set_ylabel(r"risk-aware $-$ risk-neutral" "\n" r"(k\$/h)")
    a3.set_title("The trade is stable\nacross the mix", loc="left")
    a3.legend(loc="center right")

    for a in (a1, a2, a3):
        a.set_xlabel("renewable share (%)")
        _tidy(a)
    fig.subplots_adjust(wspace=0.52)
    fig.suptitle("Total installed capacity and load held fixed; only the mix varies",
                 fontsize=8, fontweight="bold", x=0.005, ha="left", y=1.06)
    save(fig, outdir, "fig4_decarbonisation")


# =============================================================================
# Fig 5 (E2) -- who bears the tail risk.
# =============================================================================
def fig_incidence(E2, outdir):
    by = E2["by_tech"]
    items = list(by.items()) if isinstance(by, dict) else by
    gens = [(k, v) for k, v in items if "LOAD" not in k.upper()]
    gens.sort(key=lambda kv: kv[1]["per_mw"])
    names = [k for k, _ in gens]
    per = np.array([v["per_mw"] for _, v in gens])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.2),
                                 gridspec_kw={"width_ratios": [1.5, 1]})

    cols = [C["verm"] if p < 0 else C["blue"] for p in per]
    a1.barh(np.arange(len(per)), per, height=0.55, color=cols, alpha=0.9)
    a1.axvline(0, color=C["grey"], linewidth=0.9)
    a1.set_yticks(np.arange(len(per)))
    a1.set_yticklabels([n.split(" (")[0] for n in names])
    a1.invert_yaxis()
    a1.set_xlabel(r"own tail-risk premium $\pi_k$ (\$/MW capacity)")
    a1.set_title("Generators sell cover", loc="left", fontweight="bold")
    for i, p in enumerate(per):
        a1.text(p + (-0.06 if p < 0 else 0.06), i, f"{p:+.2f}",
                va="center", ha="right" if p < 0 else "left", fontsize=7,
                color="#222222")
    a1.set_xlim(per.min() * 1.32, max(per.max(), 0) + 0.52)
    # The exact zeros are a property of the DATASET, not of the theory: thermal
    # capacity is constant across scenarios, so an infra-marginal unit has a
    # constant loss L and sum_w (q_w - rho_w) L = L * 0 = 0. Say so on the
    # figure, because a reader will otherwise read it as a bug.
    if np.any(np.isclose(per, 0.0)):
        a1.text(0.015, 0.025,
                "exact zeros: thermal capacity is constant across\n"
                "scenarios in this dataset, so $L$ is constant and\n"
                r"$\sum_\omega (q_\omega-\rho_\omega)L = 0$. Forced outages"
                " would\nremove this.",
                transform=a1.transAxes, fontsize=6.1, color=C["grey"],
                va="bottom", ha="left", linespacing=1.35)

    tot = abs(E2["renewable_own"]) + abs(E2["thermal_own"]) + abs(E2["load_own"])
    shares = [abs(E2["renewable_own"]) / tot * 100,
              abs(E2["thermal_own"]) / tot * 100,
              abs(E2["load_own"]) / tot * 100]
    labs = ["renewable", "thermal", "load"]
    a2.bar(np.arange(3), shares, width=0.55,
           color=[C["green"], C["grey"], C["blue"]], alpha=0.9)
    a2.set_xticks(np.arange(3))
    a2.set_xticklabels(labs)
    a2.set_ylabel(r"share of total $|\pi|$ (%)")
    a2.set_ylim(0, 112)
    for i, s in enumerate(shares):
        a2.text(i, s + 3, f"{s:.1f}%", ha="center", fontsize=7.2)
    a2.set_title("Load is the residual\nrisk bearer", loc="left", fontweight="bold")

    for a in (a1, a2):
        _tidy(a, "x" if a is a1 else "y")
    save(fig, outdir, "fig5_risk_incidence")


# =============================================================================
# Fig 6 -- seasonal / delivery-hour robustness.
# =============================================================================
def fig_seasonal(seasonal, outdir):
    if not seasonal:
        return
    order = ["winter_h8", "winter_h18", "summer_h8", "summer_h18"]
    order = [k for k in order if k in seasonal]
    lab = {"winter_h8": "winter\n08:00", "winter_h18": "winter\n18:00",
           "summer_h8": "summer\n08:00", "summer_h18": "summer\n18:00"}
    val = [seasonal[k]["gap_per_mwh"] for k in order]
    cols = [C["grey"] if k.startswith("winter") else C["blue"] for k in order]

    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    ax.bar(np.arange(len(order)), val, width=0.56, color=cols, alpha=0.9)
    for i, v in enumerate(val):
        ax.text(i, v + 0.035, f"{v:.3f}", ha="center", fontsize=7.2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([lab[k] for k in order])
    ax.set_ylabel(r"value of insurance market" "\n" r"(\$/MWh served)")
    ax.set_ylim(0, max(val) * 1.22)
    ax.set_title("Positive in every season and\ndelivery hour", loc="left",
                 fontweight="bold")
    ax.text(0.98, 0.95, "headline config", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=C["blue"])
    _tidy(ax)
    save(fig, outdir, "fig6_seasonal_robustness")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True,
                    help="e.g. im3_S500_9253564 (the part between im3_ and _main)")
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--outdir", default="figures/im3")
    args = ap.parse_args()

    R = Path(args.results)
    out = Path(args.outdir)

    def load(suffix):
        p = R / f"im3_{args.tag}_{suffix}.json"
        if not p.exists():
            print(f"  [skip] {p} not found")
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    print(f"Figures for tag {args.tag} -> {out}/")
    main_j, oos_j, paired_j = load("main"), load("oos"), load("paired")

    if paired_j and "E5" in paired_j:
        fig_paired(paired_j["E5"], out)
    if oos_j and "E4" in oos_j:
        fig_oos(oos_j["E4"], oos_j.get("config", {}), out)
    if main_j:
        if "E6" in main_j:
            fig_delta(main_j["E6"], out)
        if "E3" in main_j:
            fig_decarb(main_j["E3"], out)
        if "E2" in main_j:
            fig_incidence(main_j["E2"], out)

    seasonal = {}
    for k in ("winter_h8", "winter_h18", "summer_h8", "summer_h18"):
        j = load(k)
        if j and "E1" in j:
            seasonal[k] = j["E1"]
    fig_seasonal(seasonal, out)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
