import numpy as np
import matplotlib.pyplot as plt

def plot_ss_bar_with_errorlabels(
    stoch_ss, cvar_ss, det_ss,
    err="std",                      # kept for API compatibility (unused)
    title="Mean with error bars",
    decimals=3,
    savepath=None,
    show=False,
):
    groups = [np.asarray(stoch_ss, float),
              np.asarray(cvar_ss,  float),
              np.asarray(det_ss,   float)]
    labels = ["Stochastic", "CVaR", "Deterministic"]

    means = np.array([g.mean() for g in groups])

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # ---- NO ERROR BARS ----
    ax.bar(
        x, means, width=0.55,
        edgecolor="black", color="#88c", alpha=0.65,
        zorder=1
    )

    # zoom y-axis around means (no errors)
    y_low, y_high = float(np.min(means)), float(np.max(means))
    span = y_high - y_low
    if span <= 0:
        # all means equal — give a small visual span
        span = max(1.0, 0.01 * max(abs(y_high), 1.0))
    ax.set_ylim(y_low - 0.15*span, y_high + 0.20*span)

    # annotate mean only
    span_for_text = ax.get_ylim()[1] - ax.get_ylim()[0]
    text_pad = 0.03 * span_for_text
    for xi, m in enumerate(means):
        ax.text(
            xi, m + text_pad,
            f"mean={m:.{decimals}f}",
            ha="center", va="bottom", fontsize=10, clip_on=True
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Social Surplus")
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_tail_welfare_means_with_errorbars(stoch_tail_welfare,
                                           cvar_tail_welfare,
                                           det_tail_welfare,
                                           err="std",            # kept for API compatibility (unused)
                                           show=False,
                                           save=True,
                                           outdir="visual_outputs",
                                           filename="tail_welfare_means.png"):
    """
    Plot mean tail welfare for Stochastic, CVaR, and Deterministic.
    NOTE: Error bars are intentionally NOT drawn.
    """
    series = [
        np.asarray(stoch_tail_welfare, dtype=float),
        np.asarray(cvar_tail_welfare, dtype=float),
        np.asarray(det_tail_welfare, dtype=float),
    ]
    labels = ["Stochastic", "CVaR", "Deterministic"]

    means = np.array([x.mean() if x.size else np.nan for x in series])

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    x = np.arange(len(labels))

    # ---- NO ERROR BARS ---- (use same blue tone)
    ax.bar(x, means, edgecolor="black", color="tab:blue", alpha=0.65)

    # zoom y-axis around means
    y_low, y_high = float(np.nanmin(means)), float(np.nanmax(means))
    span = y_high - y_low
    if not np.isfinite(span) or span <= 0:
        span = max(1.0, 0.01 * max(abs(y_high), 1.0))
    pad = 0.15 * span
    ax.set_ylim(y_low - pad, y_high + pad)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean tail social surplus")
    ax.grid(axis="y", alpha=0.2)

    # annotate mean only
    span_for_text = ax.get_ylim()[1] - ax.get_ylim()[0]
    text_pad = 0.03 * span_for_text
    for i, m in enumerate(means):
        ax.text(i, m + text_pad, f"mean={m:.2f}",
                ha="center", va="bottom", fontsize=9, clip_on=True)

    fig.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[plot_tail_welfare_means_with_errorbars] saved to: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rt_price_histograms(
    z_prices,
    cvar_prices,
    bins=30,
    show=False,
    savepath=None,
    title_left="Histogram of Π(ω) — Stochastic",
    title_right="Histogram of Π(ω) — CVaR"
):
    """
    Side-by-side histograms comparing real-time prices from the stochastic run (z_prices)
    and from the CVaR run (cvar_prices). Accepts 1-D arrays (flattened across all scenarios/instances).
    """
    z = np.asarray(z_prices, dtype=float).ravel()
    c = np.asarray(cvar_prices, dtype=float).ravel()

    # Common bin edges for fair comparison
    vmin = float(np.nanmin([z.min(), c.min()])) if z.size and c.size else 0.0
    vmax = float(np.nanmax([z.max(), c.max()])) if z.size and c.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    edges = np.linspace(vmin, vmax, int(bins) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    n1, _, _ = ax1.hist(z, bins=edges, alpha=0.7, edgecolor='black')
    ax1.set_title(title_left, fontsize=14)
    ax1.set_xlabel('Price ($/MWh)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)

    n2, _, _ = ax2.hist(c, bins=edges, alpha=0.7, edgecolor='black', color='tab:red')
    ax2.set_title(title_right, fontsize=14)
    ax2.set_xlabel('Price ($/MWh)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Match y-lims
    y_max = max(float(np.nanmax(n1)) if n1.size else 0.0,
                float(np.nanmax(n2)) if n2.size else 0.0)
    ax1.set_ylim(0, y_max * 1.05 if y_max > 0 else 1)
    ax2.set_ylim(ax1.get_ylim())

    # Stats boxes
    stats_text1 = f"Mean: {np.nanmean(z):.2f}\nStd: {np.nanstd(z):.2f}\nMin: {np.nanmin(z):.2f}\nMax: {np.nanmax(z):.2f}"
    stats_text2 = f"Mean: {np.nanmean(c):.2f}\nStd: {np.nanstd(c):.2f}\nMin: {np.nanmin(c):.2f}\nMax: {np.nanmax(c):.2f}"
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, fontsize=10,
             va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10,
             va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2)

# === Overlay plot: E[SS] means and tail-welfare means on same axes ===

def plot_ss_and_tail_overlay(
    stoch_ss, cvar_ss, det_ss,
    stoch_tail_welfare, cvar_tail_welfare, det_tail_welfare,
    title="E[SS] vs Tail Welfare (overlay)",
    decimals=2,
    savepath=None,
    show=False,
    tail_alpha=0.35
):
    """
    Overlay bar chart: solid bars show mean E[SS]; translucent bars (narrower) show mean tail welfare
    at the same x positions. No error bars; annotations show mean values only.
    """
    labels = ["Stochastic", "CVaR", "Deterministic"]

    ss_groups   = [np.asarray(stoch_ss, float),
                   np.asarray(cvar_ss,  float),
                   np.asarray(det_ss,   float)]
    tail_groups = [np.asarray(stoch_tail_welfare, float),
                   np.asarray(cvar_tail_welfare,  float),
                   np.asarray(det_tail_welfare,   float)]

    ss_means   = np.array([g.mean() if g.size else np.nan for g in ss_groups])
    tail_means = np.array([g.mean() if g.size else np.nan for g in tail_groups])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Base bars: E[SS] (solid)
    width_base = 0.55
    bars_ss = ax.bar(
        x, ss_means, width=width_base,
        edgecolor="black", color="#88c", alpha=0.90, zorder=1, label="E[SS] mean"
    )

    # Overlay bars: tail welfare (narrower & translucent)
    width_overlay = 0.35
    bars_tail = ax.bar(
        x, tail_means, width=width_overlay,
        edgecolor="black", color="tab:orange", alpha=tail_alpha, zorder=2, label="Tail welfare mean"
    )

    # Axis cosmetics
    all_means = np.concatenate([ss_means, tail_means])
    y_low  = float(np.nanmin(all_means))
    y_high = float(np.nanmax(all_means))
    span   = y_high - y_low if np.isfinite(y_high - y_low) and (y_high > y_low) else 1.0
    ax.set_ylim(y_low - 0.15*span, y_high + 0.25*span)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Social Surplus")
    # ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)

    # Mean labels (no SD), offset a bit above each bar
    pad = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for xi, m in enumerate(ss_means):
        if np.isfinite(m):
            ax.text(xi, m + pad, f"{m:.{decimals}f}", ha="center", va="bottom", fontsize=10)
    for xi, m in enumerate(tail_means):
        if np.isfinite(m):
            ax.text(xi, m + 0.5*pad, f"{m:.{decimals}f}", ha="center", va="bottom", fontsize=9, color="tab:orange")

    ax.legend(frameon=True)

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax