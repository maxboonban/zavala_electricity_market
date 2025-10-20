import numpy as np
import matplotlib.pyplot as plt

def plot_ss_bar_with_errorlabels(
    stoch_ss, cvar_ss, det_ss,
    err="std",                      # "std" or "sem"
    title="Mean with error bars",
    decimals=3,                    # how many decimals to show
    savepath=None,
    show=False,
):
    groups = [np.asarray(stoch_ss, float),
              np.asarray(cvar_ss,  float),
              np.asarray(det_ss,   float)]
    labels = ["Stochastic", "CVaR", "Deterministic"]

    means = np.array([g.mean() for g in groups])
    if err.lower() == "sem":
        errs = np.array([g.std(ddof=1)/np.sqrt(len(g)) for g in groups])
        err_name = "SEM"
    else:
        errs = np.array([g.std(ddof=1) for g in groups])
        err_name = "SD"

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        x, means, yerr=errs, width=0.55,
        edgecolor="black", color="#88c", alpha=0.65,
        ecolor="black", error_kw={"elinewidth": 2, "capsize": 6},
        zorder=1
    )

    # annotate numeric error above each bar
    y_max = float(np.max(means + errs))
    y_min = float(np.min(np.concatenate([means - errs, [0]])))
    pad = 0.03 * (y_max - y_min if y_max > y_min else 1.0)
    for xi, (m, e) in enumerate(zip(means, errs)):
        ax.text(
            xi, m + e + pad,
            f"{err_name}={e:.{decimals}f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Social welfare")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax

def plot_det_ss_distribution(
    det_ss,
    bins=30,
    title="Deterministic E[SS] distribution across instances",
    savepath="visual_outputs/det_ss_distribution.png",
    show=False,
):
    """
    Plot the distribution of deterministic expected social surplus (E[SS])
    across instances with a vertical mean line. Saves to `savepath`.

    Parameters
    ----------
    det_ss : sequence of float
        E[SS] (positive social surplus) for the deterministic runs, one per instance.
    bins : int
        Number of histogram bins.
    title : str
        Plot title.
    savepath : str
        Output path for the saved image (directories created if needed).
    show : bool
        If True, display the figure in an interactive window.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    vals = np.asarray(det_ss, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(vals, bins=bins, edgecolor="black")
    mu = float(np.mean(vals)) if vals.size else float("nan")
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")

    ax.axvline(mu, linestyle="--", linewidth=2, label=f"mean = {mu:.3f}")
    ax.set_xlabel("Deterministic expected social surplus, E[SS]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()

    # ensure output directory exists and save
    outdir = os.path.dirname(savepath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return mu, sd, savepath