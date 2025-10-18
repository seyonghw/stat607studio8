import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.lines import Line2D

SIGMA2, R2 = 1.0, 5.0

THEORY_COLOR       = "tab:blue"
SCATTER_COLOR      = "tab:orange"
LOWESS_LEFT_COLOR  = "tab:green"
LOWESS_RIGHT_COLOR = "tab:red"
SNR_COLOR          = "0.25"

Y_MIN, Y_MAX = -0.1, 10

################### helper functions ###############################

def _parse_snr_lines(s: str) -> list[float]:
    return [float(tok) for tok in s.split(",") if tok.strip()]


def _draw_snr_lines(ax, snr_vals: list[float], sigma2: float = 1.0, label=False):
    # If SNR = r^2/σ^2, asymptote level is r^2 = SNR * σ^2
    for i, snr in enumerate(snr_vals):
        y = snr * sigma2
        ax.axhline(y, ls=":", lw=1.0, alpha=0.6, label=(f"SNR = {snr}" if label else None))


def _gamma_to_splitx(gamma: np.ndarray) -> np.ndarray:
    g = np.asarray(gamma, float)
    t = np.empty_like(g, dtype=float)
    # left: [0.1, 1] → [0, 0.5]
    left = g <= 1.0
    if left.any():
        t[left] = (np.log10(g[left]) - np.log10(0.1)) / (np.log10(1.0) - np.log10(0.1)) * 0.5
    # right: (1, 10] → (0.5, 1]
    right = g > 1.0
    if right.any():
        t[right] = 0.5 + (np.log10(g[right]) - np.log10(1.0)) / (np.log10(10.0) - np.log10(1.0)) * 0.5
    return t


def _configure_split_xaxis(ax):
    # ticks at familiar gammas
    tick_gammas = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(_gamma_to_splitx(tick_gammas), [r"0.1", r"0.2", r"0.5", r"1", r"2", r"5", r"10"])
    # vertical asymptote at gamma=1 (t=0.5)
    ax.axvline(0.5, ls="--", alpha=0.6, lw=1.0)


def _plot_theory(ax):
    gL = np.logspace(-1, np.log10(0.999), 600)
    gR = np.logspace(np.log10(1.001), 1.0, 600)
    xL = _gamma_to_splitx(gL)
    xR = _gamma_to_splitx(gR)
    ax.plot(xL, theoretical_risk(gL), lw=2.0, alpha=0.95, color=THEORY_COLOR, label="Theory")
    ax.plot(xR, theoretical_risk(gR), lw=2.0, alpha=0.95, color=THEORY_COLOR, label="_nolegend_")



def _set_row_ylim(ax, y_arrays, pad=0.1):
    vals = []
    for arr in y_arrays:
        if arr is None: continue
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]
        if a.size: vals.append(a)
    if not vals: return
    allv = np.concatenate(vals)
    top = max(np.percentile(allv, 90), np.percentile(allv, 95))
    ax.set_ylim(-0.1, (1.0 + pad) * top)

#########################################################################

def make_row_axis(spec):
    fig = plt.gcf()
    ax = fig.add_subplot(spec)
    ax.set_box_aspect(1)  # square
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    return ax


def theoretical_risk(gamma: np.ndarray, sigma2: float = SIGMA2, r2: float = R2) -> np.ndarray:
    g = np.asarray(gamma, float)
    y = np.full_like(g, np.nan, float)
    left = g < 1.0
    right = g > 1.0
    y[left]  = sigma2 * (g[left] / (1.0 - g[left]))
    y[right] = r2 * (1.0 - 1.0/g[right]) + sigma2 * (1.0 / (g[right] - 1.0))
    return y


def plot_panel(spec, df_ns: pd.DataFrame, nsim: int, snr_lines: list[float], label_snr: bool):
    ax = make_row_axis(spec)
    _configure_split_xaxis(ax)
    _plot_theory(ax)

    xg = df_ns["gamma"].to_numpy()
    y  = df_ns["MSE_hat"].to_numpy()
    se = df_ns["se"].to_numpy()
    t  = _gamma_to_splitx(xg)

    if nsim == 1:
        ax.scatter(t, y, s=6, alpha=0.18, color=SCATTER_COLOR, edgecolors="none", label="Estimates")

        maskL, maskR = xg < 1.0, xg > 1.0

        if maskL.sum() >= 20:
            fitL = lowess(y[maskL], xg[maskL], frac=0.1, return_sorted=True)
            ax.plot(_gamma_to_splitx(fitL[:, 0]), fitL[:, 1],
                    lw=2.0, alpha=0.95, color=LOWESS_LEFT_COLOR, label="LOWESS")

        if maskR.sum() >= 20:
            fitR = lowess(y[maskR], xg[maskR], frac=0.1, return_sorted=True)
            ax.plot(_gamma_to_splitx(fitR[:, 0]), fitR[:, 1],
                    lw=2.0, alpha=0.95, color=LOWESS_LEFT_COLOR, label="_nolegend_")
    else:
        ax.errorbar(t, y, yerr=2.0*se, fmt="o", ms=3.5, alpha=0.9, lw=1.0, capsize=2,
                    color=SCATTER_COLOR, label="Estimates")


    _draw_snr_lines(ax, snr_lines, sigma2=1.0, label=label_snr)

    ax.set_ylabel(r"Risk")
    ax.set_title(f"{nsim} simulation replications per scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylim(Y_MIN, Y_MAX)  # common y-range

    return ax


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyze simulation results and reproduce Hastie et al. Fig. 2 style plot.")
    ap.add_argument("--snr-lines", type=str, default="5",
                    help='comma-separated SNR values for dashed horizontal lines (e.g. "1,2.33,3.66,5")')
    ap.add_argument("--in",  dest="in_path",  type=str, default="results.pkl",
                    help="Path to results .pkl produced by run_simulation.py")
    ap.add_argument("--out", dest="out_path", type=str, default="figure.pdf",
                    help="Output PDF filename")
    args = ap.parse_args(argv)
    snr_lines = _parse_snr_lines(args.snr_lines)

    df = pd.read_pickle(args.in_path)

    # sanity for nsim=1 assignment requirements
    df1 = df[df["nsim"] == 1].sort_values("gamma").copy()
    assert len(df1) == 5000, f"Expected 5000 scenarios for nsim=1, got {len(df1)}"
    g = df1["gamma"].to_numpy()
    assert np.isclose(g.min(), 0.1) and np.isclose(g.max(), 10.0), f"gamma range should be [0.1, 10], got [{g.min()}, {g.max()}]"
    assert np.all(np.diff(g) > 0), "gamma should be strictly increasing"

    # figure/layout tuned for paper: legend outside-left, tighter gaps
    fig = plt.figure(figsize=(13.5, 4.2), dpi=150)
    gs_outer = fig.add_gridspec(
        1, 3, wspace=0.12, top=0.88, bottom=0.18, left=0.20, right=0.98
    )

    axes = []
    for j, nsim in enumerate([1, 50, 1000]):
        df_ns = df[df["nsim"] == nsim].sort_values("gamma")
        ax = plot_panel(gs_outer[0, j], df_ns, nsim, snr_lines=snr_lines, label_snr=(j == 0))
        axes.append(ax)

    # hide duplicate y tick labels in middle/right panels
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    # panel letters
    for ax, letter in zip(axes, ["A", "B", "C"]):
        ax.text(0.02, 0.96, letter, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    # single proxy legend outside-left
    legend_elems = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=5, alpha=0.6,
               label='Estimates', markerfacecolor=SCATTER_COLOR, markeredgecolor='none'),
        Line2D([0], [0], color=THEORY_COLOR,       lw=2, label='Theory'),
        Line2D([0], [0], color=LOWESS_LEFT_COLOR,  lw=2, label='LOWESS'),
        Line2D([0], [0], color=SNR_COLOR,          lw=1, linestyle=':', label='SNR levels'),
    ]
    axes[0].legend(
        legend_elems, [h.get_label() for h in legend_elems],
        loc="upper right", bbox_to_anchor=(-0.15, 1.0),
        ncol=1, frameon=False, fontsize=9, borderaxespad=0.0
    )

    # title nudged higher
    plt.suptitle("Isotropic features", y=0.995, fontsize=13)

    plt.savefig(args.out_path, bbox_inches="tight")
    print(f"Saved {args.out_path}")

if __name__ == "__main__":
    main()


