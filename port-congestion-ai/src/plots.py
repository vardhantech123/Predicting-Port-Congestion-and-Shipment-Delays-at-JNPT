"""Report figures (static PNGs). Colour-blind-safe two-hue scheme: blue + orange."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from . import config as C
from .explain import explain_row

BLUE, ORANGE, GREY, INK = "#2b6cb0", "#d9822b", "#8a8f98", "#222222"
NAMES = {"long_stay": "Long-stay risk", "excess": "Excess-delay risk"}
COL = {"long_stay": BLUE, "excess": ORANGE}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150})


def _save(fig, name):
    fig.tight_layout(); fig.savefig(C.FIG_DIR / name, dpi=200); plt.close(fig)


def fig_label(df, ft, base):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ok = ft["excess"] > C.DELAY_THRESHOLD_H
    ax[0].scatter(ft.teu[~ok], ft.tat[~ok], s=9, c=GREY, alpha=.6, label="normal")
    ax[0].scatter(ft.teu[ok], ft.tat[ok], s=11, c=ORANGE, label="excess > 4 h")
    xs = np.linspace(200, 9000, 200)
    import pandas as pd
    ax[0].plot(xs, base.predict(pd.DataFrame({"teu": xs, "loa": base.loa_med})), c=BLUE, lw=2,
               label="expected TAT (workload)")
    ax[0].axhline(base.long_stay_h, c=BLUE, ls=":", lw=1)
    ax[0].text(8800, base.long_stay_h + .6, f"long-stay > {base.long_stay_h:.0f} h", ha="right",
               color=BLUE, fontsize=8)
    ax[0].set_xlim(0, 9000); ax[0].set_xlabel("TEU handled"); ax[0].set_ylabel("Turnaround (h)")
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[0].set_title("Turnaround grows with workload", loc="left", fontsize=9.5)
    ax[1].hist(ft["excess"], bins=35, color=GREY, alpha=.8)
    ax[1].axvline(C.DELAY_THRESHOLD_H, c=ORANGE, lw=2)
    ax[1].text(C.DELAY_THRESHOLD_H + .5, ax[1].get_ylim()[1] * .85, "+4 h = delayed", color=ORANGE, fontsize=8)
    ax[1].set_xlabel("Excess turnaround vs workload-expected (h)"); ax[1].set_ylabel("Calls")
    ax[1].set_title("Excess-delay label", loc="left", fontsize=9.5)
    _save(fig, "fig1_label_definition.png")


def fig_perf(preds, metrics):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1, 1.5]})
    for k, p in preds.items():
        fpr, tpr, _ = roc_curve(p.y_true, p.prob_delay)
        ax[0].plot(fpr, tpr, c=COL[k], lw=2, label=f"{NAMES[k]} (AUC {metrics[k]['auc']:.2f})")
    ax[0].plot([0, 1], [0, 1], c=GREY, ls="--", lw=1)
    ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("True-positive rate")
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[0].set_title("Held-out last 30% of period", loc="left", fontsize=9.5)
    variants = list(metrics["long_stay"]["cv"])
    y = np.arange(len(variants)); h = .38
    for i, k in enumerate(["long_stay", "excess"]):
        v = [metrics[k]["cv"][m]["auc"] for m in variants]
        ax[1].barh(y + (i - .5) * h, v, h * .92, color=COL[k], label=NAMES[k])
        for yy, vv in zip(y + (i - .5) * h, v):
            ax[1].text(vv + .008, yy, f"{vv:.2f}", va="center", fontsize=7)
    ax[1].axvline(.5, c=GREY, ls="--", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels(variants, fontsize=7.5); ax[1].invert_yaxis()
    ax[1].set_xlim(.4, 1); ax[1].set_xlabel("AUC, pooled expanding-window CV")
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[1].set_title("Model comparison & ablations", loc="left", fontsize=9.5)
    _save(fig, "fig2_performance.png")


def fig_drivers(metrics):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for a, k in zip(ax, ["long_stay", "excess"]):
        imp = metrics[k]["perm_importance"]
        items = sorted(imp.items(), key=lambda t: t[1], reverse=True)[:8]
        a.barh([C.FEATURE_LABELS[f] for f, _ in items], [v for _, v in items], color=COL[k])
        a.invert_yaxis(); a.axvline(0, c=GREY, lw=1)
        a.tick_params(axis="y", labelsize=7.5)
        a.set_xlabel("AUC drop when feature is shuffled (test)")
        a.set_title(NAMES[k], loc="left", fontsize=9.5)
    _save(fig, "fig3_drivers.png")


def fig_calibration(metrics):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.9))
    for a, k in zip(ax, ["long_stay", "excess"]):
        cal = metrics[k]["calibration"]
        a.plot([0, .8], [0, .8], c=GREY, ls="--", lw=1)
        a.plot([c["mean_pred"] for c in cal], [c["observed"] for c in cal], "o-", c=COL[k],
               lw=2, ms=6, mec="white")
        for c in cal:
            a.text(c["mean_pred"], c["observed"] + .035, f"n={c['n']}", ha="center", fontsize=7)
        a.set_xlim(0, .8); a.set_ylim(0, .8)
        a.set_xlabel("Predicted probability (test quartile mean)"); a.set_ylabel("Observed rate")
        a.set_title(NAMES[k] + " - calibration", loc="left", fontsize=9.5)
    _save(fig, "fig4_calibration.png")


def fig_example(models, ft, preds):
    """Waterfall-style explanation for the highest-risk held-out long-stay call."""
    m = models["long_stay"]
    te = ft.iloc[len(ft) - len(preds["long_stay"]):]
    i = int(np.argmax(preds["long_stay"].prob_delay.to_numpy()))
    X = te.iloc[[i]]
    contrib = m.contributions(X).iloc[0].sort_values()
    contrib = contrib[contrib.abs() > .03]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh([C.FEATURE_LABELS[f] for f in contrib.index], contrib.values,
            color=[ORANGE if v > 0 else BLUE for v in contrib.values])
    ax.axvline(0, c=GREY, lw=1)
    p = preds["long_stay"].iloc[i]
    ax.set_xlabel("Contribution to log-odds vs average call (orange = raises risk)")
    ax.set_title(f"Example: {X.term.iloc[0]} call, {X.teu.iloc[0]:,.0f} TEU - P(long stay) = "
                 f"{p.prob_delay:.0%}, confidence {p.confidence:.0%}", loc="left", fontsize=9)
    ax.tick_params(axis="y", labelsize=7.5)
    _save(fig, "fig5_example_explanation.png")


def make_all(df, ft, base, models, preds, metrics):
    fig_label(df, ft, base); fig_perf(preds, metrics); fig_drivers(metrics)
    fig_calibration(metrics); fig_example(models, ft, preds)
