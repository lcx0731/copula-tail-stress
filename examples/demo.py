"""Reproduce a fixed-seed, synthetic copula stress comparison; no market data."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from copula_stress import Copula, estimate_tail, sample_copula, tail_reference
from copula_stress.diagnostics import independent_scale_control


N = 200_000
SEED = 1729
QUANTILES = [.9, .95, .975, .99, .995, .999, .9999]
MODELS = {
    "Gaussian": Copula("gaussian", .35),
    "Student t (df=12)": Copula("student", .35, 12),
    "Student t (df=4)": Copula("student", .35, 4),
}
COLORS = ["#3976a8", "#c8792e", "#258879"]


def summarize(values):
    return {
        "sample_kendall_tau": float(kendalltau(values[:, 0], values[:, 1]).statistic),
        "marginal_means": values.mean(axis=0).tolist(),
        "marginal_exceedance_at_099": (values > .99).mean(axis=0).tolist(),
    }


def measured_point(ax, estimate, position, color, vertical=False):
    center = estimate["conditional_estimate"]
    lo, hi = estimate["conditional_interval"]
    # No conditional observations: show [0,1] without inventing a point estimate.
    if center is None:
        if vertical:
            ax.vlines(position, lo, hi, color=color, alpha=.6)
        else:
            ax.hlines(position, lo, hi, color=color, alpha=.6)
        return
    errors = np.array([[center - lo], [hi - center]])
    if vertical:
        ax.errorbar(position, center, yerr=errors, fmt="o", color=color, capsize=4, ms=5)
    else:
        ax.errorbar(center, position, xerr=errors, fmt="o", color=color, capsize=4, ms=6)


def plot_results(metrics, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "figure.facecolor": "#fafbf9", "axes.facecolor": "#fafbf9"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4))
    fig.subplots_adjust(top=.85, bottom=.13, left=.09, right=.96, hspace=.55, wspace=.30)
    fig.suptitle("Same rank correlation. Different joint extremes.",
                 x=.07, y=.97, ha="left", fontsize=22, weight="bold")
    fig.text(.07, .915, "Synthetic bivariate copulas  |  200,000 iid pairs per model  |  PCG64 seed 1729",
             fontsize=11, color="#52606b")

    ax = axes[0, 0]
    for (name, item), color in zip(metrics["models"].items(), COLORS):
        x = [-np.log10(1-row["q"]) for row in item["reference_curve"]]
        y = [row["conditional_probability"] for row in item["reference_curve"]]
        ax.plot(x, y, color=color, lw=2, label=name)
        ax.axhline(item["upper_tail_limit"], color=color, ls=":", lw=1.5)
    ax.set(title="A  Finite thresholds and asymptotic limits", ylabel="P(V > q | U > q)")
    ax.set_xticks([1, 2, 3, 4], ["0.90", "0.99", "0.999", "0.9999"])
    ax.set_xlabel("Population quantile q (tail probability on log scale)")
    ax.set_ylim(-.015, .5)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.text(.03, .06, "All models: Kendall tau = 0.35\nDotted lines: limits as q approaches 1",
            transform=ax.transAxes, fontsize=9, color="#52606b")

    ax = axes[0, 1]
    names = list(metrics["models"])
    for index, (name, color) in enumerate(zip(names, COLORS)):
        row = next(row for row in metrics["models"][name]["thresholds"] if row["estimate"]["q"] == .99)
        estimate, reference = row["estimate"], row["reference"]
        measured_point(ax, estimate, index, color)
        ax.scatter(reference["conditional_probability"], index + .14,
                   marker="D", s=32, color="#202d35", zorder=3)
        ax.text(.48, index, f"{estimate['joint_count']:,} / {estimate['conditioning_count']:,}",
                ha="right", va="center", fontsize=9, color="#52606b")
    ax.set_yticks(range(3), names, fontsize=9)
    ax.set(xlim=(0, .5), ylim=(-.55, 2.7), title="B  Simulation check at q = 0.99",
           xlabel="Conditional probability; labels show K / M")
    ax.text(.02, .95, "Dots + bars: MC + 95% interval   |   Diamonds: quadrature",
            transform=ax.transAxes, fontsize=8.5, va="top", color="#52606b")

    ax = axes[1, 0]
    item = metrics["models"]["Student t (df=4)"]
    for row in item["thresholds"]:
        estimate = row["estimate"]
        position = -np.log10(1-estimate["q"])
        measured_point(ax, estimate, position, COLORS[2], vertical=True)
        ax.text(position, .59, f"{estimate['conditioning_count']:,}", ha="center", fontsize=8)
    x = [-np.log10(1-row["q"]) for row in item["reference_curve"]]
    ax.plot(x, [row["conditional_probability"] for row in item["reference_curve"]],
            color="#202d35", lw=1.3, label="Quadrature reference")
    ax.axhline(item["upper_tail_limit"], color=COLORS[2], ls=":", lw=1.5)
    ax.set(title="C  More extreme thresholds, fewer observations", ylabel="P(V > q | U > q)",
           ylim=(0, .65), xlim=(.85, 4.15), xlabel="Population quantile q; Student t (df=4)")
    ax.set_xticks([1, 2, 3, 4], ["0.90", "0.99", "0.999", "0.9999"])
    ax.text(.02, .98, "Numbers: conditioning count M", transform=ax.transAxes,
            fontsize=8.5, va="top", color="#52606b")
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    ax = axes[1, 1]
    labels = ["Separate scales\n(independent control)", "Shared scale\n(Student t copula)"]
    for index, key in enumerate(["separate", "shared"]):
        row = metrics["zero_rank_control"][key]
        estimate = row["estimate"]
        measured_point(ax, estimate, index, COLORS[index * 2])
        ax.scatter(row["reference_conditional_probability"], index + .13,
                   marker="D", s=32, color="#202d35", zorder=3)
        ax.text(.145, index + .2,
                f"sample tau = {row['sample_kendall_tau']:.4f}\nK/M = {estimate['joint_count']:,}/{estimate['conditioning_count']:,}",
                fontsize=9, va="center", color="#52606b")
    ax.set_yticks(range(2), labels, fontsize=9)
    ax.set(xlim=(0, .27), ylim=(-.55, 1.65), title="D  A shared scale changes the model",
           xlabel="P(V > 0.99 | U > 0.99); both have population tau = 0")
    ax.text(.02, .95, "df = 4  |  Uniform marginals in both constructions",
            transform=ax.transAxes, fontsize=8.5, va="top", color="#52606b")
    for ax in axes.flat:
        ax.grid(axis="x", alpha=.12)
        ax.set_axisbelow(True)
    fig.text(.07, .05,
             "Intervals quantify pointwise iid simulation uncertainty at fixed q; they do not cover model choice or fitted parameters.\n"
             "Synthetic educational experiment only. Not investment advice. A zero tail limit does not imply finite-threshold independence.",
             fontsize=9, color="#52606b", linespacing=1.6)
    fig.savefig(output / "tail_stress.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curve_q = [float(1-10**(-power)) for power in np.linspace(1, 4, 25)]
    metrics = {
        "data": "Deterministic seeded synthetic data; no external data or fitted parameters.",
        "use": "Educational only; not investment advice.",
        "n_per_model": N, "seed": SEED, "generator": "NumPy PCG64", "alpha": .05,
        "interval_scope": "Pointwise iid Monte Carlo uncertainty at fixed population q; not uncertainty in the asymptotic limit, fitted parameters, or model choice.",
        "coupling": "Models reuse a seed and therefore are not independent of one another. No interval for a between-model difference is constructed.",
        "models": {},
    }
    for name, model in MODELS.items():
        values = sample_copula(model, N, SEED)
        metrics["models"][name] = {
            "configuration": asdict(model), "latent_rho": model.rho,
            "upper_tail_limit": model.upper_tail_limit, **summarize(values),
            "thresholds": [{"reference": asdict(tail_reference(model, q)),
                            "estimate": asdict(estimate_tail(values, q))} for q in QUANTILES],
            "reference_curve": [asdict(tail_reference(model, q)) for q in curve_q],
        }
    shared_model = Copula("student", 0, 4)
    shared = sample_copula(shared_model, N, SEED)
    separate = independent_scale_control(N, 4, SEED)
    metrics["zero_rank_control"] = {
        "q": .99, "df": 4, "population_kendall_tau": 0,
        "shared": {**summarize(shared), "estimate": asdict(estimate_tail(shared, .99)),
                   "reference_conditional_probability": tail_reference(shared_model, .99).conditional_probability,
                   "upper_tail_limit": shared_model.upper_tail_limit},
        "separate": {**summarize(separate), "estimate": asdict(estimate_tail(separate, .99)),
                     "reference_conditional_probability": 1-.99, "upper_tail_limit": 0.0},
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    plot_results(metrics, args.output)
    for name, item in metrics["models"].items():
        row = next(row for row in item["thresholds"] if row["estimate"]["q"] == .99)
        ref, est = row["reference"], row["estimate"]
        print(f"{name:18} tau={item['sample_kendall_tau']:.5f} "
              f"chi(.99)={ref['conditional_probability']:.6f} "
              f"MC={est['conditional_estimate']:.6f} "
              f"K/M={est['joint_count']}/{est['conditioning_count']}")
    print(f"Wrote metrics.json and tail_stress.png to {args.output}")


if __name__ == "__main__":
    main()
