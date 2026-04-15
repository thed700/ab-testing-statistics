"""
visualizer.py
Visualization helpers for statistical hypothesis testing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Global style
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"control": "#4C72B0", "treatment": "#DD8452", "reject": "#e74c3c", "accept": "#2ecc71"}


def plot_distributions(group1, group2, labels=("Group A", "Group B"), title="Distribution Comparison", save_path=None):
    """Plot KDE + histogram for two groups side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, label, color in zip(axes, [group1, group2], labels, [COLORS["control"], COLORS["treatment"]]):
        ax.hist(data, bins=30, alpha=0.4, color=color, density=True, edgecolor="white")
        xd = np.linspace(data.min(), data.max(), 200)
        ax.plot(xd, stats.gaussian_kde(data)(xd), color=color, lw=2)
        ax.axvline(np.mean(data), color=color, linestyle="--", lw=1.5, label=f"Mean: {np.mean(data):.2f}")
        ax.set_title(f"{label} Distribution")
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pvalue(p_value, alpha=0.05, title="p-value Visualization", save_path=None):
    """Visualize the p-value against the significance threshold."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)
    ax.plot(x, y, "k-", lw=2)

    # Shade rejection regions
    crit = stats.norm.ppf(1 - alpha / 2)
    ax.fill_between(x, y, where=(x >= crit), alpha=0.3, color=COLORS["reject"], label=f"Rejection region (α={alpha})")
    ax.fill_between(x, y, where=(x <= -crit), alpha=0.3, color=COLORS["reject"])

    # Mark the test statistic implied by p_value
    z_obs = stats.norm.ppf(1 - p_value / 2)
    ax.axvline(z_obs, color="navy", linestyle="--", lw=2, label=f"Observed z ≈ {z_obs:.2f} (p={p_value:.4f})")
    ax.axvline(-z_obs, color="navy", linestyle="--", lw=2)

    color = COLORS["reject"] if p_value < alpha else COLORS["accept"]
    verdict = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"
    ax.set_title(f"{title}\n→ {verdict}", color=color, fontweight="bold")
    ax.legend()
    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confidence_interval(means, cis, labels, title="Confidence Intervals", save_path=None):
    """
    Plot confidence intervals for multiple groups.
    
    Parameters
    ----------
    means : list of floats
    cis   : list of (lower, upper) tuples
    labels: list of str
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS["control"], COLORS["treatment"]] * len(means)

    for i, (mean, ci, label, color) in enumerate(zip(means, cis, labels, colors)):
        ax.errorbar(i, mean, yerr=[[mean - ci[0]], [ci[1] - mean]],
                    fmt="o", color=color, capsize=8, capthick=2, markersize=10,
                    label=f"{label}: {mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylabel("Value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ab_test_results(control, treatment, alpha=0.05, save_path=None):
    """
    Full A/B test summary plot: distributions + p-value + CI.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Distribution comparison
    for data, label, color in zip([control, treatment], ["Control", "Treatment"],
                                   [COLORS["control"], COLORS["treatment"]]):
        axes[0].hist(data, bins=25, alpha=0.5, color=color, density=True, label=label)
        xd = np.linspace(min(data), max(data), 200)
        axes[0].plot(xd, stats.gaussian_kde(data)(xd), color=color, lw=2)
        axes[0].axvline(np.mean(data), color=color, linestyle="--", lw=1.5)
    axes[0].set_title("Distribution Comparison")
    axes[0].legend()

    # 2. Boxplot
    axes[1].boxplot([control, treatment], labels=["Control", "Treatment"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#AEC6CF", color="navy"),
                    medianprops=dict(color="red", lw=2))
    axes[1].set_title("Boxplot")

    # 3. Conversion rates with CI
    from scipy.stats import sem, t
    means = [np.mean(control), np.mean(treatment)]
    cis = [(np.mean(d) - t.ppf(0.975, len(d)-1)*sem(d),
            np.mean(d) + t.ppf(0.975, len(d)-1)*sem(d)) for d in [control, treatment]]

    for i, (mean, ci, color, label) in enumerate(zip(means, cis,
                                                      [COLORS["control"], COLORS["treatment"]],
                                                      ["Control", "Treatment"])):
        axes[2].bar(i, mean, color=color, alpha=0.7, width=0.4)
        axes[2].errorbar(i, mean, yerr=[[mean - ci[0]], [ci[1] - mean]],
                         fmt="none", color="black", capsize=8, capthick=2)

    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Control", "Treatment"])
    axes[2].set_title("Mean ± 95% CI")

    fig.suptitle("A/B Test Results Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
