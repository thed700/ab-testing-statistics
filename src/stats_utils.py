"""
stats_utils.py
Reusable statistical functions for the hypothesis testing project.
"""

import numpy as np
from scipy import stats


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Parameters
    ----------
    group1, group2 : array-like
        Sample data for each group.
    
    Returns
    -------
    float
        Cohen's d value. Interpretation:
        0.2 = small, 0.5 = medium, 0.8 = large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_ttest(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run an independent samples t-test and return a summary dict.
    
    Parameters
    ----------
    group1, group2 : array-like
        Sample data for each group.
    alpha : float
        Significance level (default 0.05).
    
    Returns
    -------
    dict with keys: t_stat, p_value, reject_h0, effect_size, ci_95
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    d = cohens_d(group1, group2)

    # 95% confidence interval for the difference in means
    diff = np.mean(group1) - np.mean(group2)
    se = np.sqrt(np.var(group1, ddof=1) / len(group1) + np.var(group2, ddof=1) / len(group2))
    ci = stats.t.interval(0.95, df=len(group1) + len(group2) - 2, loc=diff, scale=se)

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "reject_h0": p_value < alpha,
        "effect_size_d": round(d, 4),
        "ci_95": (round(ci[0], 4), round(ci[1], 4)),
    }


def run_chi_square(observed: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run a chi-square test of independence on a contingency table.
    
    Parameters
    ----------
    observed : 2D array-like
        Contingency table of observed frequencies.
    alpha : float
        Significance level (default 0.05).
    
    Returns
    -------
    dict with keys: chi2, p_value, dof, reject_h0, cramers_v
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Cramér's V — effect size for chi-square
    n = np.sum(observed)
    min_dim = min(np.array(observed).shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "reject_h0": p_value < alpha,
        "cramers_v": round(cramers_v, 4),
    }


def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for the mean of a sample.
    
    Parameters
    ----------
    data : array-like
    confidence : float (default 0.95)
    
    Returns
    -------
    tuple : (lower, upper) bounds
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return (round(ci[0], 4), round(ci[1], 4))


def print_test_result(result: dict, test_name: str = "Statistical Test") -> None:
    """Pretty-print a test result dictionary."""
    print(f"\n{'='*50}")
    print(f"  {test_name}")
    print(f"{'='*50}")
    for key, value in result.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<22}: {value}")
    conclusion = "✅ Reject H₀ (statistically significant)" if result.get("reject_h0") else "❌ Fail to reject H₀ (not significant)"
    print(f"\n  Conclusion: {conclusion}")
    print(f"{'='*50}\n")
