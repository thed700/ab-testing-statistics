"""
test_stats_utils.py
Unit tests for stats_utils module.

Run with: python -m pytest tests/ -v
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, "../src")
from stats_utils import cohens_d, run_ttest, run_chi_square, confidence_interval

np.random.seed(42)


class TestCohensD:
    def test_zero_difference(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_known_d(self):
        # groups with mean diff = 1, std ≈ 1 → d ≈ 1.0
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        d = cohens_d(b, a)
        assert d == pytest.approx(1.0, abs=0.05)

    def test_direction(self):
        a = np.random.normal(5, 1, 100)
        b = np.random.normal(3, 1, 100)
        assert cohens_d(a, b) > 0


class TestRunTTest:
    def test_significant(self):
        a = np.random.normal(10, 1, 200)
        b = np.random.normal(12, 1, 200)
        result = run_ttest(a, b)
        assert result["reject_h0"] is True
        assert "p_value" in result
        assert "ci_95" in result

    def test_not_significant(self):
        a = np.random.normal(10, 1, 50)
        b = np.random.normal(10.05, 1, 50)
        result = run_ttest(a, b)
        assert result["reject_h0"] is False

    def test_keys_present(self):
        a = np.random.normal(5, 1, 30)
        b = np.random.normal(6, 1, 30)
        result = run_ttest(a, b)
        for key in ["t_stat", "p_value", "reject_h0", "effect_size_d", "ci_95"]:
            assert key in result


class TestRunChiSquare:
    def test_independent(self):
        # Perfectly balanced table — should not reject H0
        observed = np.array([[50, 50], [50, 50]])
        result = run_chi_square(observed)
        assert result["reject_h0"] is False

    def test_dependent(self):
        # Strong association
        observed = np.array([[90, 10], [10, 90]])
        result = run_chi_square(observed)
        assert result["reject_h0"] is True
        assert result["cramers_v"] > 0.5


class TestConfidenceInterval:
    def test_output_is_tuple(self):
        data = np.random.normal(10, 2, 100)
        ci = confidence_interval(data)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_mean_inside_ci(self):
        data = np.random.normal(10, 2, 200)
        ci = confidence_interval(data)
        assert ci[0] < np.mean(data) < ci[1]

    def test_wider_ci_for_higher_confidence(self):
        data = np.random.normal(10, 2, 100)
        ci95 = confidence_interval(data, 0.95)
        ci99 = confidence_interval(data, 0.99)
        assert (ci99[1] - ci99[0]) > (ci95[1] - ci95[0])
