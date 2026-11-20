import math

import numpy as np
import pytest
from scipy.stats import binomtest

from copula_stress import binomial_interval, estimate_tail


@pytest.mark.parametrize("k,n", [(0, 12), (12, 12), (1, 12), (4, 12), (31, 70)])
def test_beta_quantiles_match_independent_binomial_tail_inversion(k, n):
    expected = binomtest(k, n).proportion_ci(method="exact")
    lo, hi = binomial_interval(k, n)
    assert lo == pytest.approx(expected.low, abs=1e-12)
    assert hi == pytest.approx(expected.high, abs=1e-12)


def test_finite_sample_coverage_by_exact_enumeration():
    n = 12
    intervals = [binomial_interval(k, n) for k in range(n + 1)]
    for p in [0, .001, .01, .1, .3, .5, .8, .99, 1]:
        coverage = sum(math.comb(n, k) * p**k * (1-p)**(n-k)
                       for k, (lo, hi) in enumerate(intervals) if lo <= p <= hi)
        assert coverage >= .95 - 1e-12


def test_random_conditioning_count_has_valid_conditional_coverage():
    n, first_probability, conditional_probability = 10, 0.07, 0.4
    coverage = 0
    for m in range(n + 1):
        p_m = math.comb(n, m) * first_probability**m * (1-first_probability)**(n-m)
        for k in range(m + 1):
            lo, hi = binomial_interval(k, m)
            if lo <= conditional_probability <= hi:
                coverage += p_m * math.comb(m, k) * conditional_probability**k * (1-conditional_probability)**(m-k)
    assert coverage >= .95 - 1e-12


def test_strict_exceedance_counts_and_zero_trials():
    values = np.array([[.9, .9], [.9, .2], [.5, .9], [1, .8], [.7, 1]])
    result = estimate_tail(values, .8)
    assert result.conditioning_count == 3 and result.joint_count == 1
    assert result.conditional_estimate == 1 / 3 and result.joint_estimate == 1 / 5
    empty_tail = estimate_tail([[.2, .9], [.3, .1]], .99)
    assert empty_tail.conditional_estimate is None
    assert empty_tail.conditional_interval == (0, 1)
    assert empty_tail.joint_interval[1] > 0
    assert binomial_interval(0, 0) == (0, 1)


@pytest.mark.parametrize("args", [(-1, 10), (11, 10), (True, 10), (1, 0), (0, 1000001), (1, 10, 0), (1, 10, float("nan"))])
def test_invalid_interval_arguments(args):
    with pytest.raises(ValueError):
        binomial_interval(*args)
