import json
import math
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import kendalltau

from copula_stress import Copula, estimate_tail, sample_copula, tail_reference
from copula_stress.diagnostics import independent_scale_control


@pytest.mark.parametrize("model", [Copula("gaussian", 0.35), Copula("student", 0.35, 4)])
def test_sampler_marginals_rank_and_joint_tail_against_known_model(model):
    n = 120000
    values = sample_copula(model, n, 12)
    assert values.shape == (n, 2) and np.all((values >= 0) & (values <= 1))
    assert np.max(np.abs(values.mean(axis=0) - .5)) < .005
    assert np.max(np.abs(np.mean(values <= .95, axis=0) - .95)) < .004
    assert abs(kendalltau(values[:, 0], values[:, 1]).statistic - model.tau) < .015
    p = tail_reference(model, .95).joint_probability
    observed = estimate_tail(values, .95).joint_estimate
    # Broad seeded sanity check, not a claim that every 95% interval must cover.
    assert abs(observed - p) < 8 * math.sqrt(p * (1-p) / n) + 1 / n
    np.testing.assert_array_equal(sample_copula(model, 100, 3), sample_copula(model, 100, 3))


def test_shared_scale_is_required_even_when_rho_is_zero():
    shared = sample_copula(Copula("student", 0, 4), 150000, 84)
    independent = independent_scale_control(150000, 4, 84)
    assert abs(kendalltau(shared[:, 0], shared[:, 1]).statistic) < .015
    assert abs(kendalltau(independent[:, 0], independent[:, 1]).statistic) < .015
    assert estimate_tail(shared, .95).conditional_estimate - estimate_tail(independent, .95).conditional_estimate > .04
    assert np.max(np.abs(independent.mean(axis=0) - .5)) < .005


@pytest.mark.parametrize("args", [("other",), ("student", .3), ("gaussian", .3, 4),
                                 ("student", .9, 4), ("student", .3, 2), ("student", .3, 201),
                                 ("student", True, 4), ("student", float("nan"), 4)])
def test_invalid_models(args):
    with pytest.raises(ValueError):
        Copula(*args)


def test_invalid_samples_and_quantiles():
    model = Copula("gaussian")
    for n in (0, 1000001, True, 1.5):
        with pytest.raises(ValueError):
            sample_copula(model, n)
    for q in (.49, 1, float("inf"), True):
        with pytest.raises(ValueError):
            tail_reference(model, q)
    for values in ([], [[1]], [[.3, float("nan")]], [[-.01, .5]], [[True, False]], [[1j, 0]]):
        with pytest.raises(ValueError):
            estimate_tail(values, .9)


def test_installed_cli_and_invalid_configuration(tmp_path):
    result = subprocess.run([sys.executable, "-m", "copula_stress.cli", "--n", "2000", "--q", "0.95"],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data["estimate"]["samples"] == 2000
    assert 0 < data["reference"]["conditional_probability"] < 1
    result = subprocess.run([sys.executable, "-m", "copula_stress.cli", "--family", "gaussian", "--df", "4"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 2 and "error:" in result.stderr
