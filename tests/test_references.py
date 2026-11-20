import math

import pytest
from scipy.integrate import quad
from scipy.stats import norm, t

from copula_stress import Copula, tail_reference


@pytest.mark.parametrize("tau", [-0.6, 0, 0.35, 0.7])
@pytest.mark.parametrize("family,df", [("gaussian", None), ("student", 3), ("student", 8)])
def test_median_quadrant_identity(tau, family, df):
    model = Copula(family, tau, df)
    result = tail_reference(model, 0.5)
    # Exact elliptic sign identity, independent of the integration formula.
    assert result.conditional_probability == pytest.approx((1 + tau) / 2, abs=2e-9)
    assert result.joint_probability == pytest.approx((1 + tau) / 4, abs=1e-9)


@pytest.mark.parametrize("q", [0.5, 0.9, 0.99, 0.9999])
def test_gaussian_independence(q):
    result = tail_reference(Copula("gaussian", 0), q)
    assert result.conditional_probability == pytest.approx(1 - q, abs=1e-13)
    assert result.joint_probability == pytest.approx((1 - q) ** 2, abs=1e-14)
    assert result.upper_tail_limit == 0


@pytest.mark.parametrize("model", [Copula("gaussian", 0.35), Copula("student", 0.35, 4), Copula("student", -0.2, 9)])
def test_tail_coordinate_integral_matches_latent_space_integral(model):
    q = 0.975
    rho = model.rho
    if model.family == "gaussian":
        threshold = norm.ppf(q)
        integrand = lambda x: norm.pdf(x) * norm.sf((threshold - rho * x) / math.sqrt(1 - rho**2))
    else:
        threshold = t.ppf(q, model.df)
        integrand = lambda x: t.pdf(x, model.df) * t.sf((threshold - rho * x) / math.sqrt((model.df + x*x) * (1-rho*rho)/(model.df+1)), model.df+1)
    joint, error = quad(integrand, threshold, math.inf, epsabs=1e-12, epsrel=1e-10)
    reference = tail_reference(model, q)
    assert reference.joint_probability == pytest.approx(joint, abs=1e-10)
    assert reference.conditional_probability == pytest.approx(joint / (1-q), abs=1e-8)


def test_student_limit_anchor_and_finite_threshold_distinction():
    # For rho=0, nu=3: 2*sf_t4(2) has this elementary value.
    model = Copula("student", 0, 3)
    assert model.upper_tail_limit == pytest.approx(1 - 5 / (4 * math.sqrt(2)), abs=1e-14)
    moderate = tail_reference(model, 0.9).conditional_probability
    far = tail_reference(model, 0.9999).conditional_probability
    assert abs(far - model.upper_tail_limit) < 0.003
    assert moderate - model.upper_tail_limit > 0.03
    assert tail_reference(Copula("gaussian", 0.35), 0.99).conditional_probability > 0.05


def test_same_tau_different_tail_limits_and_boundary_cases():
    gaussian, student = Copula("gaussian", 0.35), Copula("student", 0.35, 4)
    assert gaussian.rho == student.rho
    assert gaussian.upper_tail_limit == 0 < student.upper_tail_limit
    assert Copula("student", -0.5, 4).upper_tail_limit > 0
    for model in (Copula("student", 0.8, 3), Copula("student", -0.8, 200), Copula("gaussian", 0.8)):
        result = tail_reference(model, 0.9999)
        assert 0 <= result.conditional_probability <= 1
        assert result.quadrature_error_estimate >= 0
