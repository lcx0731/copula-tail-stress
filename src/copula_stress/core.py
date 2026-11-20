"""Bivariate Gaussian/t copulas and fixed-quantile exceedance inference."""

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.stats import beta, norm, t


def _real(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass(frozen=True)
class Copula:
    family: str
    tau: float = 0.35
    df: float | None = None

    def __post_init__(self):
        if self.family not in ("gaussian", "student"):
            raise ValueError("family must be gaussian or student")
        _real(self.tau, -0.8, 0.8, "tau")
        if self.family == "student":
            _real(self.df, 3, 200, "df")
        elif self.df is not None:
            raise ValueError("Gaussian copula has no df parameter")

    @property
    def rho(self):
        return math.sin(math.pi * self.tau / 2)

    @property
    def upper_tail_limit(self):
        if self.family == "gaussian":
            return 0.0
        return float(2 * t.sf(math.sqrt((self.df + 1) * (1 - self.rho) / (1 + self.rho)), self.df + 1))


@dataclass(frozen=True)
class TailReference:
    q: float
    conditional_probability: float
    joint_probability: float
    quadrature_error_estimate: float
    upper_tail_limit: float


def sample_copula(model, n, seed=1729):
    """Generate iid synthetic rows; Student coordinates share one scale per row."""
    if not isinstance(model, Copula):
        raise ValueError("model must be a Copula")
    _integer(n, 1, 1_000_000, "n")
    _integer(seed, 0, 2**32 - 1, "seed")
    rng = np.random.Generator(np.random.PCG64(seed))
    independent = rng.standard_normal((n, 2))
    latent = np.column_stack((independent[:, 0], model.rho * independent[:, 0]
                             + math.sqrt(1 - model.rho**2) * independent[:, 1]))
    if model.family == "student":
        scale = np.sqrt(rng.chisquare(model.df, n) / model.df)
        latent /= scale[:, None]
        return t.cdf(latent, model.df)
    return norm.cdf(latent)


def tail_reference(model, q):
    """Numerical P(V>q | U>q) via conditional latent distributions.

    Integrate over a uniform tail coordinate, using survival quantiles to avoid
    subtracting very small probabilities from one inside the integrand.
    The returned integration error estimate is not a rigorous error certificate.
    """
    if not isinstance(model, Copula):
        raise ValueError("model must be a Copula")
    _real(q, 0.5, 0.9999, "q")
    rho = model.rho
    tail = 1 - q
    distribution = norm if model.family == "gaussian" else t(model.df)
    threshold = float(distribution.isf(tail))

    def integrand(position):
        probability = tail * (1 - position)
        if probability == 0:
            if model.family == "gaussian":
                return 1.0 if rho > 0 else (0.0 if rho < 0 else tail)
            return float(t.sf(-rho * math.sqrt((model.df + 1) / (1 - rho**2)), model.df + 1))
        latent = float(distribution.isf(probability))
        if model.family == "gaussian":
            return float(norm.sf((threshold - rho * latent) / math.sqrt(1 - rho**2)))
        scale = math.hypot(math.sqrt(model.df), latent) * math.sqrt((1 - rho**2) / (model.df + 1))
        return float(t.sf((threshold - rho * latent) / scale, model.df + 1))

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            value, error = quad(integrand, 0, 1, epsabs=5e-10, epsrel=1e-9, limit=200)
        except IntegrationWarning as warning:
            raise ArithmeticError("tail quadrature did not converge") from warning
    if not math.isfinite(value) or not 0 <= value <= 1 or not math.isfinite(error):
        raise ArithmeticError("invalid tail quadrature result")
    return TailReference(float(q), float(value), float(tail * value), float(error), model.upper_tail_limit)


def binomial_interval(successes, trials, alpha=0.05):
    """Equal-tailed Clopper-Pearson interval; no trials gives [0,1]."""
    _integer(trials, 0, 1_000_000, "trials")
    _integer(successes, 0, trials, "successes")
    _real(alpha, 1e-6, 0.5, "alpha")
    if trials == 0:
        return (0.0, 1.0)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.isf(alpha / 2, successes + 1, trials - successes))
    return lower, upper


@dataclass(frozen=True)
class TailEstimate:
    q: float
    samples: int
    conditioning_count: int
    joint_count: int
    conditional_estimate: float | None
    conditional_interval: tuple
    joint_estimate: float
    joint_interval: tuple
    alpha: float


def estimate_tail(uniforms, q, alpha=0.05):
    """For iid rows and an externally fixed q, K | M is binomial.

    The interval targets finite-q conditional probability, not its q->1 limit.
    It is pointwise and does not cover fitting or threshold selection uncertainty.
    """
    _real(q, 0.5, 0.9999, "q")
    _real(alpha, 1e-6, 0.5, "alpha")
    values = np.asarray(uniforms)
    if values.dtype.kind not in "fiu" or values.ndim != 2 or values.shape[1] != 2 or not 1 <= len(values) <= 1_000_000:
        raise ValueError("uniforms must be a numeric (n,2) array with n in 1..1000000")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("copula coordinates must lie in [0,1]")
    first = values[:, 0] > q
    m = int(np.count_nonzero(first))
    k = int(np.count_nonzero(first & (values[:, 1] > q)))
    return TailEstimate(float(q), len(values), m, k, k / m if m else None,
                        binomial_interval(k, m, alpha), k / len(values),
                        binomial_interval(k, len(values), alpha), float(alpha))
