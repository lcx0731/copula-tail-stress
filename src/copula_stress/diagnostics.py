"""A deliberately different model: separate t scales at zero correlation."""

import numpy as np
from scipy.stats import t

from .core import _integer, _real


def independent_scale_control(n, df=4, seed=1729):
    """Uniform marginals but independent coordinates, not a rho=0 t-copula."""
    _integer(n, 1, 1_000_000, "n")
    _integer(seed, 0, 2**32 - 1, "seed")
    _real(df, 3, 200, "df")
    rng = np.random.Generator(np.random.PCG64(seed))
    gaussian = rng.standard_normal((n, 2))
    scales = np.sqrt(rng.chisquare(df, (n, 2)) / df)
    return t.cdf(gaussian / scales, df)
