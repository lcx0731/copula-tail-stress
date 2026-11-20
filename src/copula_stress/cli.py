"""Audit one synthetic copula at a fixed population quantile."""

import argparse
from dataclasses import asdict
import json
import math

from scipy.stats import kendalltau

from .core import Copula, estimate_tail, sample_copula, tail_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("gaussian", "student"), default="student")
    parser.add_argument("--tau", type=float, default=0.35)
    parser.add_argument("--df", type=float)
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        df = 4 if args.family == "student" and args.df is None else args.df
        model = Copula(args.family, args.tau, df)
        reference = tail_reference(model, args.q)
        sample = sample_copula(model, args.n, args.seed)
        observed = float(kendalltau(sample[:, 0], sample[:, 1]).statistic) if args.n > 1 else None
        result = {"use": "Educational only; not investment advice. Deterministic seeded synthetic data.",
                  "model": asdict(model), "latent_rho": model.rho, "seed": args.seed,
                  "sample_kendall_tau": observed if observed is not None and math.isfinite(observed) else None,
                  "reference": asdict(reference), "estimate": asdict(estimate_tail(sample, args.q)),
                  "interval_scope": "Pointwise iid Monte Carlo uncertainty for fixed q; not parameter, model, or asymptotic-limit uncertainty"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
