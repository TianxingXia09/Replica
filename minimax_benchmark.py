from __future__ import annotations

from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SIZES = range(1, 7)
TOTAL_COPIES = [100, 300, 1000, 3000, 10000, 30000]


def variance_bound(k: int, *, pure_product: bool = False) -> Fraction:
    if k < 1:
        raise ValueError("Subsystem size must be positive.")
    return Fraction(3**k - 1) if pure_product else Fraction(4 * 3**k - 3, 4)


def sufficient_copy_budget(k: int, *, pure_product: bool = False) -> int:
    gap = 1 - Fraction(1, 2**k)
    threshold = 4 * variance_bound(k, pure_product=pure_product) / gap**2
    # Strict inequality in the integer number of pairs, including odd budgets.
    return 2 * (threshold.numerator // threshold.denominator + 1)


def main():
    rows = []
    for k in SIZES:
        gap = 1 - 2 ** (-k)
        one_copy_lower = gap / 2
        for copies in TOTAL_COPIES:
            pairs = copies // 2
            paired_worst_rmse = np.sqrt(float(variance_bound(k)) / pairs)
            rows.append(
                {
                    "subsystem_size": k,
                    "total_copies": copies,
                    "one_copy_minimax_abs_error_lower_bound": one_copy_lower,
                    "paired_local_worst_case_rmse_upper_bound": paired_worst_rmse,
                    "paired_pure_product_exact_rmse": np.sqrt((3**k - 1) / pairs),
                    "upper_over_lower": paired_worst_rmse / one_copy_lower,
                }
            )
    out = ROOT / "minimax_results"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "matched_copy_bounds.csv", index=False)
    crossings = []
    for k in SIZES:
        threshold = sufficient_copy_budget(k)
        crossings.append(
            {
                "subsystem_size": k,
                "sufficient_total_copies_for_paired_upper_below_one_copy_lower": threshold,
                "pure_product_total_copies_below_one_copy_lower": sufficient_copy_budget(k, pure_product=True),
            }
        )
    pd.DataFrame(crossings).to_csv(out / "crossover.csv", index=False)
    print(pd.DataFrame(crossings).to_string(index=False))


if __name__ == "__main__":
    main()
