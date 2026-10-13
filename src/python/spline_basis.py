"""Spline basis construction for the out-of-school cohort model.

Builds the B1 (mu, annual knots) and B5 (alpha/Theta, 5-year knots) basis
matrices referenced in the sibling Stan/oosc_cohort_model.stan
and documented in docs/model-notes.md section 4.

Choices made here (cubic B-spline, knot spacing, boundary handling) are not
fully pinned down by the paper -- see docs/model-notes.md before
treating results as a faithful reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


def _bspline_design_matrix(x: np.ndarray, interior_knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Build a B-spline design (basis) matrix, degree `degree`, over `x`.

    Uses clamped boundary knots (each boundary repeated degree+1 times), the
    same convention R's splines2::bSpline(..., intercept = TRUE) uses, so the
    two implementations should match given identical interior knots.
    """
    boundary = (x.min(), x.max())
    knots = np.concatenate([
        np.repeat(boundary[0], degree + 1),
        interior_knots,
        np.repeat(boundary[1], degree + 1),
    ])
    n_basis = len(knots) - degree - 1
    basis = np.empty((len(x), n_basis))
    for i in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        basis[:, i] = BSpline(knots, coeff, degree, extrapolate=False)(x)
    return np.nan_to_num(basis)  # endpoints can evaluate to nan; true value is 0 or 1 there


def spline_control_years(ey_grid: np.ndarray, knot_spacing: int, degree: int = 3) -> np.ndarray:
    """Return the Greville abscissa (year) associated with each spline coefficient."""
    ey_grid = np.asarray(ey_grid)
    assert np.all(np.diff(ey_grid) == 1), "ey_grid must be sorted, consecutive integers"
    interior_knots = np.arange(
        ey_grid.min() + knot_spacing,
        ey_grid.max() - knot_spacing + 1,
        knot_spacing,
    )
    interior_knots = interior_knots[(interior_knots > ey_grid.min()) & (interior_knots < ey_grid.max())]
    knots = np.concatenate([
        np.repeat(ey_grid.min(), degree + 1),
        interior_knots,
        np.repeat(ey_grid.max(), degree + 1),
    ]).astype(float)
    return np.array([
        knots[i + 1:i + degree + 1].mean()
        for i in range(len(knots) - degree - 1)
    ])


def mu_anchor_index(ey_grid: np.ndarray) -> int:
    """Find the interior annual spline coefficient anchored by the paper at 2000."""
    control_years = spline_control_years(ey_grid, knot_spacing=1)
    matches = np.flatnonzero(np.isclose(control_years, 2000))
    if len(matches) != 1 or matches[0] == 0 or matches[0] == len(control_years) - 1:
        raise ValueError(
            "The annual spline must have exactly one interior control coefficient "
            "at calendar year 2000."
        )
    return int(matches[0])


def build_spline_basis(ey_grid: np.ndarray, knot_spacing: int, degree: int = 3) -> np.ndarray:
    """Build a cubic B-spline basis over an entry-year grid.

    Parameters
    ----------
    ey_grid : sorted, consecutive integer array of entry years, e.g. 1981..2022.
    knot_spacing : years between interior knots (1 for mu, 5 for alpha/Theta).
    degree : spline degree (paper unspecified; default cubic).

    Returns
    -------
    ndarray of shape (len(ey_grid), n_basis_functions).
    """
    ey_grid = np.asarray(ey_grid)
    assert np.all(np.diff(ey_grid) == 1), "ey_grid must be sorted, consecutive integers"

    interior_knots = np.arange(
        ey_grid.min() + knot_spacing,
        ey_grid.max() - knot_spacing + 1,
        knot_spacing,
    )
    interior_knots = interior_knots[(interior_knots > ey_grid.min()) & (interior_knots < ey_grid.max())]

    return _bspline_design_matrix(ey_grid.astype(float), interior_knots.astype(float), degree)


@dataclass
class ModelBases:
    ey_grid: np.ndarray
    B1: np.ndarray
    B5: np.ndarray


def build_model_bases(min_year: int, max_year: int, max_age: int) -> ModelBases:
    """Build both basis matrices needed by the Stan model given the panel's
    implied entry-year range.

    Parameters
    ----------
    min_year : earliest observation year across the harmonized panel (e.g. 2000).
    max_year : latest observation year (e.g. 2022).
    max_age : max relative age index (A_max in the Stan model).
    """
    # entry_year = year - age + 1 (age 1-indexed, a=1 is entry age -- see ey_index()).
    # Oldest entry year occurs at the smallest year and largest age (a=max_age):
    ey_min = min_year - max_age + 1
    ey_max = max_year                        # youngest possible entry year (age 1 in max_year)
    ey_grid = np.arange(ey_min, ey_max + 1)

    return ModelBases(
        ey_grid=ey_grid,
        B1=build_spline_basis(ey_grid, knot_spacing=1),
        B5=build_spline_basis(ey_grid, knot_spacing=5),
    )


def ey_index(year: np.ndarray, age: np.ndarray, ey_grid: np.ndarray) -> np.ndarray:
    """Map (year, age) pairs to their 1-indexed row in the entry-year grid
    (1-indexed to match Stan's array indexing directly)."""
    year = np.asarray(year)
    age = np.asarray(age)
    entry_year = year - age + 1

    ey_min = ey_grid.min()
    idx0 = entry_year - ey_min  # 0-indexed
    if np.any((idx0 < 0) | (idx0 >= len(ey_grid))):
        raise ValueError(
            "Some (year, age) combinations fall outside ey_grid -- widen "
            "min_year/max_age in build_model_bases()."
        )
    return idx0 + 1  # 1-indexed for Stan


if __name__ == "__main__":
    # smoke test
    min_year, max_year, max_age = 2000, 2022, 19
    bases = build_model_bases(min_year=min_year, max_year=max_year, max_age=max_age)
    assert bases.B1.shape[0] == len(bases.ey_grid)
    assert bases.B5.shape[0] == len(bases.ey_grid)

    # Boundary cases -- these are the ones an off-by-one in ey_min/ey_max
    # actually breaks; a middle-of-range check (e.g. age=12 in 2015) will not
    # catch it, which is exactly how an earlier version of build_model_bases
    # shipped with a real off-by-one bug undetected.
    idx_oldest = ey_index(np.array([min_year]), np.array([max_age]), bases.ey_grid)
    idx_youngest = ey_index(np.array([max_year]), np.array([1]), bases.ey_grid)
    assert idx_oldest[0] == 1, f"oldest cohort (min_year, max_age) should map to grid row 1, got {idx_oldest[0]}"
    assert idx_youngest[0] == len(bases.ey_grid), (
        f"youngest cohort (max_year, age=1) should map to the last grid row, got {idx_youngest[0]}"
    )

    idx = ey_index(np.array([2015]), np.array([12]), bases.ey_grid)
    print("ey_grid range:", bases.ey_grid.min(), "-", bases.ey_grid.max())
    print("B1 shape:", bases.B1.shape, " B5 shape:", bases.B5.shape)
    print("example ey_index(2015, age=12):", idx)
    print("boundary checks passed: oldest ->", idx_oldest[0], " youngest ->", idx_youngest[0])
