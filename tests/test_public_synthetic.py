"""Public-safe regression tests using synthetic inputs only.

Run with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    ROOT / "src" / "python",
    ROOT / "D. Code" / "D.1. Paper 1 - OOSC Cohort Model (4.1.4)" / "Python",
)
for source_dir in SOURCE_DIRS:
    if (source_dir / "oosc_model_numpyro.py").is_file():
        sys.path.insert(0, str(source_dir))
        break
else:
    raise ImportError("Could not find the model's Python source directory.")

from numpyro.infer.util import log_density

from oosc_model_numpyro import ModelData, model
from spline_basis import build_spline_basis, ey_index, mu_anchor_index


class PublicSyntheticTests(unittest.TestCase):
    def test_spline_basis_is_nonnegative_and_sums_to_one(self) -> None:
        years = np.arange(1998, 2003)

        for basis in (
            build_spline_basis(years, knot_spacing=1),
            build_spline_basis(years, knot_spacing=5),
        ):
            with self.subTest(columns=basis.shape[1]):
                self.assertEqual(basis.shape[0], years.size)
                self.assertTrue(np.all(np.isfinite(basis)))
                self.assertTrue(np.all(basis >= 0))
                np.testing.assert_allclose(basis.sum(axis=1), 1.0)

    def test_anchor_and_entry_year_indices_match_model_conventions(self) -> None:
        years = np.arange(1998, 2003)
        anchor = mu_anchor_index(years)
        self.assertGreater(anchor, 0)
        self.assertLess(anchor, build_spline_basis(years, knot_spacing=1).shape[1] - 1)

        indices = ey_index(
            np.array([1998, 2000, 2002]),
            np.array([1, 1, 1]),
            years,
        )
        np.testing.assert_array_equal(indices, np.array([1, 3, 5]))

    def test_model_has_finite_joint_density_for_synthetic_observations(self) -> None:
        years = np.arange(1998, 2003)
        basis1 = build_spline_basis(years, knot_spacing=1)
        basis5 = build_spline_basis(years, knot_spacing=5)
        data = ModelData(
            C=1,
            R=1,
            A_max=2,
            N_EY=len(years),
            mu_anchor_idx=mu_anchor_index(years),
            B1=basis1,
            B5=basis5,
            region_of_country=np.array([0]),
            A_obs=np.array([0.2]),
            A_country=np.array([0]),
            A_region=np.array([0]),
            A_age=np.array([0]),
            A_ey_idx=np.array([0]),
            K_obs=np.array([0.1]),
            f_obs=np.array([0.3]),
            nu=np.array([0.05]),
            S_country=np.array([0]),
            S_age=np.array([1]),
            S_ey_idx=np.array([4]),
            S_series=np.array([0]),
            S_wave=np.array([0]),
            N_SERIES=1,
            N_WAVE=1,
        )
        initial_values = {
            "sigma_gamma_r": np.array([0.02]),
            "gamma_z": np.array([0.5]),
            "sigma_mu": np.array(0.1),
            "mu_star_2000": np.array([0.0]),
            "mu_eps_z": np.zeros((1, basis1.shape[1] - 1)),
            "tau_alpha1": np.array(1.0),
            "tau_alpha2": np.array(1.0),
            "alpha": np.zeros((1, data.A_max - 1, basis5.shape[1])),
            "sigma_beta": np.array([0.1]),
            "beta_z": np.zeros((1, data.N_SERIES)),
            "sigma_sigma": np.array(0.01),
            "sigma_adm_z": np.array(1.0),
            "omega_r_z": np.array([1.0]),
            "eta_w_z": np.array([1.0]),
        }

        joint_log_density, trace = log_density(model, (data,), {}, initial_values)

        self.assertTrue(np.isfinite(float(joint_log_density)))
        self.assertTrue(trace["A_obs"]["is_observed"])
        self.assertTrue(trace["K_obs"]["is_observed"])


if __name__ == "__main__":
    unittest.main()
