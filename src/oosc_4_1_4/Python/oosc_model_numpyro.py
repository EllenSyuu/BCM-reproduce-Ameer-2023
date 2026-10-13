"""NumPyro port of the sibling Stan/oosc_cohort_model.stan
-- the out-of-school cohort model.

Chosen over CmdStan/cmdstanpy specifically to avoid needing a system C++
compiler (RTools on Windows): JAX/NumPyro JIT-compiles via XLA and needs no
separate toolchain install. See B. Documentation/4. Model Notes.md section 8
for the
Stan <-> NumPyro correspondence and a note on where the translation required
care (the `alpha` 2D smoothing terms, Eq. 11-13, are Stan `target +=` penalty
statements on a free parameter matrix, not a generative sampling chain --
ported here via `dist.ImproperUniform` + `numpyro.factor`, NumPyro's standard
idiom for this exact GMRF/CAR pattern).

Equation numbers in comments refer to the paper's Section 3 / Appendix B, same
as the Stan file. Keep the two in sync if either is edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import json
import os
import hashlib
from pathlib import Path
import pickle
import tempfile
import time
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import ndtr as Phi  # standard normal CDF
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS


@dataclass
class ModelData:
    """Container mirroring the Stan `data` block. All index arrays are
    0-indexed (Python convention), unlike the 1-indexed Stan version."""

    C: int
    R: int
    A_max: int
    N_EY: int
    mu_anchor_idx: int  # 0-indexed spline coefficient whose Greville year is 2000
    B1: np.ndarray  # (N_EY, N_K1)
    B5: np.ndarray  # (N_EY, N_K5)
    region_of_country: np.ndarray  # (C,) int, 0-indexed region id

    # administrative observations (Eq. 1)
    A_obs: np.ndarray
    A_country: np.ndarray
    A_region: np.ndarray
    A_age: np.ndarray       # 0-indexed relative age (0 = entry age)
    A_ey_idx: np.ndarray    # 0-indexed row into ey grid

    # survey observations (Eq. 2)
    K_obs: np.ndarray
    f_obs: np.ndarray
    nu: np.ndarray
    S_country: np.ndarray
    S_age: np.ndarray
    S_ey_idx: np.ndarray
    S_series: np.ndarray
    S_wave: np.ndarray
    N_SERIES: int
    N_WAVE: int


def model(data: ModelData) -> None:
    C, A_max = data.C, data.A_max
    N_K1, N_K5 = data.B1.shape[1], data.B5.shape[1]
    if not 0 < data.mu_anchor_idx < N_K1 - 1:
        raise ValueError("mu_anchor_idx must identify an interior annual spline coefficient.")
    B1 = jnp.asarray(data.B1)
    B5 = jnp.asarray(data.B5)

    # ---- mu: smoothed random walk with drift (Eq. 4-9) ----
    sigma_gamma_r = numpyro.sample("sigma_gamma_r", dist.HalfNormal(0.02).expand([data.R]))
    gamma_z = numpyro.sample("gamma_z", dist.HalfNormal(1.0).expand([C]))
    gamma_pos = numpyro.deterministic(
        "gamma_pos", sigma_gamma_r[data.region_of_country] * gamma_z
    )
    gamma = numpyro.deterministic("gamma", -gamma_pos)
    sigma_mu = numpyro.sample("sigma_mu", dist.HalfNormal(0.1))

    # Eq. 27 anchors the annual spline coefficient at calendar year 2000.
    mu_star_2000 = numpyro.sample("mu_star_2000", dist.Normal(0, 2).expand([C]))
    mu_eps_z = numpyro.sample(
        "mu_eps_z", dist.Normal(0, 1).expand([C, N_K1 - 1])
    )
    mu_eps = numpyro.deterministic("mu_eps", sigma_mu * mu_eps_z)
    before_eps = mu_eps[:, :data.mu_anchor_idx][:, ::-1]
    after_eps = mu_eps[:, data.mu_anchor_idx:]
    before = mu_star_2000[:, None] + jnp.cumsum(
        -gamma[:, None] - before_eps, axis=1
    )[:, ::-1]
    after = mu_star_2000[:, None] + jnp.cumsum(
        gamma[:, None] + after_eps, axis=1
    )
    mu_star = jnp.concatenate(
        [before, mu_star_2000[:, None], after], axis=1
    )  # (C, N_K1)
    numpyro.deterministic("mu_star", mu_star)

    # ---- alpha: 2D P-spline smoothing over age (g) and cohort spline (k) ----
    # Eq. 11-13 are Stan `target +=` penalty statements on a free (flat-prior)
    # parameter matrix, not a generative chain -- ImproperUniform + factor()
    # is the faithful NumPyro translation (see module docstring).
    tau_alpha1 = numpyro.sample("tau_alpha1", dist.HalfNormal(1.0))
    tau_alpha2 = numpyro.sample("tau_alpha2", dist.HalfNormal(1.0))

    alpha = numpyro.sample(
        "alpha",
        dist.ImproperUniform(
            support=constraints.real,
            batch_shape=(),
            event_shape=(C, A_max - 1, N_K5),
        ),
    )
    row_diff = alpha[:, 1:, :] - alpha[:, :-1, :]  # Eq. 11
    numpyro.factor("alpha_row_smooth", dist.Normal(0, 1 / jnp.sqrt(tau_alpha1)).log_prob(row_diff).sum())
    col_diff = alpha[:, :, 1:] - alpha[:, :, :-1]  # Eq. 12
    numpyro.factor("alpha_col_smooth", dist.Normal(0, 1 / jnp.sqrt(tau_alpha2)).log_prob(col_diff).sum())
    col_sums = alpha.sum(axis=1)  # Eq. 13, (C, N_K5)
    numpyro.factor("alpha_colsum", dist.Normal(0, 2.0).log_prob(col_sums).sum())

    cum_alpha = jnp.cumsum(alpha, axis=1)  # outer sum in Eq. 10, (C, A_max-1, N_K5)

    def observed_kappa(country, age, ey_idx):
        baseline = jnp.sum(mu_star[country] * B1[ey_idx], axis=-1)
        if A_max == 1:
            return baseline
        cohort_age = jnp.maximum(age - 1, 0)
        change = jnp.sum(
            cum_alpha[country, cohort_age] * B5[ey_idx], axis=-1
        )
        return baseline + jnp.where(age > 0, change, 0.0)

    # ---- survey bias (Sec. 3.3.1) ----
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(0.1).expand([data.N_SERIES]))
    beta_z = numpyro.sample(
        "beta_z", dist.Normal(0, 1).expand([C, data.N_SERIES]).to_event(0)
    )
    beta_raw = numpyro.deterministic("beta_raw", beta_z * sigma_beta[None, :])

    # ---- variance components (Sec. 3.3.2) ----
    sigma_sigma = numpyro.sample("sigma_sigma", dist.HalfNormal(0.01))
    sigma_adm_z = numpyro.sample("sigma_adm_z", dist.HalfNormal(1.0))
    omega_r_z = numpyro.sample("omega_r_z", dist.HalfNormal(1.0).expand([data.R]))
    eta_w_z = numpyro.sample("eta_w_z", dist.HalfNormal(1.0).expand([data.N_WAVE]))
    sigma_adm = numpyro.deterministic("sigma_adm", sigma_sigma * sigma_adm_z)
    omega_r = numpyro.deterministic("omega_r", sigma_sigma * omega_r_z)
    eta_w = numpyro.deterministic("eta_w", sigma_sigma * eta_w_z)

    # ---- likelihood: administrative (Eq. 1), real space ----
    if len(data.A_obs):
        mean_A = Phi(observed_kappa(
            data.A_country, data.A_age, data.A_ey_idx
        ))
        sd_A = jnp.sqrt(sigma_adm**2 + omega_r[data.A_region] ** 2)
        numpyro.sample("A_obs", dist.Normal(mean_A, sd_A), obs=jnp.asarray(data.A_obs))

    # ---- likelihood: survey (Eq. 2), probit space ----
    if len(data.K_obs):
        mean_S = (
            observed_kappa(data.S_country, data.S_age, data.S_ey_idx)
            + beta_raw[data.S_country, data.S_series]
        )
        sd_S = jnp.sqrt((data.nu**2 + eta_w[data.S_wave] ** 2) / data.f_obs**2)
        numpyro.sample("K_obs", dist.Normal(mean_S, sd_S), obs=jnp.asarray(data.K_obs))


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as f:
            temp_path = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(10):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(1)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except PermissionError:
                pass


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_write(path, buffer.getvalue())


def run_mcmc(
    data: ModelData,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    seed=0,
    thinning=1,
    progress_bar=True,
    target_accept_prob=0.99,
) -> MCMC:
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data=data)
    return mcmc


def run_mcmc_checkpointed(
    data: ModelData,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    seed=0,
    thinning=1,
    progress_bar=True,
    target_accept_prob=0.99,
    progress_path: Path | None = None,
    *,
    checkpoint_dir: Path,
    checkpoint_every=80,
    checkpoint_id="",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

    if checkpoint_every < thinning or checkpoint_every % thinning:
        raise ValueError("checkpoint_every must be a positive multiple of thinning.")
    if num_samples % thinning:
        raise ValueError("num_samples must be divisible by thinning for checkpointed sampling.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "checkpoint.json"
    draws_path = checkpoint_dir / "posterior_draws.npz"
    config = {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "seed": seed,
        "thinning": thinning,
        "target_accept_prob": target_accept_prob,
        "checkpoint_every": checkpoint_every,
        "checkpoint_id": checkpoint_id,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != config:
            raise ValueError(
                f"Checkpoint settings in {manifest_path} do not match this run; "
                "choose a new checkpoint directory or use the original settings."
            )
        completed_steps = int(manifest["completed_steps"])
        if completed_steps < 0 or completed_steps > num_samples:
            raise ValueError(f"Invalid completed_steps in {manifest_path}.")
    else:
        completed_steps = 0
        manifest = {"version": 1, "config": config, "completed_steps": 0, "status": "new"}

    def load_draws() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if draws_path.exists():
            with np.load(draws_path) as z:
                return (
                    {k[8:]: np.array(z[k]) for k in z.files if k.startswith("sample__")},
                    {k[7:]: np.array(z[k]) for k in z.files if k.startswith("extra__")},
                )
        sample_chunks: list[dict[str, np.ndarray]] = []
        extra_chunks: list[dict[str, np.ndarray]] = []
        for end in manifest.get("chunk_ends", []):
            chunk_path = checkpoint_dir / f"samples_{end:06d}.npz"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Missing completed posterior chunk: {chunk_path}")
            with np.load(chunk_path) as z:
                sample_chunks.append({k[8:]: np.array(z[k]) for k in z.files if k.startswith("sample__")})
                extra_chunks.append({k[7:]: np.array(z[k]) for k in z.files if k.startswith("extra__")})
        if not sample_chunks:
            return {}, {}
        names = sample_chunks[0]
        samples = {k: np.concatenate([chunk[k] for chunk in sample_chunks], axis=1) for k in names}
        extra_names = extra_chunks[0]
        extras = {k: np.concatenate([chunk[k] for chunk in extra_chunks], axis=1) for k in extra_names}
        return samples, extras

    def save_complete_draws(samples: dict[str, np.ndarray], extras: dict[str, np.ndarray]) -> None:
        all_arrays = {f"sample__{k}": v for k, v in samples.items()}
        all_arrays.update({f"extra__{k}": v for k, v in extras.items()})
        _save_npz(draws_path, all_arrays)

    samples, extras = load_draws()
    if completed_steps == num_samples:
        if not draws_path.exists():
            save_complete_draws(samples, extras)
        manifest["status"] = "sampling_complete"
        _atomic_write(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
        return samples, extras

    state_path = checkpoint_dir / f"state_{completed_steps:06d}.pkl"
    if completed_steps and not state_path.exists():
        raise FileNotFoundError(f"Missing MCMC state checkpoint: {state_path}")
    if (
        completed_steps == 0
        and manifest.get("status") not in ("new", "warming_up", "sampling")
    ):
        raise ValueError(f"Unexpected checkpoint state in {manifest_path}.")

    from numpyro import util

    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_lock = Lock()
    chain_progress = {
        i: {"completed": completed_steps, "total": num_samples}
        for i in range(num_chains)
    }
    original_tqdm_auto = util.tqdm_auto
    progress_status = "Sampling"
    progress_base = completed_steps
    progress_total = num_samples

    def write_progress(status: str | None = None) -> None:
        if progress_path is None:
            return
        with progress_lock:
            total = sum(item["total"] for item in chain_progress.values())
            completed = sum(item["completed"] for item in chain_progress.values())
            percent = round(100 * completed / total) if total else 0

            def bar(value: int) -> str:
                filled = value // 5
                return f"[{'#' * filled}{'-' * (20 - filled)}] {value:3d}%"

            lines = [
                f"Status: {status or progress_status}",
                f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
                f"Overall: {bar(percent)} ({completed:,}/{total:,} chain-iterations)",
            ]
            for chain, item in sorted(chain_progress.items()):
                chain_percent = round(100 * item["completed"] / item["total"]) if item["total"] else 0
                lines.append(
                    f"Chain {chain + 1}: {bar(chain_percent)} "
                    f"({item['completed']:,}/{item['total']:,})"
                )
            progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    class TrackedProgressBar:
        def __init__(self, *args, position=0, **kwargs):
            self._chain = int(position or 0)
            self._bar = original_tqdm_auto(*args, position=position, **kwargs)
            self._base = progress_base
            chain_progress[self._chain] = {
                "completed": min(self._base + int(self._bar.n), progress_total),
                "total": progress_total,
            }
            write_progress()

        def update(self, increment):
            result = self._bar.update(increment)
            chain_progress[self._chain]["completed"] = min(
                self._base + int(self._bar.n), progress_total
            )
            write_progress()
            return result

        def close(self):
            self._bar.close()
            chain_progress[self._chain]["completed"] = min(
                self._base + int(self._bar.n), progress_total
            )
            write_progress()

        def set_description(self, *args, **kwargs):
            return self._bar.set_description(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._bar, name)

    def run_with_progress(call, status: str, base: int, total: int) -> None:
        nonlocal progress_status, progress_base, progress_total
        progress_status = status
        progress_base = base
        progress_total = total
        for i in range(num_chains):
            chain_progress[i] = {"completed": base, "total": total}
        util.tqdm_auto = TrackedProgressBar
        write_progress(status)
        try:
            call()
            jax.effects_barrier()
        finally:
            util.tqdm_auto = original_tqdm_auto

    def save_state(path: Path, state) -> None:
        _atomic_write(path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    def save_manifest(status: str, completed: int, chunk_ends: list[int]) -> None:
        manifest.update(
            {"completed_steps": completed, "chunk_ends": chunk_ends, "status": status}
        )
        _atomic_write(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))

    try:
        if completed_steps == 0 and not state_path.exists():
            manifest["status"] = "warming_up"
            _atomic_write(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
            kernel = NUTS(model, target_accept_prob=target_accept_prob)
            first_chunk = min(checkpoint_every, num_samples)
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=first_chunk,
                num_chains=num_chains,
                thinning=thinning,
                progress_bar=progress_bar,
            )
            run_with_progress(
                lambda: mcmc.warmup(jax.random.PRNGKey(seed), data=data),
                "Warmup",
                0,
                num_warmup,
            )
            save_state(state_path, mcmc.post_warmup_state)
            save_manifest("sampling", 0, [])
        else:
            kernel = NUTS(model, target_accept_prob=target_accept_prob)
            first_chunk = min(checkpoint_every, num_samples - completed_steps)
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=first_chunk,
                num_chains=num_chains,
                thinning=thinning,
                progress_bar=progress_bar,
            )
            mcmc.post_warmup_state = pickle.loads(state_path.read_bytes())

        chunk_ends = list(manifest.get("chunk_ends", []))
        while completed_steps < num_samples:
            chunk_steps = min(checkpoint_every, num_samples - completed_steps)
            if mcmc.num_samples != chunk_steps:
                mcmc = MCMC(
                    NUTS(model, target_accept_prob=target_accept_prob),
                    num_warmup=num_warmup,
                    num_samples=chunk_steps,
                    num_chains=num_chains,
                    thinning=thinning,
                    progress_bar=progress_bar,
                )
                mcmc.post_warmup_state = pickle.loads(state_path.read_bytes())
            chunk_end = completed_steps + chunk_steps
            run_with_progress(
                lambda: mcmc.run(mcmc.post_warmup_state.rng_key, data=data),
                f"Sampling checkpoint {chunk_end:,}/{num_samples:,}",
                completed_steps,
                num_samples,
            )
            chunk_samples = {
                k: np.asarray(v)
                for k, v in mcmc.get_samples(group_by_chain=True).items()
            }
            chunk_extras = {
                k: np.asarray(v)
                for k, v in mcmc.get_extra_fields(group_by_chain=True).items()
            }
            chunk_arrays = {f"sample__{k}": v for k, v in chunk_samples.items()}
            chunk_arrays.update({f"extra__{k}": v for k, v in chunk_extras.items()})
            _save_npz(checkpoint_dir / f"samples_{chunk_end:06d}.npz", chunk_arrays)
            next_state_path = checkpoint_dir / f"state_{chunk_end:06d}.pkl"
            save_state(next_state_path, mcmc.last_state)
            chunk_ends.append(chunk_end)
            completed_steps = chunk_end
            state_path = next_state_path
            samples = {
                k: np.concatenate((samples[k], chunk_samples[k]), axis=1)
                if k in samples else chunk_samples[k]
                for k in chunk_samples
            }
            extras = {
                k: np.concatenate((extras[k], chunk_extras[k]), axis=1)
                if k in extras else chunk_extras[k]
                for k in chunk_extras
            }
            save_manifest("sampling", completed_steps, chunk_ends)
            mcmc.post_warmup_state = mcmc.last_state

        save_complete_draws(samples, extras)
        save_manifest("sampling_complete", completed_steps, chunk_ends)
        write_progress("Sampling complete")
        return samples, extras
    except Exception:
        write_progress("Failed; completed checkpoints are resumable")
        raise


# --- synthetic-data smoke test -----------------------------------------------
def _build_synthetic_data(seed=0) -> ModelData:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from spline_basis import build_model_bases, ey_index, mu_anchor_index

    rng = np.random.default_rng(seed)
    C, R, A_max = 3, 2, 6
    min_year, max_year = 2015, 2020
    bases = build_model_bases(min_year=2000, max_year=2022, max_age=A_max)
    ey_grid = bases.ey_grid

    region_of_country = np.array([0, 0, 1])  # 0-indexed

    # administrative observations: every country/year/age, true rate ~ declining with age noise
    A_country, A_region, A_age, A_year, A_obs = [], [], [], [], []
    for c in range(C):
        for y in range(min_year, max_year + 1):
            for a in range(A_max):  # 0-indexed age, a=0 is entry age
                true_rate = max(0.05, 0.5 - 0.05 * a - 0.01 * (y - min_year))
                obs = true_rate + rng.normal(0, 0.05)
                A_country.append(c)
                A_region.append(region_of_country[c])
                A_age.append(a)
                A_year.append(y)
                A_obs.append(obs)
    A_ey_idx = ey_index(np.array(A_year), np.array(A_age) + 1, ey_grid) - 1  # -1: Stan->0-indexed

    # a handful of survey observations
    N_S = 15
    S_country = rng.integers(0, C, N_S)
    S_age = rng.integers(0, A_max, N_S)
    S_year = rng.integers(min_year, max_year + 1, N_S)
    S_ey_idx = ey_index(S_year, S_age + 1, ey_grid) - 1
    N_SERIES, N_WAVE = 2, 3
    S_series = rng.integers(0, N_SERIES, N_S)
    S_wave = rng.integers(0, N_WAVE, N_S)
    true_S = 0.3 + rng.normal(0, 0.05, N_S)
    K_obs = np.clip(true_S, 0.01, 0.99)
    from scipy.stats import norm

    K_obs_probit = norm.ppf(K_obs)
    f_obs = norm.pdf(K_obs_probit)
    nu = np.full(N_S, 0.05)

    return ModelData(
        C=C, R=R, A_max=A_max, N_EY=len(ey_grid),
        mu_anchor_idx=mu_anchor_index(ey_grid),
        B1=bases.B1, B5=bases.B5, region_of_country=region_of_country,
        A_obs=np.array(A_obs), A_country=np.array(A_country), A_region=np.array(A_region),
        A_age=np.array(A_age), A_ey_idx=A_ey_idx,
        K_obs=K_obs_probit, f_obs=f_obs, nu=nu,
        S_country=S_country, S_age=S_age, S_ey_idx=S_ey_idx,
        S_series=S_series, S_wave=S_wave, N_SERIES=N_SERIES, N_WAVE=N_WAVE,
    )


if __name__ == "__main__":
    data = _build_synthetic_data()
    print(f"Synthetic data: C={data.C}, N_EY={data.N_EY}, N_A={len(data.A_obs)}, N_S={len(data.K_obs)}")
    print(f"B1 shape={data.B1.shape}, B5 shape={data.B5.shape}")

    mcmc = run_mcmc(data, num_warmup=200, num_samples=200, num_chains=2, seed=0)
    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    mu = np.asarray(samples["mu_star"]) @ data.B1.T
    theta = np.einsum(
        "scgk,ek->scge",
        np.cumsum(np.asarray(samples["alpha"]), axis=2),
        data.B5,
    )
    kappa = np.concatenate(
        [mu[:, :, None, :], mu[:, :, None, :] + theta], axis=2
    )
    output_rates = np.asarray(jax.scipy.special.ndtr(kappa))
    assert np.isfinite(output_rates).all(), "Non-finite posterior out-of-school rates"
    assert (output_rates >= 0).all() and (output_rates <= 1).all(), "Phi transform escaped [0,1]"
    print(f"\nrate range: [{output_rates.min():.4f}, {output_rates.max():.4f}] (expect within [0,1])")
    print("\nSmoke test passed: model compiles, samples, no NaNs, kappa in [0,1].")
