"""
Global sensitivity analysis.
"""

from typing import Callable, NamedTuple
import numpy as np
from scipy.stats import qmc, rankdata
import jax
import jax.numpy as jnp

from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

from models.parameters import Params
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.metrics import outcome_metrics


class SensitivityResults(NamedTuple):
    param_names: list[str]
    bounds: dict[str, tuple[float, float]]
    samples: np.ndarray       # (N, d)
    outputs: np.ndarray       # (N,)
    prcc_mean: np.ndarray     # (d,)
    prcc_lower: np.ndarray    # (d,)
    prcc_upper: np.ndarray    # (d,)
    prcc_samples: np.ndarray  # (n_bootstrap, d)
    sobol_S1: np.ndarray      # (d,)
    sobol_S1_conf: np.ndarray # (d,)
    sobol_ST: np.ndarray      # (d,)
    sobol_ST_conf: np.ndarray # (d,)

_PATHOGEN_BOUNDS: dict[str, tuple[float, float]] = {
    "R_0":       (1.0, 5.0),
    "gamma_inv": (0.1, 10.0),
    "mu_s_inv":  (0.1, 10.0),
}
_PRESYMPTOMATIC_BOUNDS: dict[str, tuple[float, float]] = {
    "sigma_inv": (0.1, 10.0),
}
_ASYMPTOMATIC_BOUNDS: dict[str, tuple[float, float]] = {
    "mu_a_inv": (0.1, 10.0),
    "p":        (0.0, 1.0),
    "phi":      (0.0, 1.0),
}
_INTERVENTION_BOUNDS: dict[str, tuple[float, float]] = {
    "epsilon_s": (0.0, 1.0),
    "epsilon_w": (0.0, 1.0),
    "tau_W":     (1.0, 30.0),
    "tau_B":     (1.0, 30.0),
    "log_k":     (0.0, 3.0),
    "R_crit":    (0.8, 1.5),
}
_THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    "log_k_I":    (1.0, 4.0),
    "log_I_crit": (-4.0, -2.0),
}

def _parameter_bounds(model: Callable, scenario: str = "start") -> dict[str, tuple[float, float]]:
    """Return parameter bound dictionary for sensitivity analysis."""
    MODEL_NAMES: dict[Callable, str] = {simulate_SEIPAR_W: "SEIPAR_W", simulate_SEIAR_W: "SEIAR_W", simulate_SEIR_W: "SEIR_W",}
    name = MODEL_NAMES[model]
    bounds: dict[str, tuple[float, float]] = dict(_PATHOGEN_BOUNDS)
    if name == "SEIPAR_W": bounds |= _PRESYMPTOMATIC_BOUNDS
    if name in ("SEIPAR_W", "SEIAR_W"): bounds |= _ASYMPTOMATIC_BOUNDS
    bounds |= _INTERVENTION_BOUNDS
    if scenario == "threshold": bounds |= _THRESHOLD_BOUNDS
    elif scenario != "start": raise ValueError(f"unknown scenario: {scenario!r}; expected 'start' or 'threshold'")
    return bounds

def _symmetric_log10_bounds(value: float, frac: float = 0.2, default: float = 1.0):
    v = value if (np.isfinite(value) and value > 0.0) else default
    return (np.log10(v * (1.0-frac)), np.log10(v * (1.0+frac)))

def _parameter_bounds_around_mean(model: Callable, scenario: str = "start", mean_params: Params = None) -> dict[str, tuple[float, float]]:
    MODEL_NAMES: dict[Callable, str] = {simulate_SEIPAR_W: "SEIPAR_W", simulate_SEIAR_W: "SEIAR_W", simulate_SEIR_W: "SEIR_W",}
    name = MODEL_NAMES[model]
    if mean_params is not None:
        ps = mean_params
    else:
        ps = {"SEIPAR_W": Params.for_SEIPAR, "SEIAR_W": Params.for_SEIAR, "SEIR_W": Params.for_SEIR}[name]()
    bounds: dict[str, tuple[float, float]] = dict({
        "R_0":       (ps.R_0*0.8, ps.R_0*1.2),
        "gamma_inv": (ps.gamma_inv*0.8, ps.gamma_inv*1.2),
        "mu_s_inv":  (ps.mu_s_inv*0.8, ps.mu_s_inv*1.2),
    })
    if name == "SEIPAR_W": bounds |= {
        "sigma_inv": (ps.sigma_inv*0.8, ps.sigma_inv*1.2),
    }
    if name in ("SEIPAR_W", "SEIAR_W"): bounds |= {
        "mu_a_inv": (ps.mu_a_inv*0.8, ps.mu_a_inv*1.2),
        "p":        (ps.p*0.8, ps.p*1.2),
        "phi":      (ps.phi*0.8, ps.phi*1.2),
    }
    bounds |= {
        "epsilon_s": (ps.epsilon_s*0.8, ps.epsilon_s*1.2),
        "epsilon_w": (ps.epsilon_w*0.8, ps.epsilon_w*1.2),
        "tau_W":     (ps.tau_W*0.8, ps.tau_W*1.2),
        "tau_B":     (ps.tau_B*0.8, ps.tau_B*1.2),
        "log_k":     _symmetric_log10_bounds(ps.k),
        "R_crit":    (ps.R_crit*0.8, ps.R_crit*1.2),
    }
    if scenario == "threshold":
        bounds |= {
            "log_k_I":    _symmetric_log10_bounds(ps.k_I),
            "log_I_crit": _symmetric_log10_bounds(ps.I_crit, default=1e-4),
        }
    elif scenario != "start": raise ValueError(f"unknown scenario: {scenario!r}; expected 'start' or 'threshold'")
    return bounds

# PRCC
def _construct_latin_hypercube(bounds: dict, n: int, seed: int | None = None) -> np.ndarray:
    """Quasi Monte Carlo sampling from Latin hypercube."""
    names = list(bounds)
    try:
        latin_hypercube = qmc.scale(
            sample = qmc.LatinHypercube(d=len(names), seed=seed).random(n=n), 
            l_bounds = np.array([bounds[p][0] for p in names]),
            u_bounds = np.array([bounds[p][1] for p in names]),
        )
    except Exception as e: raise ValueError(f"{e}\nbounds={bounds}")
    return latin_hypercube

def _partial_rank_corr_coeff(latin_hypercube, y_output):
    """Compute the partial rank correlation coefficients between model parameters and output."""
    ranked_data = np.hstack((np.apply_along_axis(rankdata, 0, latin_hypercube), rankdata(y_output).reshape(-1, 1)))
    C = np.corrcoef(ranked_data, rowvar=False) 
    W = np.linalg.inv(C) # precision matrix
    prcc = np.array([
        -W[i, -1] / np.sqrt(W[i, i] * W[-1, -1]) # -Wxy / sqrt(Wxx * Wyy) for all params x and output y
            for i in range(latin_hypercube.shape[1])])
    return prcc

def _prcc_bootstrap(X: np.ndarray, y: np.ndarray, n_bootstrap: int = 100, seed: int | None = 0) -> np.ndarray:
    """Calculate PRCC confidence intervals using bootstrapping."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    out = np.empty((n_bootstrap, d))
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        out[b] = _partial_rank_corr_coeff(X, y)(X[idx], y[idx])
    return out


# SOBOL
def _salib_problem(bounds: dict) -> dict:
    """Format parameter bounds into SALib problem dictionary."""
    names = list(bounds)
    return {"num_vars": len(names), "names": names, "bounds":[list(bounds[n]) for n in names]}

def saltelli_sample(bounds: dict, n_base: int, seed: int | None = None) -> np.ndarray:
    """Generate a Saltelli sample sequence for Sobol analysis."""
    return sample(_salib_problem(bounds), N=n_base, calc_second_order=False, seed=seed)

def _sobol_indices(bounds: dict, Y: np.ndarray, seed: int | None = None) -> dict[str, np.ndarray]:
    """Compute first-order (S_1) and total-order (S_T) Sobol sensitivity indices."""
    Si = analyze(_salib_problem(bounds), np.asarray(Y), calc_second_order=False, seed=seed, print_to_console=False)
    return {k: np.asarray(Si[k]) for k in ("S1", "S1_conf", "ST", "ST_conf")}


# WORKFLOW
def _evaluate_samples(model: Callable, base_params: Params, bounds: dict, samples: np.ndarray, t1: float, E0: float = 1e-6, outcome: str = "Rt", avg_frac: float = 0.1,) -> np.ndarray:
    def _evaluator(model, base_params, names, t1, E0, outcome, avg_frac):
        def _apply(row: jnp.ndarray) -> Params:
            kwargs: dict = {}
            for i, name in enumerate(tuple(names)):
                v = row[i]
                if name == "log_k": kwargs["k"] = 10.0 ** v
                elif name == "log_k_I": kwargs["k_I"] = 10.0 ** v
                elif name == "log_I_crit": kwargs["I_crit"] = 10.0 ** v
                else: kwargs[name] = v
            return base_params.update(**kwargs)
        if outcome == "Itot":
            def _eval(row):
                params = _apply(row)
                tt, yy = model(params=params, t1=t1, E0=E0)
                _, _, Itot, _, _, _, _, _ = outcome_metrics(tt, yy, params, t1)
                return Itot
        else:
            def _eval(row):
                params = _apply(row)
                tt, yy = model(params=params, t1=t1, E0=E0)
                Rt_final, _, _, _, _, _, _, _ = outcome_metrics(tt, yy, params, t1)
                return Rt_final
        return jax.jit(jax.vmap(_eval))
    return np.asarray(_evaluator(model=model, base_params=base_params, names=list(bounds), t1=t1, E0=E0, outcome=outcome, avg_frac=avg_frac)(jnp.asarray(samples)))

def run_sensitivity_analysis(
    model: Callable,
    base_params: Params,
    scenario: str = "start",
    outcome: str = "Rt",
    t1: float = 50.0,
    E0: float = 1e-6,
    n_lhs: int = 5000,
    n_bootstrap: int = 100,
    n_sobol_base: int = 1024,
    avg_frac: float = 0.1,
    ci: tuple[float, float] = (2.5, 97.5),
    seed: int = 0,
    do_sobol: bool = True,
    manual_bounds: dict[str, tuple[float, float]] | None = None,
    around_mean: bool = False,
) -> SensitivityResults:
    if around_mean:
        bounds = _parameter_bounds_around_mean(model, scenario=scenario, mean_params=base_params)
    else:
        bounds = _parameter_bounds(model, scenario=scenario)
    if manual_bounds is not None:
        for k, v in manual_bounds.items():
            if k in bounds: 
                bounds[k] = v
    bp = base_params.update(I_crit=0.0) if scenario == "start" else base_params.update(I_crit=1e-4)

    # prcc
    X = _construct_latin_hypercube(bounds, n_lhs, seed=seed)
    y = _evaluate_samples(model=model, base_params=bp, bounds=bounds, samples=X, t1=t1, E0=E0, outcome=outcome, avg_frac=avg_frac)
    boot = _prcc_bootstrap(X, y, n_bootstrap=n_bootstrap, seed=seed+1)
    prcc_mean = boot.mean(axis=0)
    prcc_lower, prcc_upper = np.percentile(boot, ci, axis=0)

    # sobol
    names = list(bounds)
    if do_sobol:
        samples_sobol = saltelli_sample(bounds, n_base=n_sobol_base, seed=seed+2)
        Si = _sobol_indices(bounds, _evaluate_samples(model=model, base_params=bp, bounds=bounds, samples=samples_sobol, t1=t1, E0=E0, outcome=outcome, avg_frac=avg_frac), seed=seed+3)
        S1, S1_conf, ST, ST_conf = Si["S1"], Si["S1_conf"], Si["ST"], Si["ST_conf"]
    else:
        d = len(names)
        S1, S1_conf, ST, ST_conf = np.full(d, np.nan), np.full(d, np.nan), np.full(d, np.nan), np.full(d, np.nan)

    return SensitivityResults(param_names=names, bounds=bounds, samples=X, outputs=y, prcc_mean=prcc_mean, prcc_lower=prcc_lower, prcc_upper=prcc_upper, prcc_samples=boot, sobol_S1=S1, sobol_S1_conf=S1_conf, sobol_ST=ST, sobol_ST_conf=ST_conf)

def partial_rank_residuals(X: np.ndarray, y: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate partial rank residuals for an input parameter against the output."""
    X_rank = np.apply_along_axis(rankdata, 0, X)
    y_rank = rankdata(y)
    Z = np.column_stack([np.ones(len(X_rank)), np.delete(X_rank, i, axis=1)])
    beta_x, *_ = np.linalg.lstsq(Z, X_rank[:, i], rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z, y_rank, rcond=None)
    return X_rank[:, i] - Z @ beta_x, y_rank - Z @ beta_y
