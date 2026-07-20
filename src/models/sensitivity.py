"""
Global sensitivity analysis.
"""

from functools import lru_cache, partial
from typing import NamedTuple
import numpy as np
from scipy.stats import qmc, rankdata, norm
import jax
import jax.numpy as jnp

from SALib.sample.sobol import sample as saltelli_sequence
from SALib.analyze.sobol import analyze

from models.parameters import Params
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.metrics import outcome_metrics, trajectory_indices
from models.uncertainty import Priors

DEFAULT_BATCH_SIZE = 1024
MODEL_NAMES = {simulate_SEIPAR_W: "SEIPAR_W", simulate_SEIAR_W: "SEIAR_W", simulate_SEIR_W: "SEIR_W"}
DUMMY = "dummy"


def _Rt(tt, yy, params, dep_lo: float = 1e-6, dep_hi: float = 1e-2):
    """Mean Rt over the interval of susceptible depletion [dep_lo, dep_hi]."""
    idx = trajectory_indices(n_W=params.n_W, n_B=params.n_B)
    S = yy[:, idx["S"]]
    Rt = params.R_0 * params.rho * yy[:, idx["B_out"]] * S
    start = S < (1.0 - dep_lo) * S[0]
    end = S < (1.0 - dep_hi) * S[0]
    window = (tt >= jnp.where(jnp.any(start), tt[jnp.argmax(start)], tt[-1])) & (tt <= jnp.where(jnp.any(end), tt[jnp.argmax(end)], tt[-1]))
    n_in = jnp.sum(window)
    return jnp.where(n_in > 0, jnp.sum(Rt * window) / jnp.maximum(n_in, 1), Rt[-1])

_PRIMITIVES = {
    "SEIPAR_W": ("R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "p", "RR_p", "RR_a"),
    "SEIAR_W": ("R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "p", "RR_a"),
    "SEIR_W": ("R_0", "gamma_inv", "mu_s_inv")
}
_INTERVENTION_BOUNDS = {"epsilon_s": (0.0, 1.0), "epsilon_w": (0.0, 1.0), "tau_W": (1.0, 30.0), "tau_B": (1.0, 30.0), "log_k": (0.0, 3.0), "R_crit": (0.8, 1.5)}
_THRESHOLD_BOUNDS = {"log_kI_Icrit": (0.5, 2.5), "log_I_crit": (-4.0, -2.0)}


def _parameter_bounds_from_priors(model, priors: Priors, scenario: str = "start", include_dummy: bool = True):
    name = MODEL_NAMES[model]
    bounds: dict[str, tuple[float, float]] = {}
    for key in _PRIMITIVES[name]:
        m = priors.marginals[key]
        bounds[key] = (float(m.lo), float(m.hi))
    bounds |= _INTERVENTION_BOUNDS
    if scenario == "threshold":
        bounds |= _THRESHOLD_BOUNDS
    elif scenario != "start":
        raise ValueError(scenario)
    if include_dummy:
        bounds[DUMMY] = (0.0, 1.0)
    return bounds

def _relative_bounds(value, name, frac=0.2):
    lo, hi = value * (1.0 - frac), value * (1.0 + frac)
    if hi <= lo:
        if name in _INTERVENTION_BOUNDS:
            return _INTERVENTION_BOUNDS[name]
        raise ValueError(name)
    return (lo, hi)

def _parameter_bounds_around_mean(model, mean_params: Params, scenario="start", frac=0.2, include_dummy=True):
    name = MODEL_NAMES[model]
    ps = mean_params
    prim = {"R_0": float(ps.R_0), "gamma_inv": float(ps.gamma_inv), "mu_s_inv": float(ps.mu_s_inv)}
    if name in ("SEIPAR_W", "SEIAR_W"):
        prim["sigma_inv"] = float(ps.sigma_inv) if float(ps.sigma_inv) > 0 else 1.0
        prim["p"] = float(ps.p)
        prim["RR_a"] = float(ps.phi_a) * float(ps.mu_a_inv) / float(ps.mu_s_inv)
    if name == "SEIPAR_W":
        prim["RR_p"] = float(ps.phi_p) * float(ps.sigma_inv) / float(ps.mu_s_inv)
    bounds = {k: _relative_bounds(prim[k], k, frac) for k in _PRIMITIVES[name]}
    bounds |= {k: _relative_bounds(float(getattr(ps, k)), k, frac)
               for k in ("epsilon_s", "epsilon_w", "tau_W", "tau_B", "R_crit")}
    bounds["log_k"] = (np.log10(float(ps.k) * (1 - frac)), np.log10(float(ps.k) * (1 + frac)))
    if scenario == "threshold":
        I_crit = float(ps.I_crit) if float(ps.I_crit) > 0 else 1e-4
        kI_Ic = float(ps.k_I) * I_crit
        bounds["log_kI_Icrit"] = (np.log10(kI_Ic * (1 - frac)), np.log10(kI_Ic * (1 + frac)))
        bounds["log_I_crit"] = (np.log10(I_crit * (1 - frac)), np.log10(I_crit * (1 + frac)))
    elif scenario != "start":
        raise ValueError(scenario)
    if include_dummy:
        bounds[DUMMY] = (0.0, 1.0)
    return bounds

def _make_params(model_name, names):
    def _params(params, row):
        v = {n: row[i] for i, n in enumerate(names)}
        v.pop(DUMMY, None)
        kwargs: dict = {}
        if "log_k" in v:
            kwargs["k"] = 10.0 ** v.pop("log_k")
        if "log_I_crit" in v:
            I_crit = 10.0 ** v.pop("log_I_crit")
            kwargs["I_crit"] = I_crit
            kwargs["k_I"] = (10.0 ** v.pop("log_kI_Icrit")) / I_crit
        if model_name in ["SEIPAR_W", "SEIAR_W"]:
            sigma_inv = v.pop("sigma_inv")
            mu_s_inv = v["mu_s_inv"]
            mu_a_inv = sigma_inv + mu_s_inv
            kwargs["mu_a_inv"] = mu_a_inv
            kwargs["phi_a"] = jnp.where(
                mu_a_inv > 0, v.pop("RR_a") * mu_s_inv / jnp.where(mu_a_inv > 0, mu_a_inv, 1.0), 0.0)
            if model_name == "SEIPAR_W":
                kwargs["sigma_inv"] = sigma_inv
                kwargs["phi_p"] = jnp.where(sigma_inv > 0, v.pop("RR_p") * mu_s_inv / jnp.where(sigma_inv > 0, sigma_inv, 1.0), 0.0)
        kwargs.update(v)
        return params.update(**kwargs)
    return _params

@lru_cache(maxsize=None)
def _outcome_metric(model, names, outcome):
    _params = _make_params(MODEL_NAMES[model], names)
    def _out(base_params, row, t1, E0):
        params = _params(base_params, row)
        tt, yy = model(params=params, t1=t1, E0=E0)
        if outcome == "Rt": return _Rt(tt, yy, params)
        if outcome == "Itot": return outcome_metrics(tt, yy, params, t1)[2]
        else: raise ValueError(outcome)
    return jax.jit(jax.vmap(_out, in_axes=(None, 0, None, None)))

def _evaluate_samples(model, base_params, bounds, samples, t1, E0=1e-6, outcome="Rt", batch_size=DEFAULT_BATCH_SIZE):
    m = _outcome_metric(model, tuple(bounds), outcome)
    samples = np.asarray(samples)
    out = np.empty(samples.shape[0])
    for s in range(0, samples.shape[0], batch_size):
        batch = jnp.asarray(samples[s:s + batch_size])
        out[s:s + batch.shape[0]] = np.asarray(m(base_params, batch, t1, E0))
    return out


# PRCC
def _construct_latin_hypercube(bounds, n, seed=None):
    names = list(bounds)
    return qmc.scale(sample=qmc.LatinHypercube(d=len(names), seed=seed).random(n=n), l_bounds=np.array([bounds[p][0] for p in names]), u_bounds=np.array([bounds[p][1] for p in names]))

def _partial_rank_corr_coeff(X, y):
    """Compute the partial rank correlation coefficients between model parameters and output."""
    ranks = np.hstack((np.apply_along_axis(rankdata, 0, X), rankdata(y).reshape(-1, 1)))
    if np.any(ranks.std(axis=0) == 0):
        raise ValueError
    C = np.corrcoef(ranks, rowvar=False)
    W = np.linalg.pinv(C) # precision matrix
    # PRCC = -Wxy / sqrt(Wxx * Wyy)
    denom = np.sqrt(np.clip(np.diag(W) * W[-1, -1], 1e-12, None))
    return np.array([-W[i, -1] / denom[i] for i in range(X.shape[1])])

def _prcc_fisher_ci(r, n, d, alpha=0.05):
    """Eq. 10, Merino et al. (2008)."""
    df = max(n - (d - 1) - 3, 1)
    z = np.arctanh(np.clip(np.asarray(r, float), -1 + 1e-12, 1 - 1e-12))
    half = norm.ppf(1 - alpha / 2) / np.sqrt(df)
    return np.tanh(z - half), np.tanh(z + half)

def _prcc_replicates(m, bounds, n_lhs, n_replicates, seed):
    est = []
    for r in range(n_replicates):
        X = _construct_latin_hypercube(bounds, n_lhs, seed=seed + 1000 + r)
        y = m(samples=X)
        est.append(_partial_rank_corr_coeff(X, y))
    return np.asarray(est)


# SOBOL
def _salib_problem(bounds):
    """Format parameter bounds into SALib problem dictionary."""
    names = list(bounds)
    return {"num_vars": len(names), "names": names, "bounds": [list(bounds[n]) for n in names]}

def saltelli_sample(bounds, n_base, seed=None, second_order=False):
    """Generate Saltelli sample sequence."""
    return saltelli_sequence(_salib_problem(bounds), N=n_base, calc_second_order=second_order, seed=seed)

def _sobol_indices(bounds, Y, seed=None, second_order=False):
    """Compute first-order (S_1) and total-order (S_T) Sobol sensitivity indices."""
    Si = analyze(_salib_problem(bounds), np.asarray(Y), calc_second_order=second_order, seed=seed, print_to_console=False)
    keys = ["S1", "S1_conf", "ST", "ST_conf"] + (["S2", "S2_conf"] if second_order else [])
    return {k: np.asarray(Si[k]) for k in keys}


# WORKFLOW
class SensitivityResults(NamedTuple):
    param_names: list
    bounds: dict
    samples: np.ndarray
    outputs: np.ndarray
    prcc_mean: np.ndarray
    prcc_lower: np.ndarray
    prcc_upper: np.ndarray
    prcc_samples: np.ndarray
    sobol_S1: np.ndarray
    sobol_S1_conf: np.ndarray
    sobol_ST: np.ndarray
    sobol_ST_conf: np.ndarray
    prcc_fisher_lower: np.ndarray = None
    prcc_fisher_upper: np.ndarray = None
    sobol_S1_sum: float = np.nan
    sobol_S2: np.ndarray = None
    sobol_S2_conf: np.ndarray = None

def run_sensitivity_analysis(
    model, base_params: Params, scenario="start", outcome="Rt", t1=50.0, E0=1e-6, n_lhs=5000, n_replicates=20, n_sobol_base=1024, ci=(2.5, 97.5), 
    seed=0, do_sobol=True, second_order=False, around_mean=False, priors=None, include_dummy=True, batch_size=DEFAULT_BATCH_SIZE,
) -> SensitivityResults:
    if around_mean: 
        bounds = _parameter_bounds_around_mean(model, base_params, scenario=scenario, include_dummy=include_dummy)
    else:
        bounds = _parameter_bounds_from_priors(model, priors, scenario=scenario, include_dummy=include_dummy)

    names = list(bounds)
    base_params = base_params.update(I_crit=0.0) if scenario == "start" else base_params
    m = partial(_evaluate_samples, model=model, base_params=base_params, bounds=bounds, t1=t1, E0=E0, outcome=outcome, batch_size=batch_size)
    X = _construct_latin_hypercube(bounds, n_lhs, seed=seed)
    y = m(samples=X)
    prcc = _partial_rank_corr_coeff(X, y)
    f_lo, f_hi = _prcc_fisher_ci(prcc, n_lhs, len(names))
    replicates = (_prcc_replicates(m, bounds, n_lhs, n_replicates, seed) if n_replicates > 1 else prcc[None, :])
    lo, hi = np.percentile(replicates, ci, axis=0)

    if do_sobol:
        Si = _sobol_indices(
            Y=m(samples=saltelli_sample(bounds, n_base=n_sobol_base, seed=seed + 2, second_order=second_order)),
            bounds=bounds, seed=seed + 3, second_order=second_order)
        S1, S1c, ST, STc = Si["S1"], Si["S1_conf"], Si["ST"], Si["ST_conf"]
        S2 = Si.get("S2")
        S2c = Si.get("S2_conf")
    else:
        S1, S1c, ST, STc = (np.full(len(names), np.nan) for _ in range(4))
        S2, S2c = None, None

    return SensitivityResults(
        param_names=names, bounds=bounds, samples=X, outputs=y, prcc_mean=prcc, prcc_lower=lo, prcc_upper=hi, 
        prcc_samples=replicates, sobol_S1=S1, sobol_S1_conf=S1c, sobol_ST=ST, sobol_ST_conf=STc, prcc_fisher_lower=f_lo, 
        prcc_fisher_upper=f_hi, sobol_S1_sum=float(np.nansum(S1)), sobol_S2=S2, sobol_S2_conf=S2c,
    )

def partial_rank_residuals(X: np.ndarray, y: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate partial rank residuals for an input parameter against the output."""
    X_rank = np.apply_along_axis(rankdata, 0, X)
    y_rank = rankdata(y)
    Z = np.column_stack([np.ones(len(X_rank)), np.delete(X_rank, i, axis=1)])
    beta_x, *_ = np.linalg.lstsq(Z, X_rank[:, i], rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z, y_rank, rcond=None)
    return X_rank[:, i] - Z @ beta_x, y_rank - Z @ beta_y

def load_sensitivity_results(path: str) -> SensitivityResults:
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["param_names"]]
    return SensitivityResults(
        param_names=names, bounds={
            n:(float(lo),float(hi)) for n, lo, hi in zip(names, d["lower_bounds"], d["upper_bounds"])}, samples=d["samples"], 
            outputs=d["outputs"], prcc_mean=d["prcc_mean"], prcc_lower=d["prcc_lower"], prcc_upper=d["prcc_upper"], prcc_samples=d["prcc_samples"], 
            sobol_S1=d["sobol_S1"], sobol_S1_conf=d["sobol_S1_conf"], sobol_ST=d["sobol_ST"], sobol_ST_conf=d["sobol_ST_conf"]
    )

def export_sensitivity_bounds(combinations, path, npzs):
    col_mapping = {("SARS-CoV-2", "threshold"): "SARS-CoV-2", ("H1N1", "threshold"): "H1N1", ("Ebola", "threshold"): "Ebola"}
    param_defs = {
        "R_0": ("$\\mathcal{R}_0$", "basic reproductive number"), "gamma_inv": ("$1/\\gamma$", "latent period"), "mu_s_inv": ("$1/\\mu_s$", "symptomatic period"), 
        "sigma_inv": ("$1/\\sigma$", "presymptomatic period"), "mu_a_inv": ("$1/\\mu_a$", "asymptomatic period"), "p": ("$p$", "proportion asymptomatic"), 
        "phi_a": ("$\\varphi_a$", "relative asympt. infectiousness"), "phi_p": ("$\\varphi_p$", "relative presympt. infectiousness"), 
        "epsilon_s": ("$\\varepsilon_s$", "isolation efficacy"), "epsilon_w": ("$\\varepsilon_w$", "warning response efficacy"), "tau_W": ("$\\tau_W$", "reporting delay"), "tau_B": ("$\\tau_B$", "behavioural delay"), 
        "log_k": ("$\\log_{10} k$", "warning gate sharpness"), "R_crit": ("$\\mathcal{R}_{\\text{crit}}$", "warning threshold"), 
        "log_k_I": ("$\\log_{10} k_I$", "prevalence gate sharpness (*)"), "log_I_crit": ("$\\log_{10} I_{\\text{crit}}$", "prevalence threshold (*)"), 
    }
    bounds_data = {}
    for combination, fpath in zip(combinations, npzs): bounds_data[combination] = load_sensitivity_results(fpath).bounds
    with open(path, 'w') as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{llccc}\n\\toprule\n")
        header = ["Parameter", ""] + [col_mapping[c] for c in combinations]
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for p_key, (symbol, desc) in param_defs.items():
            row = [desc, symbol]
            for combination in combinations: 
                if p_key in bounds_data[combination]: row.append(f"$[{bounds_data[combination][p_key][0]:g}, {bounds_data[combination][p_key][1]:g}]$")
                else: row.append("---")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption[Parameter ranges used for sensitivity analysis]{Parameter ranges used for Latin hypercube sampling in the global sensitivity analysis. Parameters marked with an asterisk (*) are only included when the symptomatic threshold $I_\\text{crit}=10^{-4}$ is active.}\n\\label{tab:prcc-bounds}\n\\end{table}\n")
