"""
Uncertainty handling.
"""

from typing import NamedTuple, Optional, Dict, Tuple
import numpy as np
from scipy.stats import norm, beta

from models.parameters import Params
from models.metrics import growth_rate, transmission_fractions, infectious_fractions


class Marginal(NamedTuple):
    """
    Marginal fitted to quantiles.
    """
    lo: float
    hi: float
    family: str = "lognormal"
    mean: Optional[float] = None
    quant_lo: float = 0.025
    quant_hi: float = 0.975

def _ppf(m: Marginal, u: np.ndarray):
    """Inverse-CDF of a Marginal at uniform draws u in (0, 1)."""
    if m.family == "uniform":
        return m.lo + (m.hi - m.lo) * u
    z_lo, z_hi = norm.ppf(m.quant_lo), norm.ppf(m.quant_hi)
    if m.family == "lognormal":
        s = (np.log(m.hi) - np.log(m.lo)) / (z_hi - z_lo)
        return np.exp(np.log(m.lo) - s * z_lo + s * norm.ppf(u))
    if m.family == "normal":
        s = (m.hi - m.lo) / (z_hi - z_lo)
        return m.lo - s * z_lo + s * norm.ppf(u)
    if m.family == "beta":
        mean = m.mean if m.mean is not None else 0.5 * (m.lo + m.hi)
        sd = (m.hi - m.lo) / (z_hi - z_lo)
        conc = mean * (1.0 - mean) / sd**2 - 1.0
        return beta.ppf(u, mean * conc, (1.0 - mean) * conc)
    raise ValueError(m.family)


class Priors(NamedTuple):
    """Prior dictionary {name: Marginal}."""
    marginals: Dict[str, Marginal]
    corr: Optional[np.ndarray] = None
    presymptomatic: bool = True
    asymptomatic: bool = True

    @property
    def names(self):
        return list(self.marginals)


def sample_primitives(priors: Priors, n: int, seed: int = 0):
    """Draw n samples of the primitive parameters."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(priors.names)))
    if priors.corr is not None:
        Z = Z @ np.linalg.cholesky(np.asarray(priors.corr, float)).T
    U = norm.cdf(Z)
    return {name: _ppf(priors.marginals[name], U[:, i]) for i, name in enumerate(priors.names)}

def sample_derived(priors: Priors, n: int = 20000, seed: int = 0):
    """Draw n samples including derived parameters."""
    s = sample_primitives(priors, n, seed=seed)
    s["mu_a_inv"] = s["sigma_inv"] + s["mu_s_inv"] if priors.asymptomatic else np.zeros(n)
    s["phi_p"] = s["RR_p"] * s["mu_s_inv"] / s["sigma_inv"] if priors.presymptomatic else np.zeros(n)
    s["phi_a"] = s["RR_a"] * s["mu_s_inv"] / s["mu_a_inv"] if priors.asymptomatic else np.zeros(n)
    s["beta"] = s["R_0"] / (s["p"] * s["phi_a"] * s["mu_a_inv"] + (1.0 - s["p"]) * (s["phi_p"] * s["sigma_inv"] + s["mu_s_inv"]))
    return s

def epi_quantities(s: Dict[str, np.ndarray], epsilon_s: float = 0.0, epsilon_w: float = 0.0):
    """Derived epidemiological quantities."""
    R0 = s["R_0"]
    ca = s["p"] * s["phi_a"] * s["mu_a_inv"]
    cp = (1.0 - s["p"]) * s["phi_p"] * s["sigma_inv"]
    cs = (1.0 - s["p"]) * s["mu_s_inv"]
    r = ca + cp + cs
    R_a = R0 * ca/r
    R_p = R0 * cp/r
    R_s = R0 * cs/r
    theta = (R_a + R_p) / R0
    T_g = s["gamma_inv"] + (s["p"] * s["phi_a"] * s["mu_a_inv"]**2 + (1.0 - s["p"]) * (s["phi_p"] * s["sigma_inv"]**2 + s["mu_s_inv"]**2 + s["sigma_inv"] * s["mu_s_inv"])) / r
    eps_s_crit = 1.0 - (1.0 - (R_a + R_p)) / R_s
    eps_w_crit = 2.0 * (1.0 - 1.0 / R0) # assuming R_crit=1
    R_t = (R_a + R_p + (1.0 - epsilon_s) * R_s) * (1.0 - epsilon_w / 2.0)
    return dict(R_a=R_a, R_p=R_p, R_s=R_s, theta=theta, T_g=T_g, eps_s_crit=eps_s_crit, eps_w_crit=eps_w_crit, R_t=R_t)

def joint_ci(priors: Priors, names: list[str] = ["R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_p", "phi_a", "beta"], 
            n: int = 20000, seed: int = 0, quantiles: Tuple[float, float, float] = (0.025, 0.5, 0.975), **kw):
    """Return CI {name: (median, lo, hi)} and sample dict."""
    s = sample_derived(priors, n=n, seed=seed, **kw)
    ci = {}
    for name, arr in {**{k: s[k] for k in names if k in s}, **epi_quantities(s)}.items():
        lo, med, hi = np.quantile(arr, quantiles)
        ci[name] = (float(med), float(lo), float(hi))
    return ci, s

def posterior(priors: Priors, fn, n: int = 2000, seed: int = 0, quantiles=(0.025, 0.5, 0.975), **kw):
    s = sample_derived(priors, n=n, seed=seed, **kw)
    return np.quantile(np.stack([np.asarray(fn(s, i)) for i in range(n)], axis=0), quantiles, axis=0)

def params_from_priors(pr: Priors):
    m = {k: pr.marginals[k].mean for k in pr.marginals}
    if not pr.presymptomatic and not pr.asymptomatic:
        return Params.for_SEIR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], mu_s_inv=m["mu_s_inv"])
    if not pr.asymptomatic:
        mu_a_inv = 0.0
    else:
        mu_a_inv = m["sigma_inv"] + m["mu_s_inv"]
    phi_p = m["RR_p"] * m["mu_s_inv"] / m["sigma_inv"] if pr.presymptomatic else 0.0
    phi_a = m["RR_a"] * m["mu_s_inv"] / mu_a_inv if pr.asymptomatic else 0.0
    return Params.for_SEIPAR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], sigma_inv=m["sigma_inv"], mu_s_inv=m["mu_s_inv"], mu_a_inv=mu_a_inv, p=m["p"], phi_a=phi_a, phi_p=phi_p)

def as_uniform(pr):
    marg = {k: Marginal(v.lo, v.hi, "uniform") for k, v in pr.marginals.items()}
    return Priors(marginals=marg, presymptomatic=pr.presymptomatic, asymptomatic=pr.asymptomatic)

def get_model_prior_list(pr):
    if not pr.presymptomatic and not pr.asymptomatic:
        return ["R_0", "gamma_inv", "mu_s_inv"]
    return ["R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_a", "phi_p"]

def get_epi_characteristics_dict(ps: Params):
    """Return dict of epi characteristics from parameters."""
    eq = {q: float(v[0]) for q, v in epi_quantities({
            p: np.array([float(getattr(ps, p))]) for p in [
                "R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_a", "phi_p"
        ]}).items()}
    d = {"R_0": float(ps.R_0), "beta": float(ps.beta), "generation_time": eq["T_g"], "growth_rate": growth_rate(ps), 
        "theta": eq["theta"], "eps_s_crit": eq["eps_s_crit"], "eps_w_crit": eq["eps_w_crit"]}
    trans_f = transmission_fractions(ps)
    inf_f = infectious_fractions(ps)
    for k in ("a", "p", "s"):
        d[f"R_{k}"] = eq[f"R_{k}"]
        d[f"transmission_frac_{k}"] = trans_f[k]
        d[f"infectious_frac_{k}"] = inf_f[k]
    return d

def corner_kwargs(pr, which):
    m = pr.marginals
    low = (which == "best")
    d = {}
    for k in ["R_0", "sigma_inv", "mu_a_inv", "p"]: 
        if k in m:
            d[k] = m[k].lo if low else m[k].hi
    for k in ["gamma_inv", "mu_s_inv"]: 
        if k in m:
            d[k] = m[k].hi if low else m[k].lo
    d["phi_p"] = ((m["RR_p"].lo if low else m["RR_p"].hi) * m["mu_s_inv"].mean / m["sigma_inv"].mean) if pr.presymptomatic else 0.0
    d["phi_a"] = ((m["RR_a"].lo if low else m["RR_a"].hi) * m["mu_s_inv"].mean / np.mean(sample_derived(pr)["mu_a_inv"])) if pr.asymptomatic else 0.0
    return d
