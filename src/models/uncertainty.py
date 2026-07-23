"""
Prior specification and Monte Carlo uncertainty handling.
"""

from typing import NamedTuple, Optional, Dict, Tuple
import numpy as np
from scipy.stats import norm, beta

from models.parameters import Params, calculate_r
from models.metrics import growth_rate, transmission_fractions, infectious_fractions, mean_warning_multiplier


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

    def central(self) -> float:
        if self.mean is not None:
            return float(self.mean)
        if self.lo == self.hi:
            return float(self.lo)
        return float(_ppf(self, np.array([0.5]))[0])

def _ppf(m: Marginal, u: np.ndarray):
    """
    Inverse-CDF of a Marginal at uniform draws u in (0, 1):
      - uniform:   linear between lo and hi.
      - normal:    mean + s*Phi^{-1}(u) 
                   with scale s = (hi - lo)/(z_hi - z_lo) and mean = lo - s*z_lo.
      - lognormal: same as normal in logspace.
      - beta:      fit to mean and concentration mean*(1-mean)/sd^2 - 1.
    """
    u = np.asarray(u, dtype=float)
    if m.lo == m.hi:
        return np.full(u.shape, float(m.lo))
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
        if not 0.0 < mean < 1.0:
            raise ValueError(mean)
        sd = (m.hi - m.lo) / (z_hi - z_lo)
        conc = mean * (1.0 - mean) / sd**2 - 1.0
        if conc <= 0.0:
            raise ValueError()
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
    names = priors.names
    Z = rng.standard_normal((n, len(names)))
    if priors.corr is not None:
        corr = np.asarray(priors.corr, float)
        if corr.shape != (len(names), len(names)):
            raise ValueError()
        Z = Z @ np.linalg.cholesky(corr).T
    U = norm.cdf(Z)
    return {name: _ppf(priors.marginals[name], U[:, i]) for i, name in enumerate(names)}

def sample_derived(priors: Priors, n: int = 20000, seed: int = 0):
    """Draw n samples including derived parameters."""
    s = sample_primitives(priors, n, seed=seed)
    s["mu_a_inv"] = s["sigma_inv"] + s["mu_s_inv"] if priors.asymptomatic else np.zeros(n)
    s["phi_p"] = np.divide(
        s["RR_p"] * s["mu_s_inv"], s["sigma_inv"], out=np.zeros(n), where=s["sigma_inv"] > 0
    ) if priors.presymptomatic else np.zeros(n)
    s["phi_a"] = np.divide(
        s["RR_a"] * s["mu_s_inv"], s["mu_a_inv"], out=np.zeros(n), where=s["mu_a_inv"] > 0
    ) if priors.asymptomatic else np.zeros(n)
    r = calculate_r(p=s["p"], phi_a=s["phi_a"], phi_p=s["phi_p"], mu_a_inv=s["mu_a_inv"], sigma_inv=s["sigma_inv"], mu_s_inv=s["mu_s_inv"])
    s["beta"] = np.divide(s["R_0"], r, out=np.zeros(n), where=r > 0)
    return s

_SAMPLE_CACHE: Dict[tuple, tuple] = {}

def cached_sample_derived(priors: Priors, n: int = 20000, seed: int = 0):
    key = (id(priors), n, seed)
    val = _SAMPLE_CACHE.get(key)
    if val is not None and val[0] is priors:
        return val[1]
    result = sample_derived(priors, n=n, seed=seed)
    _SAMPLE_CACHE[key] = (priors, result)
    return result

def epi_quantities(s: Dict[str, np.ndarray], epsilon_s: float = 0.0, epsilon_w: float = 0.0):
    """Derived epidemiological quantities."""
    R0 = s["R_0"]
    # infectious weights (probability * relative infectiousness * duration).
    ra = s["p"] * s["phi_a"] * s["mu_a_inv"]
    rp = (1.0 - s["p"]) * s["phi_p"] * s["sigma_inv"]
    rs = (1.0 - s["p"]) * s["mu_s_inv"]
    r = ra + rp + rs
    with np.errstate(divide="ignore", invalid="ignore"):
        R_a = np.where(r > 0, R0 * ra/r, 0.0)
        R_p = np.where(r > 0, R0 * rp/r, 0.0)
        R_s = np.where(r > 0, R0 * rs/r, 0.0)
        # nonsymptomatic fraction
        theta = np.where(R0 > 0, (R_a + R_p) / np.where(R0 > 0, R0, 1.0), 0.0)
        # mean generation time
        T_g = s["gamma_inv"] + np.where(r > 0, (s["p"] * s["phi_a"] * s["mu_a_inv"]**2 + (1.0 - s["p"]) * (s["phi_p"] * s["sigma_inv"]**2 + s["mu_s_inv"]**2 + s["sigma_inv"] * s["mu_s_inv"])) / np.where(r > 0, r, 1.0), 0.0)
        # isolation efficacy needed for R_t = 1 (eps_w = 0)
        eps_s_crit = np.where(R_s > 0, 1.0 - (1.0 - (R_a + R_p)) / np.where(R_s > 0, R_s, 1.0), np.nan)
        # warning efficacy needed for R_t = 1 (eps_s = 0)
        eps_w_crit = 2.0 * (1.0 - 1.0 / np.where(R0 > 0, R0, np.nan)) # assuming R_crit=1
    R_t = (R_a + R_p + (1.0 - epsilon_s) * R_s) * mean_warning_multiplier(epsilon_w)
    return dict(R_a=R_a, R_p=R_p, R_s=R_s, theta=theta, T_g=T_g, eps_s_crit=eps_s_crit, eps_w_crit=eps_w_crit, R_t=R_t)

_DEFAULT_CI_NAMES = ("R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_p", "phi_a", "beta")

def joint_ci(priors: Priors, names=None, n: int = 20000, seed: int = 0, quantiles: Tuple[float, float, float] = (0.025, 0.5, 0.975), **kw):
    """Return CI {name: (median, lo, hi)} and sample dict."""
    names = list(_DEFAULT_CI_NAMES) if names is None else list(names)
    s = sample_derived(priors, n=n, seed=seed, **kw)
    ci = {}
    for name, arr in {**{k: s[k] for k in names if k in s}, **epi_quantities(s)}.items():
        lo, med, hi = np.quantile(arr, quantiles)
        ci[name] = (float(med), float(lo), float(hi))
    return ci, s

def pushforward(priors: Priors, fn, n: int = 2000, seed: int = 0, quantiles=(0.025, 0.5, 0.975), **kw):
    """Pointwise joint uncertainty band for given function."""
    s = sample_derived(priors, n=n, seed=seed, **kw)
    return np.quantile(np.stack([np.asarray(fn(s, i)) for i in range(n)], axis=0), quantiles, axis=0)

def params_from_priors(pr: Priors):
    """Point-estimate Params from priors."""
    m = {k: pr.marginals[k].central() for k in pr.marginals}
    if not pr.presymptomatic and not pr.asymptomatic:
        return Params.for_SEIR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], mu_s_inv=m["mu_s_inv"])
    mu_a_inv = (m["sigma_inv"] + m["mu_s_inv"]) if pr.asymptomatic else 0.0
    phi_p = (m["RR_p"] * m["mu_s_inv"] / m["sigma_inv"]) if (pr.presymptomatic and m["sigma_inv"] > 0) else 0.0
    phi_a = (m["RR_a"] * m["mu_s_inv"] / mu_a_inv) if (pr.asymptomatic and mu_a_inv > 0) else 0.0
    return Params.for_SEIPAR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], sigma_inv=m["sigma_inv"], mu_s_inv=m["mu_s_inv"], mu_a_inv=mu_a_inv, p=m["p"], phi_a=phi_a, phi_p=phi_p)

def as_uniform(pr):
    marg = {k: Marginal(v.lo, v.hi, "uniform", mean=v.mean) for k, v in pr.marginals.items()}
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
