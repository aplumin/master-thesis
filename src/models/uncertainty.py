"""
Prior parameters and Monte Carlo draws.
"""
from typing import NamedTuple

import numpy as np
from scipy.stats import beta, norm

from models.metrics import (
    critical_isolation_efficacy,
    critical_warning_efficacy,
    eps_s_boundary,
    generation_time,
    growth_rate,
    infectious_fractions,
    mean_warning_multiplier,
    theta_from_type_R_values,
    transmission_fractions,
)
from models.parameters import Params, calculate_r
from models.parameters_erlang import ParamsErlang


class Marginal(NamedTuple):
    """Marginal fitted to quantiles."""
    lo: float
    hi: float
    family: str = "lognormal"
    mean: float | None = None
    quant_lo: float = 0.025
    quant_hi: float = 0.975

    def central(self) -> float:
        if self.mean is not None:
            return float(self.mean)
        if self.lo == self.hi:
            return float(self.lo)
        return float(_ppf(self, np.array([0.5]))[0])

def _ppf(m, u):
    """
    Inverse CDF of a Marginal at uniform draws u in (0, 1):
      - uniform: linear between lo and hi.
      - normal: mean + s*Phi^{-1}(u) with scale s = (hi - lo)/(z_hi - z_lo) and mean = lo - s*z_lo.
      - lognormal: as normal but in logspace.
      - beta: fit to mean and concentration mean*(1-mean)/sd^2 - 1.
    """
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
            raise ValueError(conc)
        return beta.ppf(u, mean * conc, (1.0 - mean) * conc)
    raise ValueError(m.family)


class Priors(NamedTuple):
    """Prior dictionary {name: Marginal}."""
    marginals: dict[str, Marginal]
    corr: np.ndarray | None = None
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
    if priors.corr is not None: # apply correlation
        corr = np.asarray(priors.corr, float)
        if corr.shape != (len(names), len(names)):
            raise ValueError(corr.shape)
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

def epi_quantities(s: dict[str, np.ndarray], epsilon_s: float = 0.0, epsilon_w: float = 0.0):
    """Derived epidemiological quantities."""
    R0 = s["R_0"]
    # infectious weights (probability * relative infectiousness * duration).
    ra = s["p"] * s["phi_a"] * s["mu_a_inv"]
    rp = (1.0 - s["p"]) * s["phi_p"] * s["sigma_inv"]
    rs = (1.0 - s["p"]) * s["mu_s_inv"]
    r = ra + rp + rs
    R_a = np.where(r > 0, R0 * ra/r, 0.0)
    R_p = np.where(r > 0, R0 * rp/r, 0.0)
    R_s = np.where(r > 0, R0 * rs/r, 0.0)
    theta = theta_from_type_R_values(R0, R_a, R_p)
    T_g = generation_time(r, s["p"], s["phi_a"], s["phi_p"], s["gamma_inv"], s["sigma_inv"], s["mu_a_inv"], s["mu_s_inv"])
    eps_s_crit = critical_isolation_efficacy(R_s, R_a, R_p)
    R_eps = R_a + R_p + (1.0 - epsilon_s) * R_s
    eps_w_crit = critical_warning_efficacy(R_eps)
    R_t = R_eps * mean_warning_multiplier(epsilon_w)
    return {"R_a": R_a, "R_p": R_p, "R_s": R_s, "theta": theta, "T_g": T_g, "R_eps": R_eps, "eps_s_crit": eps_s_crit, "eps_w_crit": eps_w_crit, "R_t": R_t}

def joint_ci(priors: Priors, names=None, n: int = 20000, seed: int = 0, quantiles: tuple[float, float, float] = (0.025, 0.5, 0.975), **kw):
    """Return CI {name: (median, lo, hi)} and sample dict."""
    names = ["R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_p", "phi_a", "beta"] if names is None else list(names)
    s = sample_derived(priors, n=n, seed=seed, **kw)
    ci = {}
    for name, arr in {**{k: s[k] for k in names if k in s}, **epi_quantities(s)}.items():
        lo, med, hi = np.quantile(arr, quantiles)
        ci[name] = (float(med), float(lo), float(hi))
    return ci, s

def pushforward(priors: Priors, fn, n: int = 2000, seed: int = 0, quantiles=(0.025, 0.5, 0.975), **kw):
    """Pointwise joint uncertainty for function fn."""
    s = sample_derived(priors, n=n, seed=seed, **kw)
    return np.quantile(np.stack([np.asarray(fn(s, i)) for i in range(n)], axis=0), quantiles, axis=0)

def params_from_priors(pr: Priors, model="exponential", weighted=True):
    """Point estimate Params from priors."""
    m = {k: pr.marginals[k].central() for k in pr.marginals}
    if not pr.presymptomatic and not pr.asymptomatic:
        if model == "Erlang":
            return ParamsErlang.for_SEIR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], mu_s_inv=m["mu_s_inv"])
        return Params.for_SEIR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], mu_s_inv=m["mu_s_inv"])
    mu_a_inv = (m["sigma_inv"] + m["mu_s_inv"]) if pr.asymptomatic else 0.0
    phi_p = (m["RR_p"] * m["mu_s_inv"] / m["sigma_inv"]) if (pr.presymptomatic and m["sigma_inv"] > 0) else 0.0
    phi_a = (m["RR_a"] * m["mu_s_inv"] / mu_a_inv) if (pr.asymptomatic and mu_a_inv > 0) else 0.0
    if model == "Erlang":
        return ParamsErlang.for_SEIPAR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], sigma_inv=m["sigma_inv"], mu_s_inv=m["mu_s_inv"], mu_a_inv=mu_a_inv, p=m["p"], phi_a=phi_a, phi_p=phi_p, weighted=weighted)
    return Params.for_SEIPAR(R_0=m["R_0"], gamma_inv=m["gamma_inv"], sigma_inv=m["sigma_inv"], mu_s_inv=m["mu_s_inv"], mu_a_inv=mu_a_inv, p=m["p"], phi_a=phi_a, phi_p=phi_p)

def get_model_prior_list(pr):
    if not pr.presymptomatic and not pr.asymptomatic:
        return ["R_0", "gamma_inv", "mu_s_inv"]
    return ["R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_a", "phi_p"]

def get_epi_characteristics_dict(ps: Params):
    """Return dict of epi characteristics from parameters."""
    eq = {q: float(v[0]) for q, v in epi_quantities({p: np.array([float(getattr(ps, p))]) for p in ["R_0", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "p", "phi_a", "phi_p"]}).items()}
    d = {"R_0": float(ps.R_0), "beta": float(ps.beta), "generation_time": eq["T_g"], "growth_rate": growth_rate(ps), "theta": eq["theta"], "eps_s_crit": eq["eps_s_crit"], "eps_w_crit": eq["eps_w_crit"]}
    trans_f = transmission_fractions(ps)
    inf_f = infectious_fractions(ps)
    for k in ("a", "p", "s"):
        d[f"R_{k}"] = eq[f"R_{k}"]
        d[f"transmission_frac_{k}"] = trans_f[k]
        d[f"infectious_frac_{k}"] = inf_f[k]
    return d

def probability_uncontrollable(priors: Priors, n: int = 200_000, seed: int = 0):
    """P(R_a + R_p > 1)."""
    s = sample_derived(priors, n=n, seed=seed)
    eq = epi_quantities(s)
    R_ns = (eq["R_a"] + eq["R_p"])
    lo, med, hi = np.quantile(R_ns, [0.025, 0.5, 0.975])
    return {"P": float((R_ns > 1.0).mean()), "median": float(med), "lo": float(lo), "hi": float(hi)}

def analytic_boundary(priors: Priors, eps_ww=None, n: int = 200_000, seed: int = 0, quantiles=(0.025, 0.5, 0.975), k=None, R_crit: float = 1.0):
    """95% CI on epsilon_s required for Rt = 1."""
    eps_ww = np.linspace(0.0, 1.0, 100) if eps_ww is None else np.asarray(eps_ww, float)
    s = sample_derived(priors, n=n, seed=seed)
    eps_s = eps_s_boundary(R_0=s["R_0"][:, None], theta=epi_quantities(s)["theta"][:, None], eps_w=eps_ww[None, :], k=k, R_crit=R_crit)
    return np.nanquantile(eps_s, quantiles, axis=0)
