"""
Parameters for the Erlang variant of the compartmental model, in which each
infected stage (E, Ia, Ip, Is) is a linear chain of multiple subcompartments.
"""

import jax.numpy as jnp
from jax.scipy.stats import gamma
from typing import NamedTuple

from models.parameters import _register_static_pytree
_ERLANG_STATIC_FIELDS = ("n_W", "n_B", "nE", "nP", "nS", "nA", "weighted")


class ParamsErlang(NamedTuple):
    """
    Parameters for the compartmental model with linear chains for the infected compartments.

    Attributes:
        R_0 (float): Basic reproductive number.
        beta (float): Transmission rate.
        gamma_inv (float): Exposed period (inverse of become infectious rate).
        sigma_inv (float): Presymptomatic period (inverse of become symptomatic rate).
        mu_a_inv (float): Asymptomatic period (inverse of recovery rate).
        mu_s_inv (float): Symptomatic period (inverse of recovery rate).
        p (float): Proportion asymptomatic.
        phi_a (float): Relative infectiousness of asymptomatics.
        phi_p (float): Relative infectiousness of presymptomatics.
        epsilon_s (float): Isolation efficacy.
        epsilon_w (float): Contact rate reduction efficacy after warning response.
        k (float): Sharpness of warning response.
        R_crit (float): Rt threshold for warnings.
        tau_W (float): Reporting delay.
        tau_B (float): Behavioural delay.
        rho (float): Isolation reduction factor.
        I_crit (float): Infection threshold for interventions.
        k_I (float): Sharpness of infection threshold gate.
        n_W (int): Number of reporting delay compartments.
        n_B (int): Number of behavioural delay compartments.
        R_off (float): Lower threshold of the asymmetric warning trigger.
        eval_interval (float): Minimum time the warning state is kept before re-evaluation.
        T_lead (float): Lead time for which the estimated Rt trend is extrapolated.
        w_a, w_p, w_s (jnp array): Infectiousness weights.
        nE, nP, nS, nA (int): Number of compartments in the respective linear chains.
        weighted (bool): Whether the subcompartments are weighted individually (default False).
    """
    R_0: float
    beta: float
    gamma_inv: float
    sigma_inv: float
    mu_a_inv: float
    mu_s_inv: float
    p: float
    phi_a: float
    phi_p: float
    epsilon_s: float
    epsilon_w: float
    k: float
    R_crit: float
    tau_W: float
    tau_B: float
    rho: float
    I_crit: float
    k_I: float
    n_W: int
    n_B: int
    R_off: float
    eval_interval: float
    T_lead: float
    w_a: jnp.ndarray
    w_p: jnp.ndarray
    w_s: jnp.ndarray
    nE: int
    nP: int
    nS: int
    nA: int
    weighted: bool

    @classmethod
    def for_SEIPAR(cls,
            R_0: float = 2.69,
            phi_a: float = 0.26,
            phi_p: float = 3.72,
            gamma_inv: float = 3.0,
            sigma_inv: float = 2.5,
            mu_a_inv: float = 11.8,
            mu_s_inv: float =  9.3,
            p: float = 0.351,
            epsilon_s: float = 0.0,
            epsilon_w: float = 0.0,
            k: float = 10.0,
            R_crit: float = 1.0,
            tau_W: float = 14.0,
            tau_B: float = 7.0,
            I_crit: float = 0.0, 
            k_I: float = 1e6,
            n_W: int = 3,
            n_B: int = 1,
            R_off: float = 1.0,
            eval_interval: float = 14.0,
            T_lead: float = 0.0,
            shape: float = 8.0,
            scale: float = 5.5/8.0,
            nE: int = 10,
            nP: int = 10,
            nS: int = 10,
            nA: int = 10,
            weighted: bool = False,
        ) -> "ParamsErlang":
        w_p, w_s = (compute_weights(gamma_inv, sigma_inv, mu_s_inv, shape, scale, nP, nS, phi_p) if weighted else (jnp.ones(nP), jnp.ones(nS)))
        w_a = jnp.ones(nA)
        r = _r_weighted(p, phi_a, phi_p, sigma_inv, mu_a_inv, mu_s_inv, w_a, w_p, w_s, nP, nS, nA, epsilon_s=0.0)
        r_eps = _r_weighted(p, phi_a, phi_p, sigma_inv, mu_a_inv, mu_s_inv, w_a, w_p, w_s, nP, nS, nA, epsilon_s=epsilon_s)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, beta=beta, gamma_inv=gamma_inv, sigma_inv=sigma_inv, mu_a_inv=mu_a_inv,
            mu_s_inv=mu_s_inv, p=p, phi_a=phi_a, phi_p=phi_p,
            epsilon_s=epsilon_s, epsilon_w=epsilon_w, k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B,
            rho=rho, I_crit=I_crit, k_I=k_I, n_W=int(n_W), n_B=int(n_B), R_off=R_off,
            eval_interval=eval_interval, T_lead=T_lead, w_a=w_a, w_p=w_p, w_s=w_s,
            nE=int(nE), nP=int(nP), nS=int(nS), nA=int(nA), weighted=bool(weighted),
        )

    _DERIVED_FROM = ["R_0", "p", "phi_a", "phi_p", "mu_a_inv", "sigma_inv", "mu_s_inv", "epsilon_s", "w_a", "w_p", "w_s", "nP", "nS", "nA"]
    def update(self, **kwargs) -> "ParamsErlang":
        """Update any parameter(s)."""
        for f in _ERLANG_STATIC_FIELDS:
            if f in kwargs:
                kwargs[f] = bool(kwargs[f]) if f == "weighted" else int(kwargs[f])
        if set(self._DERIVED_FROM) & kwargs.keys():
            v = {f: kwargs.get(f, getattr(self, f)) for f in self._DERIVED_FROM}
            common = dict(
                p=v["p"], phi_a=v["phi_a"], phi_p=v["phi_p"], sigma_inv=v["sigma_inv"],
                mu_a_inv=v["mu_a_inv"], mu_s_inv=v["mu_s_inv"], w_a=v["w_a"], w_p=v["w_p"], 
                w_s=v["w_s"], nP=v["nP"], nS=v["nS"], nA=v["nA"])
            r = _r_weighted(epsilon_s=0.0, **common)
            r_eps = _r_weighted(epsilon_s=v["epsilon_s"], **common)
            kwargs.setdefault("beta", jnp.where(r > 0, v["R_0"] / r, 0.0))
            kwargs.setdefault("rho", jnp.where(r > 0, r_eps / r, 1.0))
            kwargs["R_0"] = jnp.where(r > 0, v["R_0"], 0.0)
        return self._replace(**kwargs)

_register_static_pytree(ParamsErlang, _ERLANG_STATIC_FIELDS)

def compute_weights(gamma_inv, sigma_inv, mu_s_inv, shape, scale, nP, nS, phi_p):
    """
    Infectiousness weights for the Ip and Is subcompartments.
    Computed from a Gamma pdf with given shape and scale parameters.
    """
    w_p = gamma.pdf(_mean_time(gamma_inv, sigma_inv, nP), a=shape, scale=scale)
    w_s = gamma.pdf(_mean_time(gamma_inv + sigma_inv, mu_s_inv, nS), a=shape, scale=scale)
    norm = (phi_p * sigma_inv + mu_s_inv) / (phi_p * jnp.sum(w_p) * (sigma_inv / nP) + jnp.sum(w_s) * (mu_s_inv / nS))
    return norm * w_p, norm * w_s

def _mean_time(t0, mean, n):
    """
    Mean time since infection at the midpoint of each subcompartment:
        t0 + (i - 0.5) * mean/n for i = 1, ..., n.
    """
    return t0 + (jnp.arange(1, n+1) - 0.5) * (mean/n)

def _r_weighted(p, phi_a, phi_p, sigma_inv, mu_a_inv, mu_s_inv, w_a, w_p, w_s, nP, nS, nA, epsilon_s=0.0):
    """
    Weighted infectiousness sum r = R_0 / beta for the Erlang model.
    The contribution from each type is:
    probability of the route * relative infectiousness * compartment weights * mean time.
    The mean time per subcompartment is the total compartment chain period divided by the number of subcompartments.
    """
    ra = p * phi_a * jnp.sum(w_a) * (mu_a_inv / nA)
    rp = (1.0 - p) * phi_p * jnp.sum(w_p) * (sigma_inv / nP)
    rs = (1.0 - p) * (1.0 - epsilon_s) * jnp.sum(w_s) * (mu_s_inv / nS)
    return ra + rp + rs
