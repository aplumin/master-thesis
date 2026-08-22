"""
Parameter class for compartmental models and utility functions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_DERIVED_FROM = frozenset({"R_0", "p", "phi_a", "phi_p", "mu_a_inv", "sigma_inv", "mu_s_inv", "epsilon_s"})

class Params(NamedTuple):
    """
    Parameters for compartmental models.

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

    @classmethod
    def for_SEIPAR(cls, 
            R_0: float = 2.69,
            phi_a: float = 0.252,
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
            T_lead: float = 0.0
        ) -> "Params":
        """
        Parameters for the full model with presymptomatic and asymptomatic transmission.
        Uses SARS-CoV-2 parameters by default.
        """
        # weighted infectiousness sum: r = R_0 / beta
        r = calculate_r(p=p, phi_a=phi_a, phi_p=phi_p, mu_a_inv=mu_a_inv, sigma_inv=sigma_inv, mu_s_inv=mu_s_inv)
        # weighted infectiousness sum with symptomatic isolation
        r_eps = calculate_r(p=p, phi_a=phi_a, phi_p=phi_p, mu_a_inv=mu_a_inv, sigma_inv=sigma_inv, epsilon_s=epsilon_s, mu_s_inv=mu_s_inv)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, phi_a=phi_a, phi_p=phi_p, beta=beta, gamma_inv=gamma_inv, sigma_inv=sigma_inv, 
            mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, epsilon_s=epsilon_s, epsilon_w=epsilon_w, 
            k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B, rho=rho, I_crit=I_crit, k_I=k_I,
            n_W=int(n_W), n_B=int(n_B), R_off=R_off, eval_interval=eval_interval, T_lead=T_lead
        )

    @classmethod
    def for_SEIR(cls, 
            R_0: float = 1.95,
            gamma_inv: float = 8.5,
            mu_s_inv: float = 5.0,
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
            T_lead: float = 0.0
        ) -> "Params":
        """
        Parameters for the SEIR model without asymptomatic or presymptomatic transmission.
        Uses Ebola parameters by default.
        """
        r = calculate_r(p=0.0, phi_a=0.0, phi_p=0.0, mu_a_inv=0.0, sigma_inv=0.0, mu_s_inv=mu_s_inv)
        r_eps = calculate_r(p=0.0, phi_a=0.0, phi_p=0.0, mu_a_inv=0.0, sigma_inv=0.0, mu_s_inv=mu_s_inv, epsilon_s=epsilon_s)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, phi_a=0.0, phi_p=0.0, beta=beta, gamma_inv=gamma_inv, sigma_inv=0.0, 
            mu_a_inv=0.0, mu_s_inv=mu_s_inv, p=0.0, epsilon_s=epsilon_s, epsilon_w=epsilon_w, 
            k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B, rho=rho, I_crit=I_crit, k_I=k_I,
            n_W=int(n_W), n_B=int(n_B), R_off=R_off, eval_interval=eval_interval, T_lead=T_lead
        )

    def update(self, **kwargs) -> "Params":
        """Update any parameter(s)."""
        if _DERIVED_FROM & kwargs.keys():
            v = {f: kwargs.get(f, getattr(self, f)) for f in _DERIVED_FROM}
            r = calculate_r(p=v["p"], phi_a=v["phi_a"], mu_a_inv=v["mu_a_inv"], phi_p=v["phi_p"], sigma_inv=v["sigma_inv"], mu_s_inv=v["mu_s_inv"])
            r_eps = calculate_r(p=v["p"], phi_a=v["phi_a"], mu_a_inv=v["mu_a_inv"], phi_p=v["phi_p"], sigma_inv=v["sigma_inv"], mu_s_inv=v["mu_s_inv"], epsilon_s=v["epsilon_s"])
            kwargs.setdefault("beta", jnp.where(r > 0, v["R_0"]/r, 0.0))
            kwargs.setdefault("rho",  jnp.where(r > 0, r_eps/r, 1.0))
            kwargs["R_0"] = jnp.where(r > 0, v["R_0"], 0.0)
        return self._replace(**kwargs)
    
    def concrete(self) -> "Params":
        """Untraced values for Gillespie algorithm with numba."""
        def _concretise(value):
            arr = np.asarray(value)
            if arr.ndim:
                return arr.astype(float)
            if arr.dtype == bool:
                return bool(arr)
            if np.issubdtype(arr.dtype, np.integer):
                return int(arr)
            return float(arr)
        return self._replace(**{f: _concretise(getattr(self, f)) for f in self._fields})


def _register_static_pytree(cls, static_fields=("n_W", "n_B")):
    """Register NamedTuple parameter class as JAX pytree."""
    fields = cls._fields
    dynamic = tuple(f for f in fields if f not in static_fields)
    static = tuple(f for f in fields if f in static_fields)
    def _flatten(p):
        return (tuple(getattr(p, f) for f in dynamic), tuple(getattr(p, f) for f in static))
    def _unflatten(aux, children):
        values = dict(zip(dynamic, children, strict=True))
        values.update(zip(static, aux, strict=True))
        return cls(**values)
    jax.tree_util.register_pytree_node(
        nodetype=cls, 
        flatten_func=_flatten, 
        unflatten_func=_unflatten
    )
    return cls
_register_static_pytree(Params)


def logistic_response_function(reproductive_number: float, params: Params, number_infected: float = 0.0, threshold=jnp.nan):
    """
    Logistic response function of the reproductive number for the wastewater warning response.
    The response scales the transmission rate by f = 1 - epsilon_w * gate_W * gate_I,
    where each gate is a logistic function:
        gate_W = sigma(k * (R_est - R_crit))
        gate_I = sigma(k_I * log10(I / I_crit))
    with sigma(x) = 1 / (1 + exp(-x)).
    """
    threshold = jnp.nan_to_num(threshold, nan=params.R_crit)
    gate_W = 1.0 / (1.0 + jnp.exp(-jnp.clip(params.k * (reproductive_number - threshold), -80.0, 80.0)))
    active = params.I_crit > 0.0
    exponent = jnp.clip(
        params.k_I * jnp.log10(
            jnp.where(active, jnp.maximum(number_infected, 1e-300), 1.0) / jnp.where(active, params.I_crit + 1e-30, 1.0)
            ), -80.0, 80.0)
    gate_I = jnp.where(active, 1.0 / (1.0 + jnp.exp(-exponent)), 1.0)  # no effect if threshold is 0
    return 1.0 - params.epsilon_w * gate_W * gate_I

def calculate_r(p, phi_a, phi_p, mu_a_inv, sigma_inv, mu_s_inv, epsilon_s=0.0):
    """
    Weighted infectiousness sum, r = R_0 / beta. The contribution from each type is:
    probability of the route * relative infectiousness * mean time
        asymptomatic: p * phi_a * (1/mu_a)
        presymptomatic: (1 - p) * phi_p * (1/sigma)
        symptomatic: (1 - p) * (1 - epsilon_s) * (1/mu_s)
    """
    return p * phi_a * mu_a_inv + (1-p) * (phi_p * sigma_inv + (1.0 - epsilon_s) * mu_s_inv)
