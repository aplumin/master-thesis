"""
Parameter class for compartmental models and utility functions.
"""

import jax.numpy as jnp
from typing import NamedTuple


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
        phi (float): Relative infectiousness.
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
    phi: float
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
            phi: float = 0.32,
            gamma_inv: float = 3.2,
            sigma_inv: float = 2.3,
            mu_a_inv: float = 5.0,
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
        r = _calculate_r(p=p, phi=phi, mu_a_inv=mu_a_inv, sigma_inv=sigma_inv, mu_s_inv=mu_s_inv)
        r_eps = _calculate_r_eps(p=p, phi=phi, mu_a_inv=mu_a_inv, sigma_inv=sigma_inv, epsilon_s=epsilon_s, mu_s_inv=mu_s_inv)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, phi=phi, beta=beta, gamma_inv=gamma_inv, sigma_inv=sigma_inv, 
            mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, epsilon_s=epsilon_s, epsilon_w=epsilon_w, 
            k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B, rho=rho, I_crit=I_crit, k_I=k_I,
            n_W=n_W, n_B=n_B, R_off=R_off, eval_interval=eval_interval, T_lead=T_lead
        )

    @classmethod
    def for_SEIAR(cls,
            R_0: float = 1.46,
            phi: float = 0.57,
            gamma_inv: float = 1.65,
            mu_a_inv: float = 3.38,
            mu_s_inv: float = 3.38,
            p: float = 0.36,
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
        Parameters for the SEIAR model with asymptomatic but no presymptomatic transmission.
        Uses Influenza A parameters by default.
        """
        r = _calculate_r(p=p, phi=phi, mu_a_inv=mu_a_inv, sigma_inv=0.0, mu_s_inv=mu_s_inv)
        r_eps = _calculate_r_eps(p=p, phi=phi, mu_a_inv=mu_a_inv, sigma_inv=0.0, epsilon_s=epsilon_s, mu_s_inv=mu_s_inv)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, phi=phi, beta=beta, gamma_inv=gamma_inv, sigma_inv=0.0, 
            mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, epsilon_s=epsilon_s, epsilon_w=epsilon_w, 
            k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B, rho=rho, I_crit=I_crit, k_I=k_I,
            n_W=n_W, n_B=n_B, R_off=R_off, eval_interval=eval_interval, T_lead=T_lead
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
        beta = R_0 / mu_s_inv
        rho = 1 - epsilon_s
        return cls(
            R_0=R_0, phi=0.0, beta=beta, gamma_inv=gamma_inv, sigma_inv=0.0, 
            mu_a_inv=0.0, mu_s_inv=mu_s_inv, p=0.0, epsilon_s=epsilon_s, epsilon_w=epsilon_w, 
            k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B, rho=rho, I_crit=I_crit, k_I=k_I,
            n_W=n_W, n_B=n_B, R_off=R_off, eval_interval=eval_interval, T_lead=T_lead
        )

    def update(self, **kwargs) -> "Params":
        """Update any parameter(s)."""
        base_params = {"R_0", "p", "phi", "mu_a_inv", "sigma_inv", "mu_s_inv", "epsilon_s",}
        if base_params & kwargs.keys():
            v = {f: kwargs.get(f, getattr(self, f)) for f in base_params}
            r = _calculate_r(p=v["p"], phi=v["phi"], mu_a_inv=v["mu_a_inv"], sigma_inv=v["sigma_inv"], mu_s_inv=v["mu_s_inv"])
            r_eps = _calculate_r_eps(p=v["p"], phi=v["phi"], mu_a_inv=v["mu_a_inv"], sigma_inv=v["sigma_inv"], mu_s_inv=v["mu_s_inv"], epsilon_s=v["epsilon_s"])
            kwargs.setdefault("beta", jnp.where(r > 0, v["R_0"]/r, 0.0))
            kwargs.setdefault("rho",  jnp.where(r > 0, r_eps/r, 1.0))
            kwargs["R_0"] = jnp.where(r > 0, v["R_0"], 0.0)
        return self._replace(**kwargs)


def logistic_response_function(reproductive_number: float, params: Params, number_infected: float):
    """Logistic response function of the reproductive number for the wastewater warning response."""
    gate_W = 1.0 / (1.0 + jnp.exp(-params.k * (reproductive_number - params.R_crit)))
    gate_I = jnp.where( # no effect if threshold set to 0
        params.I_crit > 0.0, 
        1.0 / (1.0 + jnp.exp(-params.k_I * (number_infected - params.I_crit))), 
        1.0
    )
    return 1.0 - params.epsilon_w * gate_W * gate_I

def _calculate_r_eps(p, phi, mu_a_inv, sigma_inv, epsilon_s, mu_s_inv):
    return p * phi * mu_a_inv + (1-p) * (sigma_inv + (1-epsilon_s) * mu_s_inv)

def _calculate_r(p, phi, mu_a_inv, sigma_inv, mu_s_inv):
    return p * phi * mu_a_inv + (1-p) * (sigma_inv + mu_s_inv)
