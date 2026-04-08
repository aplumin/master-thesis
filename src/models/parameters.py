"""
Parameter class for compartmental models and utility functions.
"""

import jax.numpy as jnp
from typing import NamedTuple


class Params(NamedTuple):
    """Parameters for compartmental models."""
    R_0: float       # basic reproductive number
    beta: float      # transmission rate
    gamma_inv: float # exposed period (inverse of become infectious rate)
    sigma_inv: float # presymptomatic period (inverse of become symptomatic rate)
    mu_a_inv: float  # asymptomatic period (inverse of recovery rate)
    mu_s_inv: float  # symptomatic period (inverse of recovery rate)
    p: float         # proportion asymptomatic
    phi: float       # relative infectiousness
    epsilon_s: float # isolation efficacy
    epsilon_w: float # contact rate reduction efficacy after warning response
    k: float         # sharpness of warning response
    R_crit: float    # Rt threshold for warnings
    tau: float       # reporting delay
    rho: float       # isolation reduction factor
    # TODO: add I_crit and k_I params
    # I_crit: float    # infection threshold for intervention
    # k_I: float       # sharpness of gate for infection threshold for interventions
    
    @classmethod
    def for_SEIPAR(cls,
            R_0: float = 2.5,
            phi: float = 0.1,
            gamma_inv: float = 3.0,
            sigma_inv: float = 2.5,
            mu_a_inv: float = 5.0,
            mu_s_inv: float =  3.0,
            p: float = 0.4,
            epsilon_s: float = 0.0,
            epsilon_w: float = 0.0,
            k: float = 1.0,
            R_crit: float = 1.0,
            tau: float = 7.0,
            # I_crit: float = 0.0, 
            # k_I: float = 100.0
        ) -> "Params":
        """
        Parameters for the full model with presymptomatic and asymptomatic transmission.
        Uses SARS-CoV-2 parameters by default.
        """
        r = p * phi * mu_a_inv + (1-p)*(sigma_inv + mu_s_inv)
        r_eps = p * phi * mu_a_inv + (1-p) * (sigma_inv + (1-epsilon_s) * mu_s_inv)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, phi=phi, beta=beta, gamma_inv=gamma_inv, sigma_inv=sigma_inv, 
            mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, epsilon_s=epsilon_s, 
            epsilon_w=epsilon_w, k=k, R_crit=R_crit, tau=tau, rho=rho
        )

    @classmethod
    def for_SEIAR(cls,
            R_0: float = 1.5,
            phi: float = 0.5,
            gamma_inv: float = 2.0,
            mu_a_inv: float = 3.5,
            mu_s_inv: float = 3.5,
            p: float = 0.4,
            epsilon_s: float = 0.0,
            epsilon_w: float = 0.0,
            k: float = 1.0,
            R_crit: float = 1.0,
            tau: float = 7.0,
        ) -> "Params":
        """
        Parameters for the SEIAR model with asymptomatic but no presymptomatic transmission.
        Uses Influenza A parameters by default.
        """
        r = p * phi * mu_a_inv + (1-p) * mu_s_inv
        r_eps = p * phi * mu_a_inv + (1-p) * ((1-epsilon_s) * mu_s_inv)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, phi=phi, beta=beta, gamma_inv=gamma_inv, sigma_inv=0.0, 
            mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, epsilon_s=epsilon_s, 
            epsilon_w=epsilon_w, k=k, R_crit=R_crit, tau=tau, rho=rho
        )

    @classmethod
    def for_SEIR(cls, # ebola params
            R_0: float = 2.0,
            gamma_inv: float = 11.0,
            mu_s_inv: float = 7.0,
            epsilon_s: float = 0.0,
            epsilon_w: float = 0.0,
            k: float = 1.0,
            R_crit: float = 1.0,
            tau: float = 7.0,
        ) -> "Params":
        """
        Parameters for the SEIR model without asymptomatic or presymptomatic transmission.
        Uses Ebola parameters by default.
        """
        beta = R_0 / mu_s_inv
        rho = 1 - epsilon_s
        return cls(
            R_0=R_0, phi=0.0, beta=beta, gamma_inv=gamma_inv, sigma_inv=0.0, 
            mu_a_inv=0.0, mu_s_inv=mu_s_inv, p=0.0, epsilon_s=epsilon_s, 
            epsilon_w=epsilon_w, k=k, R_crit=R_crit, tau=tau, rho=rho
        )

def update_epsilons(params: Params, epsilon_w: float, epsilon_s: float) -> Params:
    """Update NPI efficacy parameters epsilon for a given parameter set."""
    r = params.p * params.phi * params.mu_a_inv + (1-params.p) * (params.sigma_inv + params.mu_s_inv)
    r_eps = params.p * params.phi * params.mu_a_inv + (1-params.p) * (params.sigma_inv + (1-epsilon_s) * params.mu_s_inv)
    rho = r_eps / r
    beta = params.R_0 / r
    return params._replace(epsilon_w=epsilon_w, epsilon_s=epsilon_s, rho=rho, beta=beta)

def update_asymptomatic_params(params: Params, p: float, phi: float):
    """Update asymptomatic parameters for a given parameter set."""
    r = p * phi * params.mu_a_inv + (1-p) * (params.sigma_inv + params.mu_s_inv)
    r_eps = p * phi * params.mu_a_inv + (1-p) * (params.sigma_inv + (1-params.epsilon_s) * params.mu_s_inv)
    rho = r_eps / r
    beta = params.R_0 / r
    return params._replace(p=p, phi=phi, rho=rho, beta=beta)

# TODO: include I_crit gate
def f(reproductive_number, params): # TODO: rename
    """Logistic response function of the reproductive number for the wastewater warning response."""
    logistic_term = 1.0 / (1.0 + jnp.exp(-params.k * (reproductive_number - params.R_crit)))
    return 1 - (params.epsilon_w * logistic_term)
