import jax.numpy as jnp
from jax.scipy.stats import gamma 
from typing import NamedTuple


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
        w_p, w_s (jnp array): Infectiousness weights.
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
            mu_a_inv: float = 11.6,
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
        w_p, w_s = compute_weights(gamma_inv, sigma_inv, mu_s_inv, shape, scale, nP, nS) if weighted else jnp.ones(nP), jnp.ones(nS)
        r = _r_weighted(p, phi_a, phi_p, sigma_inv, mu_a_inv, mu_s_inv, w_p, w_s, nP, nS, nA, epsilon_s=0.0)
        r_eps = _r_weighted(p, phi_a, phi_p, sigma_inv, mu_a_inv, mu_s_inv, w_p, w_s, nP, nS, nA, epsilon_s=epsilon_s)
        beta = R_0 / r
        rho = r_eps / r
        return cls(
            R_0=R_0, beta=beta, gamma_inv=gamma_inv, sigma_inv=sigma_inv, mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, phi_a=phi_a,
            epsilon_s=epsilon_s, epsilon_w=epsilon_w, k=k, R_crit=R_crit, tau_W=tau_W, tau_B=tau_B, rho=rho, I_crit=I_crit, k_I=k_I,
            n_W=n_W, n_B=n_B, R_off=R_off, eval_interval=eval_interval, T_lead=T_lead, w_p=w_p, w_s=w_s, nE=nE, nP=nP, nS=nS, nA=nA,
        )

def compute_weights(gamma_inv, sigma_inv, mu_s_inv, shape, scale, nP, nS):
    w_p = gamma.pdf(x=_mean_time(gamma_inv, sigma_inv, nP), a=shape, scale=scale)
    w_s = gamma.pdf(x=_mean_time(gamma_inv+sigma_inv, mu_s_inv, nS), a=shape, scale=scale)
    return w_p, w_s

def _mean_time(t0, mean, n):
    return t0 + (jnp.arange(1, n+1) - 0.5) * (mean/n)

def _r_weighted(p, phi_a, phi_p, sigma_inv, mu_a_inv, mu_s_inv, w_p, w_s, nP, nS, nA, epsilon_s=0.0):
    ra = p * phi_a * jnp.sum(jnp.ones(nA)) * (mu_a_inv / nA)
    rp = (1.0 - p) * phi_p * jnp.sum(w_p) * (sigma_inv / nP)
    rs = (1.0 - p) * (1.0 - epsilon_s) * jnp.sum(w_s) * (mu_s_inv / nS)
    return ra + rp + rs
