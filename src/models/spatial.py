"""
Spatial metapopulation model.
"""

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
from functools import partial
from typing import NamedTuple

from models.parameters import Params, logistic_response_function


class SpatialParams(NamedTuple):
    """Parameters for spatial metapopulation model with two demes, A and B."""
    epi_params: Params
    N_A: float
    m: float
    ww_in_B: bool = False

    def update(self, **kwargs) -> "SpatialParams":
        fields = set(Params._fields)
        return self._replace(
            epi_params = self.epi_params.update(**{k: v for k, v in kwargs.items() if k in fields}),
            **{k: v for k, v in kwargs.items() if k not in fields}
        )

@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'ww_in_B', 'response_in_B_to_A', 'n_ts'])
def simulate_SEIPAR_W_spatial(
    spatial_params: SpatialParams = SpatialParams(epi_params=Params.for_SEIPAR(), N_A=1.0, m=0.0),
    t1: float = 200.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1, n_ts: int = 5000,
    primary_in_A: bool = True, ww_in_B: bool = False, response_in_B_to_A: bool = False,
):
    """
    Two-deme SEIPAR model with migration.
    States: S_A, E_A, Ia_A, Ip_A, Is_A, R_A, S_B, E_B, Ia_B, Ip_A, Is_B, R_B, W_A(n_W), B_A(n_B), [W_B(n_W), B_B(n_B)]
    The W_B/B_B chains are only included if ww_in_B=True or response_in_B_to_A=True.
    """
    epi_params = spatial_params.epi_params
    N_A = spatial_params.N_A
    N_B = 1.0 - N_A
    m_AB = spatial_params.m
    m_BA = jnp.where(N_B > 0.0, spatial_params.m * N_A/N_B, 0.0)

    def _SEIAR_spatial(t, y, args):
        # unpack compartments
        idx = 12
        S_A, E_A, Ia_A, Ip_A, Is_A, R_A, S_B, E_B, Ia_B, Ip_B, Is_B, R_B = y[:idx]
        W_A = y[idx:idx + n_W]; idx += n_W
        B_A = y[idx:idx + n_B]; idx += n_B
        if ww_in_B:
            W_B = y[idx:idx + n_W]; idx += n_W
            B_B = y[idx:idx + n_B]; idx += n_B

        W_out_A = W_A[-1]
        B_out_A = B_A[-1]
        B_out_B = B_B[-1] if ww_in_B else jnp.full_like(B_out_A, 1.0)
        if response_in_B_to_A:
            B_out_B = jnp.copy(B_A[-1])

        # compute mass flows
        prevalence_syx_A = jnp.where(N_A > 0.0, Is_A / N_A, 0.0)
        prevalence_syx_B = jnp.where(N_B > 0.0, Is_B / N_B, 0.0)
        prevalence_asyx_A = jnp.where(N_A > 0.0, Ia_A / N_A, 0.0)
        prevalence_asyx_B = jnp.where(N_B > 0.0, Ia_B / N_B, 0.0)
        prevalence_presyx_A = jnp.where(N_A > 0.0, Ip_A / N_A, 0.0)
        prevalence_presyx_B = jnp.where(N_B > 0.0, Ip_B / N_B, 0.0)
        lambda_A = B_out_A * epi_params.beta * (epi_params.phi * prevalence_asyx_A + prevalence_presyx_A + (1.0 - epi_params.epsilon_s) * prevalence_syx_A) * S_A
        lambda_B = B_out_B * epi_params.beta * (epi_params.phi * prevalence_asyx_B + prevalence_presyx_B + (1.0 - epi_params.epsilon_s) * prevalence_syx_B) * S_B

        become_infectious_A = E_A / epi_params.gamma_inv
        become_infectious_B = E_B / epi_params.gamma_inv
        become_symptomatic_A = Ip_A / epi_params.sigma_inv
        become_symptomatic_B = Ip_B / epi_params.sigma_inv
        recover_asyx_A = Ia_A / epi_params.mu_a_inv
        recover_asyx_B = Ia_B / epi_params.mu_a_inv
        recover_syx_A = Is_A / epi_params.mu_s_inv
        recover_syx_B = Is_B / epi_params.mu_s_inv

        def migration(X_A, X_B):
            return m_BA * X_B - m_AB * X_A

        # flow compartments
        dS_A = -lambda_A + migration(S_A, S_B)
        dE_A = lambda_A - become_infectious_A + migration(E_A, E_B)
        dIa_A = epi_params.p * become_infectious_A - recover_asyx_A + migration(Ia_A, Ia_B)
        dIp_A = (1.0 - epi_params.p) * become_infectious_A - become_symptomatic_A + migration(Ip_A, Ip_B)
        dIs_A = become_symptomatic_A - recover_syx_A + migration(Is_A, Is_B)
        dR_A = recover_asyx_A + recover_syx_A + migration(R_A, R_B)

        dS_B = -lambda_B - migration(S_A, S_B)
        dE_B = lambda_B - become_infectious_B - migration(E_A, E_B)
        dIa_B = epi_params.p * become_infectious_B - recover_asyx_B - migration(Ia_A, Ia_B)
        dIp_B = (1.0 - epi_params.p) * become_infectious_B - become_symptomatic_B - migration(Ip_A, Ip_B)
        dIs_B = become_symptomatic_B - recover_syx_B - migration(Is_A, Is_B)
        dR_B = recover_asyx_B + recover_syx_B - migration(R_A, R_B)

        dFlow = jnp.array([dS_A, dE_A, dIa_A, dIp_A, dIs_A, dR_A, dS_B, dE_B, dIa_B, dIp_B, dIs_B, dR_B])

        # reporting and behavioural delay in A
        reporting_delay_rate = n_W / epi_params.tau_W
        Rt_A = jnp.where(N_A > 0.0, epi_params.R_0 * epi_params.rho * B_out_A * S_A/N_A, 0.0)
        W_in_A = jnp.concatenate([jnp.array([Rt_A]), W_A[:-1]])
        dW_A = reporting_delay_rate * (W_in_A - W_A)

        behavioural_delay_rate = n_B / epi_params.tau_B
        Rt_reported_A = logistic_response_function(W_out_A, epi_params, prevalence_syx_A)
        B_in_A = jnp.concatenate([jnp.array([Rt_reported_A]), B_A[:-1]])
        dB_A = behavioural_delay_rate * (B_in_A - B_A)

        ODE_list = [dFlow, dW_A, dB_A]

        # reporting and behavioural delay in B
        if ww_in_B:
            W_out_B = W_B[-1]
            Rt_B = jnp.where(N_B > 0.0, epi_params.R_0 * epi_params.rho * B_out_B * S_B/N_B, 0.0)
            W_in_B = jnp.concatenate([jnp.array([Rt_B]), W_B[:-1]])
            dW_B = reporting_delay_rate * (W_in_B - W_B)
            Rt_reported_B = logistic_response_function(W_out_B, epi_params, prevalence_syx_B)
            B_in_B = jnp.concatenate([jnp.array([Rt_reported_B]), B_B[:-1]])
            dB_B = behavioural_delay_rate * (B_in_B - B_B)
            ODE_list += [dW_B, dB_B]

        return jnp.concatenate(ODE_list)

    # primary case can be in A or B
    E0_A = jnp.where(primary_in_A, E0 * N_A, 0.0)
    E0_B = jnp.where(primary_in_A, 0.0, E0 * N_B)
    
    # initialisation
    y0_flow = jnp.array([
        N_A-E0_A, E0_A, 0.0, 0.0, 0.0, 0.0, # deme A
        N_B-E0_B, E0_B, 0.0, 0.0, 0.0, 0.0, # deme B
    ])
    if ww_in_B: # delay chains for both
        y0 = jnp.concatenate([y0_flow, jnp.zeros(n_W), jnp.ones(n_B), jnp.zeros(n_W), jnp.ones(n_B)])
    else: # delay chains for A only
        y0 = jnp.concatenate([y0_flow, jnp.zeros(n_W), jnp.ones(n_B)])

    solution = diffeqsolve(
        terms=ODETerm(_SEIAR_spatial),
        solver=Tsit5(),
        t0=0.0, t1=t1, dt0=0.1,
        y0=y0,
        args=None,
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, n_ts)),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), max_steps=50_000,
    )
    return solution.ts, solution.ys


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'ww_in_B', 'response_in_B_to_A', 'n_ts'])
def simulate_SEIAR_W_spatial(
    spatial_params: SpatialParams = SpatialParams(epi_params=Params.for_SEIAR(), N_A=1.0, m=0.0),
    t1: float = 200.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1, n_ts: int = 5000,
    primary_in_A: bool = True, ww_in_B: bool = False, response_in_B_to_A: bool = False,
):
    """
    Two-deme SEIAR model with migration.
    States: S_A, E_A, Ia_A, Is_A, R_A, S_B, E_B, Ia_B, Is_B, R_B, W_A(n_W), B_A(n_B), [W_B(n_W), B_B(n_B)]
    The W_B/B_B chains are only included if ww_in_B=True or response_in_B_to_A=True.
    """
    epi_params = spatial_params.epi_params
    N_A = spatial_params.N_A
    N_B = 1.0 - N_A
    m_AB = spatial_params.m
    m_BA = jnp.where(N_B > 0.0, spatial_params.m * N_A/N_B, 0.0)

    def _SEIAR_spatial(t, y, args):
        # unpack compartments
        idx = 10
        S_A, E_A, Ia_A, Is_A, R_A, S_B, E_B, Ia_B, Is_B, R_B = y[:idx]
        W_A = y[idx:idx + n_W]; idx += n_W
        B_A = y[idx:idx + n_B]; idx += n_B
        if ww_in_B:
            W_B = y[idx:idx + n_W]; idx += n_W
            B_B = y[idx:idx + n_B]; idx += n_B

        W_out_A = W_A[-1]
        B_out_A = B_A[-1]
        B_out_B = B_B[-1] if ww_in_B else jnp.full_like(B_out_A, 1.0)
        if response_in_B_to_A:
            B_out_B = jnp.copy(B_A[-1])

        # compute mass flows
        prevalence_syx_A = jnp.where(N_A > 0.0, Is_A / N_A, 0.0)
        prevalence_syx_B = jnp.where(N_B > 0.0, Is_B / N_B, 0.0)
        prevalence_asyx_A = jnp.where(N_A > 0.0, Ia_A / N_A, 0.0)
        prevalence_asyx_B = jnp.where(N_B > 0.0, Ia_B / N_B, 0.0)
        lambda_A = B_out_A * epi_params.beta * (epi_params.phi * prevalence_asyx_A + (1.0 - epi_params.epsilon_s) * prevalence_syx_A) * S_A
        lambda_B = B_out_B * epi_params.beta * (epi_params.phi * prevalence_asyx_B + (1.0 - epi_params.epsilon_s) * prevalence_syx_B) * S_B

        become_infectious_A = E_A / epi_params.gamma_inv
        become_infectious_B = E_B / epi_params.gamma_inv
        recover_asyx_A = Ia_A / epi_params.mu_a_inv
        recover_asyx_B = Ia_B / epi_params.mu_a_inv
        recover_syx_A = Is_A / epi_params.mu_s_inv
        recover_syx_B = Is_B / epi_params.mu_s_inv

        def migration(X_A, X_B):
            return m_BA * X_B - m_AB * X_A

        # flow compartments
        dS_A = -lambda_A + migration(S_A, S_B)
        dE_A = lambda_A - become_infectious_A + migration(E_A, E_B)
        dIa_A = epi_params.p * become_infectious_A - recover_asyx_A + migration(Ia_A, Ia_B)
        dIs_A = (1.0 - epi_params.p) * become_infectious_A - recover_syx_A + migration(Is_A, Is_B)
        dR_A = recover_asyx_A + recover_syx_A + migration(R_A, R_B)

        dS_B = -lambda_B - migration(S_A, S_B)
        dE_B = lambda_B - become_infectious_B - migration(E_A, E_B)
        dIa_B = epi_params.p * become_infectious_B - recover_asyx_B - migration(Ia_A, Ia_B)
        dIs_B = (1.0 - epi_params.p) * become_infectious_B - recover_syx_B - migration(Is_A, Is_B)
        dR_B = recover_asyx_B + recover_syx_B - migration(R_A, R_B)

        dFlow = jnp.array([dS_A, dE_A, dIa_A, dIs_A, dR_A, dS_B, dE_B, dIa_B, dIs_B, dR_B])

        # reporting and behavioural delay in A
        reporting_delay_rate = n_W / epi_params.tau_W
        Rt_A = jnp.where(N_A > 0.0, epi_params.R_0 * epi_params.rho * B_out_A * S_A/N_A, 0.0)
        W_in_A = jnp.concatenate([jnp.array([Rt_A]), W_A[:-1]])
        dW_A = reporting_delay_rate * (W_in_A - W_A)

        behavioural_delay_rate = n_B / epi_params.tau_B
        Rt_reported_A = logistic_response_function(W_out_A, epi_params, prevalence_syx_A)
        B_in_A = jnp.concatenate([jnp.array([Rt_reported_A]), B_A[:-1]])
        dB_A = behavioural_delay_rate * (B_in_A - B_A)

        ODE_list = [dFlow, dW_A, dB_A]

        # reporting and behavioural delay in B
        if ww_in_B:
            W_out_B = W_B[-1]
            Rt_B = jnp.where(N_B > 0.0, epi_params.R_0 * epi_params.rho * B_out_B * S_B/N_B, 0.0)
            W_in_B = jnp.concatenate([jnp.array([Rt_B]), W_B[:-1]])
            dW_B = reporting_delay_rate * (W_in_B - W_B)
            Rt_reported_B = logistic_response_function(W_out_B, epi_params, prevalence_syx_B)
            B_in_B = jnp.concatenate([jnp.array([Rt_reported_B]), B_B[:-1]])
            dB_B = behavioural_delay_rate * (B_in_B - B_B)
            ODE_list += [dW_B, dB_B]

        return jnp.concatenate(ODE_list)

    # primary case can be in A or B
    E0_A = jnp.where(primary_in_A, E0 * N_A, 0.0)
    E0_B = jnp.where(primary_in_A, 0.0, E0 * N_B)
    
    # initialisation
    y0_flow = jnp.array([
        N_A-E0_A, E0_A, 0.0, 0.0, 0.0, # deme A
        N_B-E0_B, E0_B, 0.0, 0.0, 0.0, # deme B
    ])
    if ww_in_B: # delay chains for both
        y0 = jnp.concatenate([y0_flow, jnp.zeros(n_W), jnp.ones(n_B), jnp.zeros(n_W), jnp.ones(n_B)])
    else: # delay chains for A only
        y0 = jnp.concatenate([y0_flow, jnp.zeros(n_W), jnp.ones(n_B)])

    solution = diffeqsolve(
        terms=ODETerm(_SEIAR_spatial),
        solver=Tsit5(),
        t0=0.0, t1=t1, dt0=0.1,
        y0=y0,
        args=None,
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, n_ts)),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), max_steps=50_000,
    )
    return solution.ts, solution.ys


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'ww_in_B', 'response_in_B_to_A', 'n_ts'])
def simulate_SEIR_W_spatial(
    spatial_params: SpatialParams = SpatialParams(epi_params=Params.for_SEIR(), N_A=1.0, m=0.0),
    t1: float = 200.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1, n_ts: int = 5000,
    primary_in_A: bool = True, ww_in_B: bool = False, response_in_B_to_A: bool = False,
):
    """
    Two-deme SEIR model with migration.
    States: S_A, E_A, I_A, R_A, S_B, E_B, I_B, R_B, W_A(n_W), B_A(n_B), [W_B(n_W), B_B(n_B)]
    The W_B/B_B chains are only included if ww_in_B=True or response_in_B_to_A=True.
    """
    epi_params = spatial_params.epi_params
    N_A = spatial_params.N_A
    N_B = 1.0 - N_A
    m_AB = spatial_params.m
    m_BA = jnp.where(N_B > 0.0, spatial_params.m * N_A/N_B, 0.0)

    def _SEIR_spatial(t, y, args):
        # unpack compartments
        idx = 8
        S_A, E_A, I_A, R_A, S_B, E_B, I_B, R_B = y[:idx]
        W_A = y[idx:idx + n_W]; idx += n_W
        B_A = y[idx:idx + n_B]; idx += n_B
        if ww_in_B:
            W_B = y[idx:idx + n_W]; idx += n_W
            B_B = y[idx:idx + n_B]; idx += n_B

        W_out_A = W_A[-1]
        B_out_A = B_A[-1]
        B_out_B = B_B[-1] if ww_in_B else jnp.full_like(B_out_A, 1.0)
        if response_in_B_to_A:
            B_out_B = jnp.copy(B_A[-1])

        # compute mass flows
        prevalence_A = jnp.where(N_A > 0.0, I_A / N_A, 0.0)
        prevalence_B = jnp.where(N_B > 0.0, I_B / N_B, 0.0)
        lambda_A = B_out_A * epi_params.beta * (1.0 - epi_params.epsilon_s) * prevalence_A * S_A
        lambda_B = B_out_B * epi_params.beta * (1.0 - epi_params.epsilon_s) * prevalence_B * S_B

        become_infectious_A = E_A / epi_params.gamma_inv
        become_infectious_B = E_B / epi_params.gamma_inv
        recover_A = I_A / epi_params.mu_s_inv
        recover_B = I_B / epi_params.mu_s_inv

        def migration(X_A, X_B):
            return m_BA * X_B - m_AB * X_A

        # flow compartments
        dS_A = -lambda_A + migration(S_A, S_B)
        dE_A = lambda_A - become_infectious_A + migration(E_A, E_B)
        dI_A = become_infectious_A - recover_A + migration(I_A, I_B)
        dR_A = recover_A + migration(R_A, R_B)

        dS_B = -lambda_B - migration(S_A, S_B)
        dE_B = lambda_B - become_infectious_B - migration(E_A, E_B)
        dI_B = become_infectious_B - recover_B - migration(I_A, I_B)
        dR_B = recover_B - migration(R_A, R_B)

        dFlow = jnp.array([dS_A, dE_A, dI_A, dR_A, dS_B, dE_B, dI_B, dR_B])

        # reporting and behavioural delay in A
        reporting_delay_rate = n_W / epi_params.tau_W
        Rt_A = jnp.where(N_A > 0.0, epi_params.R_0 * epi_params.rho * B_out_A * S_A/N_A, 0.0)
        W_in_A = jnp.concatenate([jnp.array([Rt_A]), W_A[:-1]])
        dW_A = reporting_delay_rate * (W_in_A - W_A)

        behavioural_delay_rate = n_B / epi_params.tau_B
        Rt_reported_A = logistic_response_function(W_out_A, epi_params, prevalence_A)
        B_in_A = jnp.concatenate([jnp.array([Rt_reported_A]), B_A[:-1]])
        dB_A = behavioural_delay_rate * (B_in_A - B_A)

        ODE_list = [dFlow, dW_A, dB_A]

        # reporting and behavioural delay in B
        if ww_in_B:
            W_out_B = W_B[-1]
            Rt_B = jnp.where(N_B > 0.0, epi_params.R_0 * epi_params.rho * B_out_B * S_B/N_B, 0.0)
            W_in_B = jnp.concatenate([jnp.array([Rt_B]), W_B[:-1]])
            dW_B = reporting_delay_rate * (W_in_B - W_B)
            Rt_reported_B = logistic_response_function(W_out_B, epi_params, prevalence_B)
            B_in_B = jnp.concatenate([jnp.array([Rt_reported_B]), B_B[:-1]])
            dB_B = behavioural_delay_rate * (B_in_B - B_B)
            ODE_list += [dW_B, dB_B]

        return jnp.concatenate(ODE_list)

    # primary case can be in A or B
    E0_A = jnp.where(primary_in_A, E0 * N_A, 0.0)
    E0_B = jnp.where(primary_in_A, 0.0, E0 * N_B)
    
    # initialisation
    y0_flow = jnp.array([
        N_A-E0_A, E0_A, 0.0, 0.0, # deme A
        N_B-E0_B, E0_B, 0.0, 0.0, # deme B
    ])
    if ww_in_B: # delay chains for both
        y0 = jnp.concatenate([y0_flow, jnp.zeros(n_W), jnp.ones(n_B), jnp.zeros(n_W), jnp.ones(n_B)])
    else: # delay chains for A only
        y0 = jnp.concatenate([y0_flow, jnp.zeros(n_W), jnp.ones(n_B)])

    solution = diffeqsolve(
        terms=ODETerm(_SEIR_spatial),
        solver=Tsit5(),
        t0=0.0, t1=t1, dt0=0.1,
        y0=y0,
        args=None,
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, n_ts)),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), max_steps=50_000,
    )
    return solution.ts, solution.ys



def run_spatial_SEIR(N_A, m, epi_params=None, response_in_B_to_A=False, t1=1000.0, E0=1e-6):
    sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
    _, ys = simulate_SEIR_W_spatial(sp, primary_in_A=False, response_in_B_to_A=response_in_B_to_A, t1=t1, E0=E0)
    d = unpack_spatial(ys, model="SEIR")
    N_B = 1 - N_A
    Itot_A = d["R_A"][-1] / jnp.maximum(N_A, 1e-6)
    Itot_B = d["R_B"][-1] / jnp.maximum(N_B, 1e-6)
    peak_Is_A = jnp.max(d["I_A"] / jnp.maximum(N_A, 1e-6))
    peak_Is_B = jnp.max(d["I_B"] / jnp.maximum(N_B, 1e-6))
    total_infections = d["R_A"][-1] + d["R_B"][-1]
    return Itot_A, Itot_B, peak_Is_A, peak_Is_B, total_infections

def run_spatial_SEIAR(N_A, m, epi_params=None, response_in_B_to_A=False, t1=1000.0, E0=1e-6):
    sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
    _, ys = simulate_SEIAR_W_spatial(sp, primary_in_A=False, response_in_B_to_A=response_in_B_to_A, t1=t1, E0=E0)
    d = unpack_spatial(ys, model="SEIAR")
    N_B = 1 - N_A
    I_tot_A = d["Ia_A"] + d["Is_A"]
    I_tot_B = d["Ia_B"] + d["Is_B"]
    Itot_A = d["R_A"][-1] / jnp.maximum(N_A, 1e-6)
    Itot_B = d["R_B"][-1] / jnp.maximum(N_B, 1e-6)
    peak_Is_A = jnp.max(I_tot_A / jnp.maximum(N_A, 1e-6))
    peak_Is_B = jnp.max(I_tot_B / jnp.maximum(N_B, 1e-6))
    total_infections = d["R_A"][-1] + d["R_B"][-1]
    return Itot_A, Itot_B, peak_Is_A, peak_Is_B, total_infections

def run_spatial_SEIPAR(N_A, m, epi_params=None, response_in_B_to_A=False, t1=1000.0, E0=1e-6):
    sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
    _, ys = simulate_SEIPAR_W_spatial(sp, primary_in_A=False, response_in_B_to_A=response_in_B_to_A, t1=t1, E0=E0)
    d = unpack_spatial(ys, model="SEIPAR")
    N_B = 1 - N_A
    Is_A = d["Is_A"]
    Is_B = d["Is_B"]
    Itot_A = d["R_A"][-1] / jnp.maximum(N_A, 1e-6)
    Itot_B = d["R_B"][-1] / jnp.maximum(N_B, 1e-6)
    peak_Is_A = jnp.max(Is_A / jnp.maximum(N_A, 1e-6))
    peak_Is_B = jnp.max(Is_B / jnp.maximum(N_B, 1e-6))
    total_infections = d["R_A"][-1] + d["R_B"][-1]
    return Itot_A, Itot_B, peak_Is_A, peak_Is_B, total_infections

def run_spatial(N_A, m, epi_params=None, response_in_B_to_A=False, model="SEIPAR", t1=1000.0, E0=1e-6):
    if model == "SEIR": return run_spatial_SEIR(N_A, m, epi_params=epi_params, response_in_B_to_A=response_in_B_to_A, t1=t1, E0=E0)
    elif model == "SEIAR": return run_spatial_SEIAR(N_A, m, epi_params=epi_params, response_in_B_to_A=response_in_B_to_A, t1=t1, E0=E0)
    elif model == "SEIPAR": return run_spatial_SEIPAR(N_A, m, epi_params=epi_params, response_in_B_to_A=response_in_B_to_A, t1=t1, E0=E0)
    else: raise ValueError(f"Unknown model: {model}")


def unpack_spatial_SEIR(ys, n_W: int = 3, n_B: int = 1, ww_in_B: bool = False, response_in_B_to_A: bool = False):
    out = {"S_A": ys[:,0], "E_A": ys[:,1], "I_A": ys[:,2], "R_A": ys[:,3], "S_B": ys[:,4], "E_B": ys[:,5], "I_B": ys[:,6], "R_B": ys[:,7]}
    idx = 8
    out["W_A"] = ys[:,idx:idx + n_W]; idx += n_W
    out["B_A"] = ys[:,idx:idx + n_B]; idx += n_B
    if ww_in_B:
        out["W_B"] = ys[:,idx:idx + n_W]; idx += n_W
        out["B_B"] = ys[:,idx:idx + n_B]; idx += n_B
    elif response_in_B_to_A:
        out["W_B"], out["B_B"] = out["W_A"], out["B_A"]
    return out

def unpack_spatial_SEIAR(ys, n_W: int = 3, n_B: int = 1, ww_in_B: bool = False, response_in_B_to_A: bool = False):
    out = {"S_A": ys[:,0], "E_A": ys[:,1], "Ia_A": ys[:,2], "Is_A": ys[:,3], "R_A": ys[:,4], "S_B": ys[:,5], "E_B": ys[:,6], "Ia_B": ys[:,7], "Is_B": ys[:,8], "R_B": ys[:,9]}
    idx = 10
    out["W_A"] = ys[:,idx:idx + n_W]; idx += n_W
    out["B_A"] = ys[:,idx:idx + n_B]; idx += n_B
    if ww_in_B:
        out["W_B"] = ys[:,idx:idx + n_W]; idx += n_W
        out["B_B"] = ys[:,idx:idx + n_B]; idx += n_B
    elif response_in_B_to_A:
        out["W_B"], out["B_B"] = out["W_A"], out["B_A"]
    return out

def unpack_spatial_SEIPAR(ys, n_W: int = 3, n_B: int = 1, ww_in_B: bool = False, response_in_B_to_A: bool = False):
    out = {"S_A": ys[:,0], "E_A": ys[:,1], "Ia_A": ys[:,2], "Ip_A": ys[:,3], "Is_A": ys[:,4], "R_A": ys[:,5], "S_B": ys[:,6], "E_B": ys[:,7], "Ia_B": ys[:,8], "Ip_B": ys[:,9], "Is_B": ys[:,10], "R_B": ys[:,11]}
    idx = 12
    out["W_A"] = ys[:,idx:idx + n_W]; idx += n_W
    out["B_A"] = ys[:,idx:idx + n_B]; idx += n_B
    if ww_in_B:
        out["W_B"] = ys[:,idx:idx + n_W]; idx += n_W
        out["B_B"] = ys[:,idx:idx + n_B]; idx += n_B
    elif response_in_B_to_A:
        out["W_B"], out["B_B"] = out["W_A"], out["B_A"]
    return out

def unpack_spatial(ys, n_W: int = 3, n_B: int = 1, ww_in_B: bool = False, response_in_B_to_A: bool = False, model: str = "SEIR"):
    if model == "SEIR": return unpack_spatial_SEIR(ys, n_W=n_W, n_B=n_B, ww_in_B=ww_in_B, response_in_B_to_A=response_in_B_to_A)
    elif model == "SEIAR": return unpack_spatial_SEIAR(ys, n_W=n_W, n_B=n_B, ww_in_B=ww_in_B, response_in_B_to_A=response_in_B_to_A)
    elif model == "SEIPAR": return unpack_spatial_SEIPAR(ys, n_W=n_W, n_B=n_B, ww_in_B=ww_in_B, response_in_B_to_A=response_in_B_to_A)
    else: raise ValueError(f"Unknown model: {model}")
