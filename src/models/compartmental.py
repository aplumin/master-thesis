"""
TODO: don't hardcode times to measure effect on Rt

Deterministic compartmental models:
    - SEIPAR_W with presymptomatic and asymptomatic transmission and wastewater feedback
    - SEIAR_W with presymptomatic transmission and wastewater feedback
    - SEIR_W with no presymptomatic or asymptomatic transmission and wastewater feedback
"""

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
from functools import partial

from models.parameters import Params, logistic_response_function


def _SEIPAR_W(t, y, params):
    # unpack compartments
    S, E, Ia, Ip, Is, R = y[:6]
    W = y[6:]
    W_out = W[-1]

    # compute mass flows
    f_W_out = logistic_response_function(W_out, params, Is)
    lambda_S = f_W_out * params.beta * (params.phi * Ia + Ip + (1.0 - params.epsilon_s) * Is) * S
    become_infectious = E / params.gamma_inv
    become_symptomatic = Ip / params.sigma_inv
    recover_asyx = Ia / params.mu_a_inv
    recover_syx = Is / params.mu_s_inv
    
    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dIa = params.p * become_infectious - recover_asyx
    dIp = (1.0 - params.p) * become_infectious - become_symptomatic
    dIs = become_symptomatic - recover_syx
    dR = recover_asyx + recover_syx
    dFlow = jnp.array([dS, dE, dIa, dIp, dIs, dR])

    # delay compartments
    delay_rate = params.num_delay_compartments / params.tau
    Rt = params.R_0 * params.rho * f_W_out * S # use current Rt as input to the linear chain
    W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
    dW = delay_rate * (W_in - W)

    return jnp.concatenate([dFlow, dW])

@partial(jax.jit, static_argnames=['t1', 'num_delay_compartments'])
def simulate_SEIPAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, num_delay_compartments: int = 3):
    """Compartmental model with linear feedback chain for delayed wastewater response."""
    solution = diffeqsolve(
        terms = ODETerm(_SEIPAR_W), 
        solver = Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.concatenate([
            jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0]), 
            jnp.zeros(num_delay_compartments)
        ]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


def _SEIAR_W(t, y, params):
    # unpack compartments
    S, E, Ia, Is, R = y[:5]
    W = y[5:]
    W_out = W[-1]

    # compute mass flows
    f_W_out = logistic_response_function(W_out, params, Is)
    lambda_S = f_W_out * params.beta * (params.phi * Ia + (1.0 - params.epsilon_s) * Is) * S
    become_infectious = E / params.gamma_inv
    recover_asyx = Ia / params.mu_a_inv
    recover_syx = Is / params.mu_s_inv

    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dIa = params.p * become_infectious - recover_asyx
    dIs = (1.0 - params.p) * become_infectious - recover_syx
    dR = recover_asyx + recover_syx
    dFlow = jnp.array([dS, dE, dIa, dIs, dR])

    # delay compartments
    delay_rate = params.num_delay_compartments / params.tau
    Rt = params.R_0 * params.rho * f_W_out * S
    W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
    dW = delay_rate * (W_in - W)

    return jnp.concatenate([dFlow, dW])

@partial(jax.jit, static_argnames=['t1', 'num_delay_compartments'])
def simulate_SEIAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, num_delay_compartments: int = 3):
    """Simplified compartmental model without presymptomatic transmission."""
    solution = diffeqsolve(
        terms = ODETerm(_SEIAR_W), 
        solver = Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.concatenate([
            jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0]), 
            jnp.zeros(num_delay_compartments)
        ]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


def _SEIR_W(t, y, params):
    # unpack compartments
    S, E, II, R = y[:4]
    W = y[4:]
    W_out = W[-1]

    # compute mass flows
    f_W_out = logistic_response_function(W_out, params, II)
    lambda_S = f_W_out * params.beta * (1.0 - params.epsilon_s) * II * S
    become_infectious = E / params.gamma_inv
    recover = II / params.mu_s_inv

    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dI = become_infectious - recover
    dR = recover
    dFlow = jnp.array([dS, dE, dI, dR])

    # delay compartments
    delay_rate = params.num_delay_compartments / params.tau
    Rt = params.R_0 * params.rho * f_W_out * S
    W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
    dW = delay_rate * (W_in - W)

    return jnp.concatenate([dFlow, dW])

@partial(jax.jit, static_argnames=['t1', 'num_delay_compartments'])
def simulate_SEIR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, num_delay_compartments: int = 3):
    """Simplified compartmental model without presymptomatic or asymptomatic transmission."""
    solution = diffeqsolve(
        terms = ODETerm(_SEIR_W), 
        solver = Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.concatenate([
            jnp.array([1.0 - E0, E0, 0.0, 0.0]), 
            jnp.zeros(num_delay_compartments)
        ]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys
