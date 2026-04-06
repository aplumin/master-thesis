"""
TODO: use flexible length array for delay states. (maybe with mean and var of Erlang/Gamma)
TODO: Start interventions only if infections higher than some threshold
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

from models.parameters import Params, f


def SEIPAR_W(t, y, params):
    """Compartmental model with linear feedback chain for delayed wastewater response."""
    S, E, Ia, Ip, Is, R, W1, W2, W3 = y
    f_W3 = f(W3, params)
    lambda_S = f_W3 * params.beta * (params.phi * Ia + Ip + (1.0 - params.epsilon_s) * Is) * S
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

    # delay compartments
    delay_rate = 3.0 / params.tau # TODO: make num_compartments flexible (fit to empirical delay distributions)
    Rt = params.R_0 * params.rho * f_W3 * S # use current Rt as input to the linear chain
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)

    return jnp.array([dS, dE, dIa, dIp, dIs, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIPAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6):
    solution = diffeqsolve(
        ODETerm(SEIPAR_W), Tsit5(),
        t0 = 0.0, t1 = t1,  dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


def SEIAR_W(t, y, params):
    """Simplified compartmental model without presymptomatic transmission."""
    S, E, Ia, Is, R, W1, W2, W3 = y
    f_W3 = f(W3, params)
    lambda_S = f_W3 * params.beta * (params.phi * Ia + (1.0 - params.epsilon_s) * Is) * S
    become_infectious = E / params.gamma_inv
    recover_asyx = Ia / params.mu_a_inv
    recover_syx = Is / params.mu_s_inv

    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dIa = params.p * become_infectious - recover_asyx
    dIs = (1.0 - params.p) * become_infectious - recover_syx
    dR = recover_asyx + recover_syx

    # delay compartments
    delay_rate = 3.0 / params.tau
    Rt = params.R_0 * params.rho * f_W3 * S # use current Rt as input to the linear chain
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)
    return jnp.array([dS, dE, dIa, dIs, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6):
    solution = diffeqsolve(
        ODETerm(SEIAR_W), Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


def SEIR_W(t, y, params):
    """Simplified compartmental model without presymptomatic or asymptomatic transmission."""
    S, E, II, R, W1, W2, W3 = y
    f_W3 = f(W3, params)
    lambda_S = f_W3 * params.beta * (1.0 - params.epsilon_s) * II * S
    become_infectious = E / params.gamma_inv
    recover = II / params.mu_s_inv

    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dI = become_infectious - recover
    dR = recover

    # delay compartments
    delay_rate = 3.0 / params.tau
    Rt = params.R_0 * params.rho * f_W3 * S # use current Rt as input to the linear chain
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)
    return jnp.array([dS, dE, dI, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6):
    solution = diffeqsolve(
        ODETerm(SEIR_W), Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


# TODO: integrate into main functions (set gate_I to 1)
def SEIPAR_W_with_I_gate(t, y, args):
    """Compartmental model with gated wastewater response."""
    params, I_crit, k_I = args
    S, E, Ia, Ip, Is, R, W1, W2, W3 = y

    # infection gate
    II = Ia + Ip + Is
    gate_I = 1.0 / (1.0 + jnp.exp(-k_I * (II - I_crit)))
    logistic_term_W = 1.0 / (1.0 + jnp.exp(-params.k * (W3 - params.R_crit)))
    f_W3 = 1.0 - params.epsilon_w * logistic_term_W * gate_I
    
    lambda_S = f_W3 * params.beta * (params.phi * Ia + Ip + (1.0 - params.epsilon_s) * Is) * S
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

    # delay compartments
    delay_rate = 3.0 / params.tau
    Rt = params.R_0 * params.rho * f_W3 * S 
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)

    return jnp.array([dS, dE, dIa, dIp, dIs, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIPAR_W_with_I_gate(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, I_crit: float = 0.001, k_I: float = 10000.0):
    solution = diffeqsolve(
        ODETerm(SEIPAR_W_with_I_gate), Tsit5(),
        t0 = 0.0, t1 = t1,  dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = (params, I_crit, k_I),
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys
