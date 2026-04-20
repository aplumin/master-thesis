"""
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


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B'])
def simulate_SEIPAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1):
    """Compartmental model with linear feedback chain for delayed wastewater response."""
    def _SEIPAR_W(t, y, params):
        # unpack compartments
        S, E, Ia, Ip, Is, R = y[:6]
        W = y[6:6+n_W]
        W_out = W[-1]
        B = y[6+n_W:]
        B_out = B[-1]

        # compute mass flows
        lambda_S = B_out * params.beta * (params.phi * Ia + Ip + (1.0 - params.epsilon_s) * Is) * S
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

        # reporting delay compartments W
        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S # use current Rt as input to the linear chain
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        # behavioural delay compartments B
        behavioural_delay_rate = n_B / params.tau_B
        Rt_reported = logistic_response_function(W_out, params, Is)
        B_in = jnp.concatenate([jnp.array([Rt_reported]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    solution = diffeqsolve(
        terms = ODETerm(_SEIPAR_W), 
        solver = Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.concatenate([
            jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0]), 
            jnp.zeros(n_W),
            jnp.ones(n_B),
        ]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B'])
def simulate_SEIAR_W(params: Params = Params.for_SEIAR(), t1: float = 100.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1):
    """Simplified compartmental model without presymptomatic transmission."""
    def _SEIAR_W(t, y, params):
        # unpack compartments
        S, E, Ia, Is, R = y[:5]
        W = y[5:5+n_W]
        W_out = W[-1]
        B = y[5+n_W:]
        B_out = B[-1]

        # compute mass flows
        lambda_S = B_out * params.beta * (params.phi * Ia + (1.0 - params.epsilon_s) * Is) * S
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

        # reporting delay compartments W
        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S # use current Rt as input to the linear chain
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        # behavioural delay compartments B
        behavioural_delay_rate = n_B / params.tau_B
        Rt_reported = logistic_response_function(W_out, params, Is)
        B_in = jnp.concatenate([jnp.array([Rt_reported]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    solution = diffeqsolve(
        terms = ODETerm(_SEIAR_W), 
        solver = Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.concatenate([
            jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0]), 
            jnp.zeros(n_W),
            jnp.ones(n_B),
        ]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B'])
def simulate_SEIR_W(params: Params = Params.for_SEIR(), t1: float = 100.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1):
    """Simplified compartmental model without presymptomatic or asymptomatic transmission."""
    def _SEIR_W(t, y, params):
        # unpack compartments
        S, E, II, R = y[:4]
        W = y[4:4+n_W]
        W_out = W[-1]
        B = y[4+n_W:]
        B_out = B[-1]

        # compute mass flows
        lambda_S = B_out * params.beta * (1.0 - params.epsilon_s) * II * S
        become_infectious = E / params.gamma_inv
        recover = II / params.mu_s_inv

        # flow compartments
        dS = -lambda_S
        dE = lambda_S - become_infectious
        dI = become_infectious - recover
        dR = recover
        dFlow = jnp.array([dS, dE, dI, dR])

        # reporting delay compartments W
        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S # use current Rt as input to the linear chain
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        # behavioural delay compartments B
        behavioural_delay_rate = n_B / params.tau_B
        Rt_reported = logistic_response_function(W_out, params, II)
        B_in = jnp.concatenate([jnp.array([Rt_reported]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    solution = diffeqsolve(
        terms = ODETerm(_SEIR_W), 
        solver = Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.concatenate([
            jnp.array([1.0 - E0, E0, 0.0, 0.0]), 
            jnp.zeros(n_W),
            jnp.ones(n_B),
        ]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys
