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

RTOL = 1e-7
ATOL = 1e-9
MAX_STEPS = 50_000
DT0 = 0.1


def linear_chain(X, inflow, rate):
    """
    Linear compartment chain.
    Returns a tuple (updated subcompartment densities (jnp array), density flowing out of the chain (float)).

    Attributes:
        X (jnp array): Current subcompartment densities.
        inflow (float): Density flow into the chain.
        rate (float): Transition rate between the subcompartments.
    """
    outflow = rate * X
    dx = jnp.concatenate([jnp.reshape(inflow, (1,)), outflow[:-1]]) - outflow
    return dx, outflow[-1]

def chain_derivative(X, inflow, rate):
    """dX = rate * (X_in - X)."""
    return rate * (jnp.concatenate([jnp.reshape(inflow, (1,)), X[:-1]]) - X)

def _delay_ODEs(S, W, B, prevalence, params):
    """Wastewater reporting and behavioural adaptation delay compartments. Uses current Rt as input to the linear W chain."""
    Rt = params.R_0 * params.rho * B[-1] * S
    dW = chain_derivative(W, Rt, W.shape[0] / params.tau_W)
    reported = logistic_response_function(W[-1], params, prevalence)
    dB = chain_derivative(B, reported, B.shape[0] / params.tau_B)
    return dW, dB

def _initial_state(E0, n_I, n_W, n_B):
    """Initial state [1-E0, E0, n_I zeros, 0, nW zeros, nB ones]."""
    flow = jnp.concatenate([jnp.stack([1.0 - E0, E0]), jnp.zeros(n_I+1)])
    return jnp.concatenate([flow, jnp.zeros(n_W), jnp.ones(n_B)])

def _solve(diffeq, y0, params, t1, n_ts=5000):
    """diffrax ODE solve."""
    solution = diffeqsolve(
        terms=ODETerm(diffeq),
        solver=Tsit5(),
        t0=0.0, t1=t1, dt0=DT0,
        y0=y0,
        args=params,
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, n_ts)),
        stepsize_controller=PIDController(rtol=RTOL, atol=ATOL), max_steps=MAX_STEPS,
    )
    return solution.ts, solution.ys


@partial(jax.jit, static_argnames=['n_ts'])
def simulate_SEIPAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, n_ts: int = 5000):
    """Compartmental model with presymptomatic and asymptomatic transmission."""
    n_W, n_B = params.n_W, params.n_B
    def _SEIPAR_W(t, y, params):
        S, E, Ia, Ip, Is, R = y[:6]
        W = y[6:6+n_W]
        B = y[6+n_W:]
        lambda_S = B[-1] * params.beta * (params.phi_a * Ia + params.phi_p * Ip + (1.0 - params.epsilon_s) * Is) * S
        become_infectious = E / params.gamma_inv
        become_symptomatic = Ip / params.sigma_inv
        recover_asyx = Ia / params.mu_a_inv
        recover_syx = Is / params.mu_s_inv
        dFlow = jnp.stack([
            -lambda_S,
            lambda_S - become_infectious,
            params.p * become_infectious - recover_asyx,
            (1.0 - params.p) * become_infectious - become_symptomatic,
            become_symptomatic - recover_syx,
            recover_asyx + recover_syx,
        ])
        dW, dB = _delay_ODEs(S, W, B, Is, params)
        return jnp.concatenate([dFlow, dW, dB])
    return _solve(_SEIPAR_W, _initial_state(E0, n_I=3, n_W=n_W, n_B=n_B), params, t1, n_ts)


@partial(jax.jit, static_argnames=['n_ts'])
def simulate_SEIAR_W(params: Params = Params.for_SEIAR(), t1: float = 100.0, E0: float = 1e-6, n_ts: int = 5000):
    """Compartmental model without presymptomatic transmission."""
    n_W, n_B = params.n_W, params.n_B
    def _SEIAR_W(t, y, params):
        S, E, Ia, Is, R = y[:5]
        W = y[5:5 + n_W]
        B = y[5 + n_W:]
        lambda_S = B[-1] * params.beta * (params.phi_a * Ia + (1.0 - params.epsilon_s) * Is) * S
        become_infectious = E / params.gamma_inv
        recover_asyx = Ia / params.mu_a_inv
        recover_syx = Is / params.mu_s_inv
        dFlow = jnp.stack([
            -lambda_S,
            lambda_S - become_infectious,
            params.p * become_infectious - recover_asyx,
            (1.0 - params.p) * become_infectious - recover_syx,
            recover_asyx + recover_syx,
        ])
        dW, dB = _delay_ODEs(S, W, B, Is, params)
        return jnp.concatenate([dFlow, dW, dB])
    return _solve(_SEIAR_W, _initial_state(E0, n_I=2, n_W=n_W, n_B=n_B), params, t1, n_ts)


@partial(jax.jit, static_argnames=['n_ts'])
def simulate_SEIR_W(params: Params = Params.for_SEIR(), t1: float = 100.0, E0: float = 1e-6, n_ts: int = 5000):
    """Simplified compartmental model without presymptomatic or asymptomatic transmission."""
    n_W, n_B = params.n_W, params.n_B
    def _SEIR_W(t, y, params):
        S, E, II, R = y[:4]
        W = y[4:4 + n_W]
        B = y[4 + n_W:]
        lambda_S = B[-1] * params.beta * (1.0 - params.epsilon_s) * II * S
        become_infectious = E / params.gamma_inv
        recover = II / params.mu_s_inv
        dFlow = jnp.stack([
            -lambda_S,
            lambda_S - become_infectious,
            become_infectious - recover,
            recover,
        ])
        dW, dB = _delay_ODEs(S, W, B, II, params)
        return jnp.concatenate([dFlow, dW, dB])
    return _solve(_SEIR_W, _initial_state(E0, n_I=1, n_W=n_W, n_B=n_B), params, t1, n_ts)
