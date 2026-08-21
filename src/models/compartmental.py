"""
Deterministic compartmental models with wastewater feedback:
    - SEIPAR_W with presymptomatic and asymptomatic transmission
    - SEIR_W with no presymptomatic or asymptomatic transmission
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from diffrax import RESULTS, ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from models.parameters import Params, logistic_response_function

RTOL = 1e-7
ATOL = 1e-9
MAX_STEPS = 50_000
DT0 = 0.1


### POPULATION FLOW EQUATIONS
def _rate(X, period):
    """X / period, or 0 when the compartment period is 0."""
    is_compartment = period > 0.0
    return jnp.where(is_compartment, X / jnp.where(is_compartment, period, 1.0), 0.0)

def seipar_flow(y, params, B_out):
    """(S, E, Ia, Ip, Is, R)."""
    S, E, Ia, Ip, Is, _ = y[:6]
    lambda_S = B_out * params.beta * (params.phi_a * Ia + params.phi_p * Ip + (1.0 - params.epsilon_s) * Is) * S
    become_infectious = _rate(E, params.gamma_inv)
    become_symptomatic = _rate(Ip, params.sigma_inv)
    recover_asyx = _rate(Ia, params.mu_a_inv)
    recover_syx = _rate(Is, params.mu_s_inv)
    dFlow = jnp.stack([
        -lambda_S,
        lambda_S - become_infectious,
        params.p * become_infectious - recover_asyx,
        (1.0 - params.p) * become_infectious - become_symptomatic,
        become_symptomatic - recover_syx,
        recover_asyx + recover_syx,
    ])
    return dFlow, S, Is

def seir_flow(y, params, B_out):
    """(S, E, I, R)."""
    S, E, II, _ = y[:4]
    lambda_S = B_out * params.beta * (1.0 - params.epsilon_s) * II * S
    become_infectious = _rate(E, params.gamma_inv)
    recover = _rate(II, params.mu_s_inv)
    dFlow = jnp.stack([
        -lambda_S,
        lambda_S - become_infectious,
        become_infectious - recover,
        recover,
    ])
    return dFlow, S, II


FLOW_MODELS = { # name: (model function, number population compartments)
    "SEIPAR": (seipar_flow, 6), "SEIR": (seir_flow, 4)}

def chain_derivative(X, inflow, rate):
    """dX = rate * (X_in - X)."""
    return rate * (jnp.concatenate([jnp.reshape(inflow, (1,)), X[:-1]]) - X)

def _delay_ODEs(S, W, B, prevalence, params):
    """Wastewater reporting and behavioural adaptation delay compartments."""
    Rt = params.R_0 * params.rho * B[-1] * S
    dW = chain_derivative(W, Rt, W.shape[0] / params.tau_W)
    reported = logistic_response_function(W[-1], params, prevalence)
    dB = chain_derivative(B, reported, B.shape[0] / params.tau_B)
    return dW, dB

def initial_state(E0, n_flow, n_W, n_B):
    """Initial state [1-E0, E0, (n_flow - 2) zeros, n_W zeros, n_B ones]."""
    return jnp.concatenate([jnp.stack([1.0 - E0, E0]), jnp.zeros(n_flow - 2), jnp.zeros(n_W), jnp.ones(n_B)])

def get_n_ts(t1, n_ts=None):
    """Number of saved time points."""
    if n_ts is None or math.isnan(n_ts):
        # default 0.1 days
        return int(t1 / DT0) + 1
    return int(n_ts)

def solve(diffeq, y0, params, t1, n_ts=None, max_steps=MAX_STEPS, throw=True):
    """diffrax ODE solve."""
    solution = diffeqsolve(
        terms=ODETerm(diffeq),
        solver=Tsit5(),
        t0=0.0, t1=t1, dt0=DT0,
        y0=y0, args=params,
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, get_n_ts(t1, n_ts))),
        stepsize_controller=PIDController(rtol=RTOL, atol=ATOL),
        max_steps=max_steps, throw=throw,
    )
    ys = jnp.where(solution.result == RESULTS.successful, jnp.maximum(solution.ys, 0.0), jnp.nan)
    return solution.ts, ys


def _simulate(model_name, params, t1, E0, n_ts, max_steps, throw):
    flow_fn, n_flow = FLOW_MODELS[model_name]
    n_W, n_B = params.n_W, params.n_B

    def _rhs(t, y, params):
        W = y[n_flow:n_flow + n_W]
        B = y[n_flow + n_W:]
        dFlow, S, prevalence = flow_fn(y, params, B[-1])
        dW, dB = _delay_ODEs(S, W, B, prevalence, params)
        return jnp.concatenate([dFlow, dW, dB])

    y0 = initial_state(E0, n_flow, n_W, n_B)
    return solve(_rhs, y0, params, t1, n_ts, max_steps=max_steps, throw=throw)


@partial(jax.jit, static_argnames=['t1', 'n_ts', 'max_steps', 'throw'])
def simulate_SEIPAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, n_ts=None, max_steps: int = MAX_STEPS, throw: bool = True):
    """Compartmental model with presymptomatic and asymptomatic transmission."""
    return _simulate("SEIPAR", params, t1, E0, n_ts, max_steps, throw)

@partial(jax.jit, static_argnames=['t1', 'n_ts', 'max_steps', 'throw'])
def simulate_SEIR_W(params: Params = Params.for_SEIR(), t1: float = 100.0, E0: float = 1e-6, n_ts=None, max_steps: int = MAX_STEPS, throw: bool = True):
    """Simplified compartmental model without presymptomatic or asymptomatic transmission."""
    return _simulate("SEIR", params, t1, E0, n_ts, max_steps, throw)
