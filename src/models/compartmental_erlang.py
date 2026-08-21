"""
SEIPAR model with wastewater feedback and linear (Erlang) chains for the infected compartments.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from models.compartmental import chain_derivative, solve
from models.parameters import logistic_response_function
from models.parameters_erlang import ParamsErlang


class ErlangIndices(NamedTuple):
    """Start indices of the Erlang chains."""
    iE: int
    iIa: int
    iIp: int
    iIs: int
    iR: int
    n_flow: int


def get_erlang_indices(params: ParamsErlang) -> ErlangIndices:
    """S(1), E(nE), Ia(nA), Ip(nP), Is(nS), R(1)."""
    iE = 1
    iIa = iE + params.nE
    iIp = iIa + params.nA
    iIs = iIp + params.nP
    iR = iIs + params.nS
    return ErlangIndices(iE, iIa, iIp, iIs, iR, iR + 1)

def linear_chain(X, inflow, rate):
    """Linear compartment chain."""
    outflow = jnp.where(jnp.isfinite(rate) & (rate > 0.0), rate, 0.0) * X
    dx = jnp.concatenate([jnp.reshape(inflow, (1,)), outflow[:-1]]) - outflow
    return dx, outflow[-1]

def seipar_erlang_flow(y, params, B_out, idx: ErlangIndices):
    """Flow equations of the Erlang/LCT SEIPAR model."""
    S = y[0]
    E = y[idx.iE:idx.iIa]
    Ia = y[idx.iIa:idx.iIp]
    Ip = y[idx.iIp:idx.iIs]
    Is = y[idx.iIs:idx.iR]

    # weighted force of infection: relative infectiousness * subcompartment weights * subcompartment vector
    infectious = (
        params.phi_a * jnp.dot(params.w_a, Ia) +
        params.phi_p * jnp.dot(params.w_p, Ip) +
        (1.0 - params.epsilon_s) * jnp.dot(params.w_s, Is)
    )
    lambda_S = B_out * params.beta * infectious * S

    dE, E_out = linear_chain(E, lambda_S, params.nE / params.gamma_inv)
    dIa, Ia_out = linear_chain(Ia, params.p * E_out, params.nA / params.mu_a_inv)
    dIp, Ip_out = linear_chain(Ip, (1.0 - params.p) * E_out, params.nP / params.sigma_inv)
    dIs, Is_out = linear_chain(Is, Ip_out, params.nS / params.mu_s_inv)
    dFlow = jnp.concatenate([jnp.atleast_1d(-lambda_S), dE, dIa, dIp, dIs, jnp.atleast_1d(Ia_out + Is_out)])
    return dFlow, S, jnp.sum(Is)

def erlang_initial_state(E0, idx: ErlangIndices, n_W, n_B):
    """Initial state: all exposed density in the first E subcompartment."""
    return jnp.concatenate([
        jnp.stack([1.0 - E0, E0]), jnp.zeros(idx.n_flow - 2),
        jnp.zeros(n_W), jnp.ones(n_B),
    ])


@partial(jax.jit, static_argnames=['t1', 'n_ts'])
def simulate_SEIPAR_W_Erlang(params: ParamsErlang = ParamsErlang.for_SEIPAR(), t1: float = 100.0,
        E0: float = 1e-6, n_ts=None):
    """SEIPAR model with wastewater feedback and linear chains for infected compartments."""
    idx = get_erlang_indices(params)
    n_W, n_B = params.n_W, params.n_B
    iW = idx.n_flow

    def _SEIPAR_W(t, y, params):
        W = y[iW:iW + n_W]
        B = y[iW + n_W:]
        B_out = B[-1]
        dFlow, S, prevalence = seipar_erlang_flow(y, params, B_out, idx)

        # reporting and behavioural delay chains
        Rt = params.R_0 * params.rho * B_out * S
        dW = chain_derivative(W, Rt, n_W / params.tau_W)
        reported = logistic_response_function(W[-1], params, prevalence)
        dB = chain_derivative(B, reported, n_B / params.tau_B)
        return jnp.concatenate([dFlow, dW, dB])

    return solve(_SEIPAR_W, erlang_initial_state(E0, idx, n_W, n_B), params, t1, n_ts)
