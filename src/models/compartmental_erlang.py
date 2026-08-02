"""
SEIPAR model with wastewater feedback and linear chains for infected compartments.
"""

from functools import partial

import jax
import jax.numpy as jnp

from models.compartmental import _solve, chain_derivative, linear_chain
from models.parameters import logistic_response_function
from models.parameters_erlang import ParamsErlang


@partial(jax.jit, static_argnames=['n_ts'])
def simulate_SEIPAR_W_Erlang(params: ParamsErlang = ParamsErlang.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, n_ts: int = 5000):
    """SEIPAR model with wastewater feedback and linear chains for infected compartments."""
    nE, nP, nS, nA = params.nE, params.nP, params.nS, params.nA
    n_W, n_B = params.n_W, params.n_B

    # compartment indices
    # S(1), E(nE), Ia(nA), Ip(nP), Is(nS), R(1), W(n_W), B(n_B)
    iIa = 1 + nE
    iIp = iIa + nA
    iIs = iIp + nP
    iR = iIs + nS
    iW = iR + 1
    iB = iW + n_W

    def _SEIPAR_W(t, y, params):
        # unpack compartments
        # each infected compartment is a chain of subcompartments
        S = y[0]
        E = y[1:1 + nE]
        Ia = y[iIa:iIa + nA]
        Ip = y[iIp:iIp + nP]
        Is = y[iIs:iIs + nS]
        W = y[iW:iW + n_W]
        B = y[iB:iB + n_B]
        B_out = B[-1]

        # weighted force of infection
        # relative infectiousness * subcompartment weights * subcompartment vector
        infectious = (
            params.phi_a * jnp.dot(params.w_a, Ia) + 
            params.phi_p * jnp.dot(params.w_p, Ip) + 
            (1.0 - params.epsilon_s) * jnp.dot(params.w_s, Is)
        )
        lambda_S = B_out * params.beta * infectious * S

        # flow compartments with Erlang chains for infected
        dE, E_out = linear_chain(E, lambda_S, nE / params.gamma_inv)
        dIa, Ia_out = linear_chain(Ia, params.p * E_out, nA / params.mu_a_inv)
        dIp, Ip_out = linear_chain(Ip, (1.0 - params.p) * E_out, nP / params.sigma_inv)
        dIs, Is_out = linear_chain(Is, Ip_out, nS / params.mu_s_inv)
        dS = -lambda_S
        dR = Ia_out + Is_out
        dFlow = jnp.concatenate([jnp.atleast_1d(dS), dE, dIa, dIp, dIs, jnp.atleast_1d(dR)])

        # reporting and behavioural delay chains
        Rt = params.R_0 * params.rho * B_out * S
        dW = chain_derivative(W, Rt, n_W / params.tau_W)
        reported = logistic_response_function(W[-1], params, jnp.sum(Is))
        dB = chain_derivative(B, reported, n_B / params.tau_B)

        return jnp.concatenate([dFlow, dW, dB])

    y0 = jnp.concatenate([
        jnp.atleast_1d(1.0 - E0), # S
        jnp.atleast_1d(E0), jnp.zeros(nE - 1), # E
        jnp.zeros(nA), jnp.zeros(nP), jnp.zeros(nS), # I
        jnp.zeros(1), # R
        jnp.zeros(n_W), jnp.ones(n_B), # delays
    ])

    return _solve(_SEIPAR_W, y0, params, t1, n_ts)

@partial(jax.jit, static_argnames=['n_ts'])
def simulate_SEIR_W_Erlang(params: ParamsErlang = ParamsErlang.for_SEIR(), t1: float = 100.0, E0: float = 1e-6, n_ts: int = 5000):
    """SEIR model with wastewater feedback and linear chains for infected compartments."""
    nE, nS = params.nE, params.nS
    n_W, n_B = params.n_W, params.n_B

    # compartment indices
    # S(1), E(nE), Is(nS), R(1), W(n_W), B(n_B)
    iIs = 1 + nE
    iR = iIs + nS
    iW = iR + 1
    iB = iW + n_W

    def _SEIR_W(t, y, params):
        # unpack compartments
        # each infected compartment is a chain of subcompartments
        S = y[0]
        E = y[1:iIs]
        Is = y[iIs:iIs + nS]
        W = y[iW:iW + n_W]
        B = y[iB:iB + n_B]
        B_out = B[-1]

        # weighted force of infection
        # relative infectiousness * subcompartment weights * subcompartment vector
        infectious = (1.0 - params.epsilon_s) * jnp.dot(params.w_s, Is)
        lambda_S = B_out * params.beta * infectious * S

        # flow compartments with Erlang chains for infected
        dE, E_out = linear_chain(E, lambda_S, nE / params.gamma_inv)
        dIs, Is_out = linear_chain(Is, E_out, nS / params.mu_s_inv)
        dS = -lambda_S
        dR = Is_out
        dFlow = jnp.concatenate([jnp.atleast_1d(dS), dE, dIs, jnp.atleast_1d(dR)])

        # reporting and behavioural delay chains
        Rt = params.R_0 * params.rho * B_out * S
        dW = chain_derivative(W, Rt, n_W / params.tau_W)
        reported = logistic_response_function(W[-1], params, jnp.sum(Is))
        dB = chain_derivative(B, reported, n_B / params.tau_B)

        return jnp.concatenate([dFlow, dW, dB])

    y0 = jnp.concatenate([
        jnp.atleast_1d(1.0 - E0), # S
        jnp.atleast_1d(E0), jnp.zeros(nE - 1), # E
        jnp.zeros(nS), # I
        jnp.zeros(1), # R
        jnp.zeros(n_W), jnp.ones(n_B), # delays
    ])

    return _solve(_SEIR_W, y0, params, t1, n_ts)
