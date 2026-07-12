"""
SEIPAR model with wastewater feedback and linear chains for infected compartments.
"""

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
from functools import partial

from models.parameters_erlang import ParamsErlang
from models.parameters import logistic_response_function


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
    dx = jnp.concatenate([jnp.array([inflow]), outflow[:-1]]) - outflow
    return dx, outflow[-1]

@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'nE', 'nP', 'nS', 'nA'])
def simulate_SEIPAR_W_Erlang(
    params: ParamsErlang = ParamsErlang.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6,
    n_W: int = 3, n_B: int = 1, nE: int = 6, nP: int = 3, nS: int = 6, nA: int = 4
):
    """SEIPAR model with wastewater feedback and linear chains for infected compartments."""
    # compartment indices
    iIa = 1 + nE
    iIp = iIa + nA
    iIs = iIp + nP
    iR = iIs + nS
    iW = iR + 1
    iB = iW + n_W

    def _SEIPAR_W(t, y, params):
        # unpack compartments
        S = y[0]
        E = y[1:1 + nE]
        Ia = y[iIa:iIa + nA]
        Ip = y[iIp:iIp + nP]
        Is = y[iIs:iIs + nS]
        W = y[iW:iW + n_W]
        W_out = W[-1]
        B = y[iB:iB + n_B]
        B_out = B[-1]

        # weighted force of infection
        infectious = (
            params.phi * jnp.dot(params.w_a, Ia) + 
            jnp.dot(params.w_p, Ip) + 
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
        dFlow = jnp.array([dS, dE, dIa, dIp, dIs, dR])

        # reporting delay compartments W
        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        # behavioural delay compartments B
        behavioural_delay_rate = n_B / params.tau_B
        Rt_reported = logistic_response_function(W_out, params, jnp.sum(Is))
        B_in = jnp.concatenate([jnp.array([Rt_reported]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    y0 = jnp.concatenate([
        jnp.array([1.0 - E0]), # S
        jnp.array([E0]), jnp.zeros(nE - 1), # E
        jnp.zeros(nA), jnp.zeros(nP), jnp.zeros(nS), # I
        jnp.array([0.0]), # R
        jnp.zeros(n_W), jnp.ones(n_B), # delays
    ])

    solution = diffeqsolve(
        terms=ODETerm(_SEIPAR_W),
        solver=Tsit5(),
        t0=0.0, t1=t1, dt0=0.1,
        y0=y0,
        args=params,
        saveat=SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), max_steps=50_000,
    )
    return solution.ts, solution.ys
