"""
Piecewise-constant warning policies with an Erlang LCT model.
The warning decision is updated only at discrete check points (default: 0.1 days).

  - Asymmetric thresholds: the warning switches on when the reported Rt rises
    above R_crit but only switches off once it falls below a lower threshold
    R_off < R_crit.
  - Discrete evaluation intervals: once a warning is issued, it is held for at 
    least eval_interval days before it is re-evaluated.
  - Lead-time extrapolation: use a linear forecast as the reported Rt:
    R_est = R_reported + T_lead * d(R_reported)/dt.
"""

from functools import partial

import jax
import jax.numpy as jnp

from models.compartmental import chain_derivative, linear_chain
from models.compartmental_piecewise import (
    _PIECEWISE_STATIC,
    _published_response,
    _solve_piecewise,
)
from models.parameters_erlang import ParamsErlang


def _SEIPAR(y, params, B_out, idx):
    nE, nA, nP, nS = params.nE, params.nA, params.nP, params.nS
    iIa, iIp, iIs = idx
    S = y[0]
    E = y[1:iIa]
    Ia = y[iIa:iIp]
    Ip = y[iIp:iIs]
    Is = y[iIs:iIs + nS]

    infectious = (
        params.phi_a * jnp.dot(params.w_a, Ia) +
        params.phi_p * jnp.dot(params.w_p, Ip) +
        (1.0 - params.epsilon_s) * jnp.dot(params.w_s, Is)
    )
    lambda_S = B_out * params.beta * infectious * S

    dE, E_out = linear_chain(E, lambda_S, nE / params.gamma_inv)
    dIa, Ia_out = linear_chain(Ia, params.p * E_out, nA / params.mu_a_inv)
    dIp, Ip_out = linear_chain(Ip, (1.0 - params.p) * E_out, nP / params.sigma_inv)
    dIs, Is_out = linear_chain(Is, Ip_out, nS / params.mu_s_inv)
    dFlow = jnp.concatenate([jnp.atleast_1d(-lambda_S), dE, dIa, dIp, dIs, jnp.atleast_1d(Ia_out + Is_out)])
    return dFlow, S, jnp.sum(Is)

def _SEIR(y, params, B_out, idx):
    nE, nS = params.nE, params.nS
    iIs, = idx
    S = y[0]
    E = y[1:iIs]
    Is = y[iIs:iIs + nS]
    infectious = (1.0 - params.epsilon_s) * jnp.dot(params.w_s, Is)
    lambda_S = B_out * params.beta * infectious * S
    dE, E_out = linear_chain(E, lambda_S, nE / params.gamma_inv)
    dIs, Is_out = linear_chain(Is, E_out, nS / params.mu_s_inv)
    dFlow = jnp.concatenate([jnp.atleast_1d(-lambda_S), dE, dIs, jnp.atleast_1d(Is_out)])
    return dFlow, S, jnp.sum(Is)


def _get_indices(model_name, params):
    if model_name == "SEIPAR":
        iIa = 1 + params.nE
        iIp = iIa + params.nA
        iIs = iIp + params.nP
        idx = (iIa, iIp, iIs)
    else: # SEIR
        iIs = 1 + params.nE
        idx = (iIs,)
    return idx, iIs + params.nS + 1


def _simulate_piecewise_erlang(model_name, params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg):
    flow_fn = {"SEIPAR": _SEIPAR, "SEIR": _SEIR}[model_name]
    idx, n_flow = _get_indices(model_name, params)
    n_W, n_B = params.n_W, params.n_B

    def _model(t, y, args):
        params, m, floored = args
        W = y[n_flow:n_flow + n_W]
        B = y[n_flow + n_W:]
        B_out = B[-1]
        dFlow, S, prevalence = flow_fn(y, params, B_out, idx)
        Rt = params.R_0 * params.rho * B_out * S
        dW = chain_derivative(W, Rt, n_W / params.tau_W)
        response = _published_response(W, dW, prevalence, params, m, floored)
        dB = chain_derivative(B, response, n_B / params.tau_B)
        return jnp.concatenate([dFlow, dW, dB])

    y0 = jnp.concatenate([jnp.stack([1.0 - E0, E0]), jnp.zeros(n_flow - 2), jnp.zeros(n_W), jnp.ones(n_B)])
    return _solve_piecewise(_model, y0=y0, w_out_idx=n_flow + n_W - 1, params=params, t1=t1, 
        check_interval=check_interval, asymmetric=asymmetric, discrete_eval=discrete_eval, save_per_seg=save_per_seg)


@partial(jax.jit, static_argnames=_PIECEWISE_STATIC)
def simulate_SEIPAR_W_piecewise_Erlang(params: ParamsErlang = ParamsErlang.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6,
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 1.0, save_per_seg: int = 1):
    return _simulate_piecewise_erlang("SEIPAR", params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg)

@partial(jax.jit, static_argnames=_PIECEWISE_STATIC)
def simulate_SEIR_W_piecewise_Erlang(params: ParamsErlang = ParamsErlang.for_SEIR(), t1: float = 100.0, E0: float = 1e-6,
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 1.0, save_per_seg: int = 1):
    return _simulate_piecewise_erlang("SEIR", params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg)
