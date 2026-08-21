"""
Piecewise-constant warning policies with an Erlang LCT model:

  - Asymmetric thresholds: the warning switches on when the reported Rt rises
    above R_crit but only switches off once it falls below a lower threshold
    R_off < R_crit.
  - Discrete evaluation intervals: once a warning is issued, it is held for at
    least eval_interval days before it is re-evaluated.
  - Lead-time extrapolation: R_est = R_reported + T_lead * d(R_reported)/dt.
"""

from functools import partial

import jax
import jax.numpy as jnp

from models.compartmental import chain_derivative
from models.compartmental_erlang import (
    erlang_initial_state,
    get_erlang_indices,
    seipar_erlang_flow,
)
from models.compartmental_piecewise import (
    PIECEWISE_STATIC,
    published_response,
    solve_piecewise,
)
from models.parameters_erlang import ParamsErlang


def _simulate_piecewise_erlang(params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg):
    idx = get_erlang_indices(params)
    n_flow, n_W, n_B = idx.n_flow, params.n_W, params.n_B

    def _model(t, y, args):
        params, m, floored = args
        W = y[n_flow:n_flow + n_W]
        B = y[n_flow + n_W:]
        B_out = B[-1]
        dFlow, S, prevalence = seipar_erlang_flow(y, params, B_out, idx)
        Rt = params.R_0 * params.rho * B_out * S
        dW = chain_derivative(W, Rt, n_W / params.tau_W)
        response = published_response(W, dW, prevalence, params, m, floored)
        dB = chain_derivative(B, response, n_B / params.tau_B)
        return jnp.concatenate([dFlow, dW, dB])

    return solve_piecewise(
        _model, y0=erlang_initial_state(E0, idx, n_W, n_B), w_out_idx=n_flow + n_W - 1,
        params=params, t1=t1, check_interval=check_interval,
        asymmetric=asymmetric, discrete_eval=discrete_eval, save_per_seg=save_per_seg,
    )

@partial(jax.jit, static_argnames=PIECEWISE_STATIC)
def simulate_SEIPAR_W_piecewise_Erlang(params: ParamsErlang = ParamsErlang.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6,
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 0.1, save_per_seg: int = 1):
    return _simulate_piecewise_erlang(params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg)
