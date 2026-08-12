"""
Piecewise-constant warning policies.
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
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from models.compartmental import ATOL, MAX_STEPS, RTOL, chain_derivative
from models.parameters import Params


def _published_response(W, dW, prevalence, params, m, floored):
    """Logistic response function for the alternative warning systems.
    m is the current warning state (1 = on) and floored=True means that the warning is kept on.
    """
    R_est = W[-1] + params.T_lead * dW[-1] # lead estimate
    threshold = jnp.where(m > 0.5, params.R_off, params.R_crit) # asymmetric: R_off if warning is on, R_crit if it is off
    signal = jnp.where(floored > 0.5, jnp.maximum(params.R_crit, R_est), R_est) # floor at R_crit if in evaluation interval
    gate_W = 1.0 / (1.0 + jnp.exp(-params.k * (signal - threshold)))
    gate_I = jnp.where(
        params.I_crit > 0.0,
        1.0 / (1.0 + jnp.exp(-params.k_I * jnp.log10(prevalence / (params.I_crit+1e-30)))),
        1.0,
    )
    return 1.0 - params.epsilon_w * gate_W * gate_I


def _SEIPAR(y, params, B_out):
    S, E, Ia, Ip, Is, _ = y[:6]
    lambda_S = B_out * params.beta * (params.phi_a * Ia + params.phi_p * Ip + (1.0 - params.epsilon_s) * Is) * S
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
    return dFlow, S, Is

def _SEIR(y, params, B_out):
    S, E, II, _ = y[:4]
    lambda_S = B_out * params.beta * (1.0 - params.epsilon_s) * II * S
    become_infectious = E / params.gamma_inv
    recover = II / params.mu_s_inv
    dFlow = jnp.stack([
        -lambda_S,
        lambda_S - become_infectious,
        become_infectious - recover,
        recover,
    ])
    return dFlow, S, II

# model name -> ODE function, number of population flow compartments
_MODELS = {"SEIPAR": (_SEIPAR, 6), "SEIR": (_SEIR, 4)}


def _solve_piecewise(
    diffeq, # (t, y, (params, m, floored))
    y0, w_out_idx, params, t1, check_interval, asymmetric, discrete_eval, save_per_seg
    ):
    """Integrate the ODE over [0, t1] with fixed length segments."""
    n_segments = round(t1 / check_interval)
    dt0 = min(0.1, 0.5 * check_interval)
    R_off, R_crit, eval_interval = params.R_off, params.R_crit, params.eval_interval
    def _step(carry, i):
        y_start, m, floored, t = carry
        t0 = i * check_interval
        t_end = t0 + check_interval
        ts_save = jnp.linspace(t0, t_end, save_per_seg + 1)[1:] # drop first to avoid duplicate
        sol = diffeqsolve(
            terms=ODETerm(diffeq),
            solver=Tsit5(),
            t0=t0, t1=t_end, dt0=dt0, y0=y_start,
            args=(params, m, floored), saveat=SaveAt(ts=ts_save),
            stepsize_controller=PIDController(rtol=RTOL, atol=ATOL), max_steps=MAX_STEPS,
        )
        ys_seg = sol.ys
        y_end = ys_seg[-1]
        # forecast the reported Rt at the end of the segment
        dy_end = diffeq(t_end, y_end, (params, m, floored))
        R_est = y_end[w_out_idx] + params.T_lead * dy_end[w_out_idx]

        # Asymmetric: if the warning is on, keep it on until R_est falls below R_off
        # if it is off, turn it on when R_est > R_crit
        if asymmetric: # R_off if warning is on, R_crit if it is off
            m_next = jnp.where(m > 0.5, R_est >= R_off, R_est > R_crit).astype(jnp.float64)
        else:
            m_next = m

        # Discrete intervals: re-evaluate warning if next evaluation date reached, else keep warning state
        if discrete_eval:
            t2 = t + check_interval
            reevaluate = t2 >= eval_interval - 1e-9
            floored_next = jnp.where(reevaluate, (R_est > R_crit).astype(jnp.float64), floored)
            t_next = jnp.where(reevaluate, 0.0, t2)
        else:
            floored_next = floored
            t_next = t

        published = (R_est >= R_crit).astype(jnp.float64)
        state = jnp.maximum(jnp.maximum(m, floored), published if not (asymmetric or discrete_eval) else 0.0)
        return (y_end, m_next, floored_next, t_next), (sol.ts, ys_seg, jnp.full((save_per_seg,), state))

    zero = jnp.asarray(0.0, jnp.float64)
    init = (y0, zero, zero, zero)
    _, (ts_c, ys_c, ms_c) = jax.lax.scan(init=init, xs=jnp.arange(n_segments), f=_step)
    ts = jnp.concatenate([jnp.zeros(1), ts_c.reshape(-1)])
    ys = jnp.concatenate([y0[None, :], ys_c.reshape(-1, y0.shape[0])], axis=0)
    ms = jnp.concatenate([jnp.zeros(1, jnp.float64), ms_c.reshape(-1)])
    return ts, ys, ms


def _simulate_piecewise(model_name, params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg):
    flow_fn, n_flow = _MODELS[model_name]
    n_W, n_B = params.n_W, params.n_B

    def _rhs(t, y, args):
        params, m, floored = args
        W = y[n_flow:n_flow + n_W]
        B = y[n_flow + n_W:]
        B_out = B[-1]

        dFlow, S, prevalence = flow_fn(y, params, B_out)

        Rt = params.R_0 * params.rho * B_out * S
        dW = chain_derivative(W, Rt, n_W / params.tau_W)
        response = _published_response(W, dW, prevalence, params, m, floored)
        dB = chain_derivative(B, response, n_B / params.tau_B)

        return jnp.concatenate([dFlow, dW, dB])

    y0 = jnp.concatenate([
        jnp.stack([1.0 - E0, E0]), jnp.zeros(n_flow - 2),
        jnp.zeros(n_W), jnp.ones(n_B),
    ])
    return _solve_piecewise(
        _rhs, y0=y0, w_out_idx=n_flow + n_W - 1,
        params=params, t1=t1, check_interval=check_interval,
        asymmetric=asymmetric, discrete_eval=discrete_eval, save_per_seg=save_per_seg,
    )


_PIECEWISE_STATIC = ['t1', 'asymmetric', 'discrete_eval', 'check_interval', 'save_per_seg']

@partial(jax.jit, static_argnames=_PIECEWISE_STATIC)
def simulate_SEIPAR_W_piecewise(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6,
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 0.1, save_per_seg: int = 1):
    return _simulate_piecewise("SEIPAR", params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg)

@partial(jax.jit, static_argnames=_PIECEWISE_STATIC)
def simulate_SEIR_W_piecewise(params: Params = Params.for_SEIR(), t1: float = 100.0, E0: float = 1e-6,
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 0.1, save_per_seg: int = 1):
    return _simulate_piecewise("SEIR", params, t1, E0, asymmetric, discrete_eval, check_interval, save_per_seg)
