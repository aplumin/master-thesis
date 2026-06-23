import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
from functools import partial

from models.parameters import Params


def _published_response(W, dW, prevalence, params, m, floored):
    R_est = W[-1] + params.T_lead * dW[-1] # lead estimate
    threshold = jnp.where(m > 0.5, params.R_off, params.R_crit) # asymmetric: R_off if warning is on, R_crit if it is off
    signal = jnp.where(floored > 0.5, jnp.maximum(params.R_crit, R_est), R_est) # floor at R_crit if in evaluation interval
    gate_W = 1.0 / (1.0 + jnp.exp(-params.k * (signal - threshold)))
    gate_I = jnp.where(
        params.I_crit > 0.0,
        1.0 / (1.0 + jnp.exp(-params.k_I * (prevalence - params.I_crit))),
        1.0,
    )
    return 1.0 - params.epsilon_w * gate_W * gate_I

def _solve_piecewise(
    vector_field, # (t, y, (params, m, floored))
    y0, w_out_idx, params, t1, check_interval, asymmetric, discrete_eval, save_per_seg
    ):
    n_segments = int(round(t1 / check_interval))
    dt0 = min(0.1, 0.5 * check_interval)
    R_off, R_crit, eval_interval = params.R_off, params.R_crit, params.eval_interval
    def _step(carry, i):
        y_start, m, floored, t = carry
        t0 = i * check_interval
        t_end = t0 + check_interval
        ts_save = jnp.linspace(t0, t_end, save_per_seg + 1)[1:] # drop first to avoid duplicate
        sol = diffeqsolve(
            terms=ODETerm(vector_field),
            solver=Tsit5(),
            t0=t0, t1=t_end, dt0=dt0, y0=y_start,
            args=(params, m, floored),
            saveat=SaveAt(ts=ts_save),
            stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), max_steps=50_000,
        )
        ys_seg = sol.ys
        y_end = ys_seg[-1]
        dy_end = vector_field(t_end, y_end, (params, m, floored))
        R_est = y_end[w_out_idx] + params.T_lead * dy_end[w_out_idx]

        # next warning state
        if asymmetric: # R_off if warning is on, R_crit if it is off
            m_next = jnp.where(m > 0.5, R_est >= R_off, R_est > R_crit).astype(jnp.float32)
        else:
            m_next = m

        # re-evaluate warning if next evaluation date reached, else keep warning state
        if discrete_eval:
            t2 = t + check_interval
            reevaluate = t2 >= eval_interval - 1e-9
            floored_next = jnp.where(reevaluate, (R_est > R_crit).astype(jnp.float32), floored)
            t_next = jnp.where(reevaluate, 0.0, t2)
        else:
            floored_next = floored
            t_next = t

        warning_state = jnp.full((save_per_seg,), jnp.maximum(m, floored))
        return (y_end, m_next, floored_next, t_next), (sol.ts, ys_seg, warning_state)

    init = (y0, jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32))
    _, (ts_c, ys_c, ms_c) = jax.lax.scan(init=init, xs=jnp.arange(n_segments), f=_step)
    ts = jnp.concatenate([jnp.array([0.0]), ts_c.reshape(-1)])
    ys = jnp.concatenate([y0[None, :], ys_c.reshape(-1, y0.shape[0])], axis=0)
    ms = jnp.concatenate([jnp.array([0.0], jnp.float32), ms_c.reshape(-1)])
    return ts, ys, ms


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'asymmetric', 'discrete_eval', 'check_interval', 'save_per_seg'])
def simulate_SEIPAR_W_piecewise(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1, 
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 0.1, save_per_seg: int = 1
    ):
    def _SEIPAR_W(t, y, args):
        params, m, floored = args
        S, E, Ia, Ip, Is, R = y[:6]
        W = y[6:6+n_W]
        B = y[6+n_W:]
        B_out = B[-1]

        lambda_S = B_out * params.beta * (params.phi * Ia + Ip + (1.0 - params.epsilon_s) * Is) * S
        become_infectious = E / params.gamma_inv
        become_symptomatic = Ip / params.sigma_inv
        recover_asyx = Ia / params.mu_a_inv
        recover_syx = Is / params.mu_s_inv

        dS = -lambda_S
        dE = lambda_S - become_infectious
        dIa = params.p * become_infectious - recover_asyx
        dIp = (1.0 - params.p) * become_infectious - become_symptomatic
        dIs = become_symptomatic - recover_syx
        dR = recover_asyx + recover_syx
        dFlow = jnp.array([dS, dE, dIa, dIp, dIs, dR])

        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        behavioural_delay_rate = n_B / params.tau_B
        response = _published_response(W, dW, Is, params, m, floored)
        B_in = jnp.concatenate([jnp.array([response]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    return _solve_piecewise(
        _SEIPAR_W,
        y0=jnp.concatenate([jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0]), jnp.zeros(n_W), jnp.ones(n_B)]),
        w_out_idx=6 + n_W - 1,
        params=params, t1=t1, check_interval=check_interval,
        asymmetric=asymmetric, discrete_eval=discrete_eval, save_per_seg=save_per_seg,
    )


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'asymmetric', 'discrete_eval', 'check_interval', 'save_per_seg'])
def simulate_SEIAR_W_piecewise(params: Params = Params.for_SEIAR(), t1: float = 100.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1, 
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 0.1, save_per_seg: int = 1
    ):
    def _SEIAR_W(t, y, args):
        params, m, floored = args
        S, E, Ia, Is, R = y[:5]
        W = y[5:5+n_W]
        B = y[5+n_W:]
        B_out = B[-1]

        lambda_S = B_out * params.beta * (params.phi * Ia + (1.0 - params.epsilon_s) * Is) * S
        become_infectious = E / params.gamma_inv
        recover_asyx = Ia / params.mu_a_inv
        recover_syx = Is / params.mu_s_inv

        dS = -lambda_S
        dE = lambda_S - become_infectious
        dIa = params.p * become_infectious - recover_asyx
        dIs = (1.0 - params.p) * become_infectious - recover_syx
        dR = recover_asyx + recover_syx
        dFlow = jnp.array([dS, dE, dIa, dIs, dR])

        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        behavioural_delay_rate = n_B / params.tau_B
        response = _published_response(W, dW, Is, params, m, floored)
        B_in = jnp.concatenate([jnp.array([response]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    return _solve_piecewise(
        _SEIAR_W,
        y0=jnp.concatenate([jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0]), jnp.zeros(n_W), jnp.ones(n_B)]),
        w_out_idx=5 + n_W - 1,
        params=params, t1=t1, check_interval=check_interval,
        asymmetric=asymmetric, discrete_eval=discrete_eval, save_per_seg=save_per_seg,
    )


@partial(jax.jit, static_argnames=['t1', 'n_W', 'n_B', 'asymmetric', 'discrete_eval', 'check_interval', 'save_per_seg'])
def simulate_SEIR_W_piecewise(params: Params = Params.for_SEIR(), t1: float = 100.0, E0: float = 1e-6, n_W: int = 3, n_B: int = 1, 
        asymmetric: bool = False, discrete_eval: bool = False, check_interval: float = 0.1, save_per_seg: int = 1
    ):
    def _SEIR_W(t, y, args):
        params, m, floored = args
        S, E, II, R = y[:4]
        W = y[4:4+n_W]
        B = y[4+n_W:]
        B_out = B[-1]

        lambda_S = B_out * params.beta * (1.0 - params.epsilon_s) * II * S
        become_infectious = E / params.gamma_inv
        recover = II / params.mu_s_inv

        dS = -lambda_S
        dE = lambda_S - become_infectious
        dI = become_infectious - recover
        dR = recover
        dFlow = jnp.array([dS, dE, dI, dR])

        reporting_delay_rate = n_W / params.tau_W
        Rt = params.R_0 * params.rho * B_out * S
        W_in = jnp.concatenate([jnp.array([Rt]), W[:-1]])
        dW = reporting_delay_rate * (W_in - W)

        behavioural_delay_rate = n_B / params.tau_B
        response = _published_response(W, dW, II, params, m, floored)
        B_in = jnp.concatenate([jnp.array([response]), B[:-1]])
        dB = behavioural_delay_rate * (B_in - B)

        return jnp.concatenate([dFlow, dW, dB])

    return _solve_piecewise(
        _SEIR_W,
        y0=jnp.concatenate([jnp.array([1.0 - E0, E0, 0.0, 0.0]), jnp.zeros(n_W), jnp.ones(n_B)]),
        w_out_idx=4 + n_W - 1,
        params=params, t1=t1, check_interval=check_interval,
        asymmetric=asymmetric, discrete_eval=discrete_eval, save_per_seg=save_per_seg,
    )
