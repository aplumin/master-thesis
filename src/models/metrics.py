"""
Functions for running models.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from functools import partial
from typing import Callable

from models.parameters import Params


def calculate_averaged_Rt(params: Params, tt, t1, dt, N_t, S, Is, rt_true, delta_dep):
    # t_0 = t_Icrit + tau_W+tau_B + 2*sigma
    crosses = Is >= params.I_crit
    t_I_crit = jnp.where(jnp.any(crosses), tt[jnp.argmax(crosses)],t1)
    sd = jnp.sqrt(params.tau_W**2/params.n_W + params.tau_B**2/params.n_B)
    t_0 = t_I_crit + params.tau_W+params.tau_B + 2.0*sd
    # t_1 = first time after t_0 S drops below (1 - delta_dep) * S_0
    depleted = (tt > t_0) & (S < (1.0 - delta_dep) * S[0])
    t_1 = jnp.where(jnp.any(depleted), tt[jnp.argmax(depleted)], tt[-1])
    t_1 = jnp.where(t_1 > 10.0*t_0, 10.0*t_0, t_1) # clip at 10*t_0
    # plain mean over [t_0,t_1]
    in_window = (tt >= t_0) & (tt <= t_1) 
    n_in = jnp.sum(in_window)
    mean_Rt = jnp.where(n_in > 0, jnp.sum(rt_true * in_window) / jnp.maximum(n_in, 1), rt_true[-1])
    # normalised autocorrelation of centred R_t
    F = jnp.fft.fft(jnp.where(in_window, rt_true - mean_Rt, 0.0), n=2*N_t)
    acorr = jnp.real(jnp.fft.ifft(F * jnp.conj(F)))[:N_t]
    acorr = acorr / jnp.maximum(acorr[0],1e-12)
    # T_osc: first local maximum
    is_local_max = jnp.concatenate([jnp.array([False]), (acorr[1:-1] > acorr[:-2]) & (acorr[1:-1] > acorr[2:]), jnp.array([False])])
    has_period = jnp.any(is_local_max)
    T_osc = jnp.argmax(is_local_max) * dt
    # largest m with t_0 + m * T_osc <= t_1
    m = jnp.where(has_period, jnp.floor((t_1-t_0) / jnp.maximum(T_osc,1e-9)).astype(jnp.int32), jnp.int32(0))
    # period-aligned mean
    window_floor = (tt >= t_0) & (tt <= t_0 + m * T_osc)
    mean_floor = jnp.sum(rt_true * window_floor) / jnp.maximum(jnp.sum(window_floor), 1)
    Rt_final = jnp.where(has_period & (m >= 1), mean_floor, mean_Rt)
    return Rt_final

def trajectory_indices(n_W, n_B):
    return {"S": 0, "Is": -(n_W + n_B + 2), "R": -(n_W + n_B + 1), "W_out": -(n_B + 1), "B_out": -1}

def outcome_metrics(tt, yy, params, t1, delta_dep=0.05, population_size=1, warning_state=None):
    """Compute outcome metrics from model trajectories."""
    N_t = tt.shape[0]
    dt = (tt[-1] - tt[0]) / jnp.maximum(N_t - 1, 1)
    idx = trajectory_indices(n_W=params.n_W, n_B=params.n_B)
    S = yy[:, idx["S"]] / population_size
    Is = yy[:, idx["Is"]] / population_size
    rt_true = params.R_0 * params.rho * yy[:, idx["B_out"]] * S

    # final Rt
    Rt_final = calculate_averaged_Rt(params, tt, t1, dt, N_t, S, Is, rt_true, delta_dep)
    time_to_below = jnp.where(jnp.any(rt_true < 1.0), tt[jnp.argmax(rt_true < 1.0)], t1)
    Itot = S[0] - S[-1]
    peak_Is = jnp.max(Is)
    extinction_time = tt[-1]
    first_100th = rt_true[:max(rt_true.shape[0] // 100, 1)]
    amplitude = jnp.max(first_100th) - jnp.min(first_100th)

    # warning duration / switch count
    if warning_state is not None:
        above = (warning_state >= 0.5).astype(jnp.int32)
    else:
        rt_reported = yy[:, -(params.n_B + 1)]
        above = (rt_reported >= params.R_crit).astype(jnp.int32)
    total_time_above = above.mean() * t1
    num_crossings = jnp.sum(jnp.diff(above) > 0)

    return Rt_final, time_to_below, Itot, peak_Is, extinction_time, amplitude, total_time_above, num_crossings


@partial(jax.jit, static_argnames=['model', 't1'])
def compute_R_grid(model: Callable, base_params: Params, eps_ww: float, eps_ss: float, t1: float = 100.0, E0: float = 1e-6):
    """Compute a 2D grid of Rt values with wastewater warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0)
        Rt,_,_,_,_,_,_,_ = outcome_metrics(tt, yy, params, t1, delta_dep=0.05)
        return Rt #params.R_0 * params.rho * yy[-1, -1] * yy[-1, 0]
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid(model: Callable, base_params: Params, eps_ww, eps_ss, t1: float = 100.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline. 
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    def I_tot(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        _, yy, *_ =  model(params=params, t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)
    return I_tot_grid / I_tot(0.0, 0.0)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Rt(model: Callable, base_params: Params, p: float, phi_a: float, t1: float = 50.0, E0: float = 1e-6):
    """
    Compute a 2D grid of the reproductive number after interventions.
    Asymptomatic proportion p on the x axis and relative infectiousness phi_a on the y axis.
    """
    def final_R(p, phi_a):
        params = base_params.update(p=p, phi_a=phi_a)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0)
        Rt,_,_,_,_,_,_,_ = outcome_metrics(tt, yy, params, t1, delta_dep=0.05)
        return Rt
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(p, phi_a)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Itot(model: Callable, base_params: Params, p: float, phi_a: float, t1: float = 600.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline.
    Asymptomatic proportion p on the x axis and relative infectiousness phi_a on the y axis.
    """
    def I_tot(p, phi_a):
        _, yy, *_ = model(params=base_params.update(p=p, phi_a=phi_a), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(p, phi_a)
    return I_tot_grid # return absolute fraction infected

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid_delayed_ww(model: Callable, base_params: Params, taus, I_crit_list, t1: float = 100.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to baseline across different
    behavioural delays and infection intervention thresholds.
    """
    def I_tot(tau_B, I_crit):
        _, yy, *_ = model(params=base_params.update(tau_B=tau_B, I_crit=I_crit), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]

    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(taus, I_crit_list)
    _, yy_base, *_ = model(params=base_params.update(epsilon_w=0.0), t1=t1, E0=E0)
    return I_tot_grid / (yy_base[0,0] - yy_base[-1,0])

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_metrics(model, base_params, eps_ww, eps_ss, t1, E0, delta_dep=0.05):
    def wrap_metrics(w, s):
            params = base_params.update(epsilon_w=w, epsilon_s=s)
            tt, yy, *_ = model(params=params, t1=t1, E0=E0)
            Rt_final, time_to_below, Itot, peak_Is, _, _, _, _ = outcome_metrics(tt, yy, params, t1, delta_dep)
            return Rt_final, time_to_below, Itot, peak_Is
    return jax.vmap(jax.vmap(wrap_metrics, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_delay_metrics_grid(model, base_params, taus_W, taus_B, t1=10000.0, E0=1e-6, delta_dep=0.05):
    def wrap_delay_metrics(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0)
        Rt_final, time_to_below, Itot, peak_Is, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep)
        return Rt_final, time_to_below, Itot, peak_Is, amplitude, total_time_above, num_crossings
    return jax.vmap(jax.vmap(wrap_delay_metrics, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_delay_metrics_grid_piecewise(model, base_params, taus_W, taus_B, t1=10000.0, E0=1e-6, delta_dep=0.05):
    def wrap_delay_metrics(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        tt, yy, ms = model(params=params, t1=t1, E0=E0)
        Rt_final, time_to_below, Itot, peak_Is, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep, warning_state=ms)
        return Rt_final, time_to_below, Itot, peak_Is, amplitude, total_time_above, num_crossings
    return jax.vmap(jax.vmap(wrap_delay_metrics, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

@partial(jax.jit, static_argnames=['model', 't1', 'sweep_field'])
def compute_amplitude_duration_piecewise(model, base_params, sweep_values, sweep_field='R_off', t1=10000.0, E0=1e-6, delta_dep=0.05):
    def wrap(value):
        params = base_params.update(**{sweep_field: value})
        tt, yy, ms = model(params=params, t1=t1, E0=E0)
        _, _, _, _, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep, warning_state=ms)
        return amplitude, total_time_above, num_crossings
    return jax.vmap(wrap)(sweep_values)

def R0_decomposition(params: Params) -> dict[str, float]:
    """Decomposition of R0 into asymptomatic, presymptomatic and symptomatic contributions."""
    R_a = float(params.beta * params.p * params.phi_a * params.mu_a_inv)
    R_p = float(params.beta * (1.0 - params.p) * params.phi_p * params.sigma_inv)
    R_s = float(params.beta * (1.0 - params.p) * params.mu_s_inv)
    return {"a": R_a, "p": R_p, "s": R_s}

def transmission_fractions(params: Params) -> dict[str, float]:
    """Fraction of all transmission events from to each type."""
    R = R0_decomposition(params)
    total = R["a"] + R["p"] + R["s"]
    return {k: (R[k] / total if total > 0 else 0.0) for k in ("a", "p", "s")}

def _infection_jacobian(params: Params) -> tuple[np.ndarray, list[str]]:
    """Jacobian linearised around the disease-free equilibrium."""
    beta = float(params.beta)
    gamma = 1.0 / float(params.gamma_inv)
    mu_s = 1.0 / float(params.mu_s_inv)
    p = float(params.p)
    phi_a = float(params.phi_a)
    phi_p = float(params.phi_p)
    has_presymptomatic = float(params.sigma_inv) > 0.0
    has_asymptomatic = float(params.mu_a_inv) > 0.0
    if has_presymptomatic and has_asymptomatic: # SEIPAR
        sigma = 1.0 / float(params.sigma_inv)
        mu_a = 1.0 / float(params.mu_a_inv)
        J = np.array([
            [-gamma,        beta * phi_a, beta * phi_p, beta ],
            [p * gamma,     -mu_a,        0.0,          0.0  ],
            [(1-p) * gamma, 0.0,          -sigma,       0.0  ],
            [0.0,           0.0,          sigma,        -mu_s],
        ])
        labels = ["a", "p", "s"]
    elif has_asymptomatic: # SEIAR
        mu_a = 1.0 / float(params.mu_a_inv)
        J = np.array([
            [-gamma,          beta * phi_a, beta ],
            [p * gamma,       -mu_a,        0.0  ],
            [(1 - p) * gamma, 0.0,          -mu_s],
        ])
        labels = ["a", "s"]
    else: # SEIR
        J = np.array([
            [-gamma, beta ],
            [gamma,  -mu_s],
        ])
        labels = ["s"]
    return J, labels

def growth_rate(params: Params) -> float:
    """Initial exponential growth rate alpha (dominant eigenvalue of the Jacobian)."""
    J, _ = _infection_jacobian(params)
    return float(np.linalg.eig(J)[0].real.max())

def infectious_fractions(params: Params) -> dict[str, float]:
    """Fraction of infectious individuals of each type during the initial exponential growth phase."""
    J, labels = _infection_jacobian(params)
    w, V = np.linalg.eig(J)
    v = np.abs(np.real(V[:, int(np.argmax(w.real))]))
    infectious = v[1:]
    total = infectious.sum()
    fractions = infectious / total if total > 0 else infectious
    f = {"a": 0.0, "p": 0.0, "s": 0.0}
    f.update({lab: float(f) for lab, f in zip(labels, fractions)})
    return f

def calculate_mt_branching_q(ps, ew, es):
    def extinction_prob(q):
        asyx = ps.phi_a * ps.beta * ps.mu_a_inv * (1-ew/2)
        presyx = ps.phi_p * ps.beta * ps.sigma_inv * (1-ew/2)
        syx = ps.beta * ps.mu_s_inv * (1-es) * (1-ew/2)
        return ps.p / (1 + asyx * (1-q)) + (1-ps.p) / ((1 + presyx * (1-q)) * (1 + syx * (1-q))) - q
    ext_prob = 1.0
    try: 
        ext_prob = brentq(extinction_prob, 0.0, 1.0-1e-9)
    except ValueError: pass
    return ext_prob

def calculate_mt_branching_q_with_superspreading(k, ps, ew, es):
    def g_r(q,r):
        return 1-(1+(1-q)/r)**(-r)
    def extinction_prob(q):
        asyx = ps.phi_a * ps.beta * ps.mu_a_inv * (1-ew/2)
        presyx = ps.phi_p * ps.beta * ps.sigma_inv * (1-ew/2)
        syx = ps.beta * ps.mu_s_inv * (1-es) * (1-ew/2)
        return ps.p / (1 + asyx * g_r(q,k)) + (1-ps.p) / ((1 + presyx * g_r(q,k)) * (1 + syx * g_r(q,k))) - q
    ext_prob = 1.0
    try: ext_prob = brentq(extinction_prob, 0.0, 1.0 - 1e-9)
    except ValueError: pass
    return ext_prob

def strategy_metrics(tau_W, tau_B, n_W, n_B, model, base_params, t1, asymmetric, discrete_eval, check_interval, T_lead_on=False):
    params = base_params.update(tau_W=tau_W, tau_B=tau_B)
    ts, ys, ms = model(params=params, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval)
    S, B_out = ys[:, 0], ys[:, -1]
    rt_true = params.R_0 * params.rho * B_out * S

    mask = ts >= 2.0 * t1 / 3.0
    amplitude = (jnp.max(jnp.where(mask, rt_true, -jnp.inf)) - jnp.min(jnp.where(mask, rt_true, jnp.inf)))
    if asymmetric or discrete_eval:
        time_above = ms.mean() * t1
    else:
        W_out = ys[:, -(1 + n_B)]
        if T_lead_on:
            R_est = W_out + base_params.T_lead * (n_W / params.tau_W) * (ys[:, -(1 + n_B) - 1] - W_out)
        else:
            R_est = W_out
        time_above = (R_est >= params.R_crit).mean() * t1

    cost = jnp.trapezoid(1.0 - B_out, ts)
    Itot = S[0] - S[-1]
    Is = ys[:, -(n_W + n_B + 2)]
    peak_Is = jnp.max(Is)
    N_t = ts.shape[0]
    dt = (ts[-1] - ts[0]) / jnp.maximum(N_t - 1, 1)
    Rt_final = calculate_averaged_Rt(params, ts, t1, dt, N_t, S, Is, rt_true, 0.05)
    return jnp.stack([Rt_final, Itot, peak_Is, amplitude, time_above, cost])

@partial(jax.jit, static_argnames=['model', 't1', 'asymmetric', 'discrete_eval', 'check_interval', 'n_W', 'n_B', 'T_lead_on'])
def strategy_metric_grid(model, base_params, taus_W, taus_B, t1, asymmetric, discrete_eval, check_interval, n_W=3, n_B=1, T_lead_on=False):
    strategy_metrics_vmap = partial(strategy_metrics, n_W=n_W, n_B=n_B, model=model, base_params=base_params, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval, T_lead_on=T_lead_on)
    return jax.vmap(jax.vmap(strategy_metrics_vmap, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

def strategy_grid(
    model, base_params, k, eps_w, eps_s, strategies,
    t1=300.0, taus_W=np.linspace(1.0, 30.0, 100), taus_B=np.linspace(1.0, 30.0, 100), 
    R_off=0.8, eval_interval=14.0,
):
    base_params = base_params.update(epsilon_s=eps_s, epsilon_w=eps_w, k=k, R_off=R_off, eval_interval=eval_interval)
    taus_W, taus_B = jnp.asarray(taus_W), jnp.asarray(taus_B)
    data = {}
    for s, (asym, disc, tl, ci) in strategies.items():
        bp = base_params.update(T_lead=tl)
        grid = strategy_metric_grid(model, bp, taus_W, taus_B, t1, asymmetric=asym, discrete_eval=disc, check_interval=ci, n_W=int(bp.n_W), n_B=int(bp.n_B), T_lead_on=(tl > 0.0))
        data[s] = np.asarray(grid)
    return data, list(np.asarray(taus_W)), list(np.asarray(taus_B))

def R_boundary(theta, eps_s, eps_w):
    return 1.0 / ((1.0 - eps_w / 2.0) * (1.0 - eps_s * (1.0 - theta)))
