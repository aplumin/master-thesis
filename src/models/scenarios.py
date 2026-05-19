"""
Functions for running models.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from models.parameters import Params, logistic_response_function


@partial(jax.jit, static_argnames=['model', 't1'])
def compute_R_grid(model: Callable, base_params: Params, eps_ww: float, eps_ss: float, t1: float = 100.0, E0: float = 1e-6):
    """Compute a 2D grid of Rt values with wastewater warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        tt, yy = model(params=params, t1=t1, E0=E0)
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
        _, yy =  model(params=params, t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)
    return I_tot_grid / I_tot(0.0, 0.0)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Rt(model: Callable, base_params: Params, p: float, phi: float, t1: float = 50.0, E0: float = 1e-6):
    """
    Compute a 2D grid of the reproductive number after interventions.
    Asymptomatic proportion p on the x axis and relative infectiousness phi on the y axis.
    """
    def final_R(p, phi):
        params = base_params.update(p=p, phi=phi)
        _, yy = model(params=params, t1=t1, E0=E0)
        Is_final = yy[-1, -(params.n_W + params.n_B + 2)]
        # TODO: this assumes n_B > 0
        return params.R_0 * params.rho * logistic_response_function(yy[-1,-1], params, Is_final) * yy[-1,0]
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(p, phi)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Itot(model: Callable, base_params: Params, p: float, phi: float, t1: float = 600.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline.
    Asymptomatic proportion p on the x axis and relative infectiousness phi on the y axis.
    """
    def I_tot(p, phi):
        _, yy = model(params=base_params.update(p=p, phi=phi), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(p, phi)
    return I_tot_grid # return absolute fraction infected

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid_delayed_ww(model: Callable, base_params: Params, taus, I_crit_list, t1: float = 100.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to baseline across different
    behavioural delays and infection intervention thresholds.
    """
    def I_tot(tau_B, I_crit):
        _, yy = model(params=base_params.update(tau_B=tau_B, I_crit=I_crit), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(taus, I_crit_list)
    _, yy_base = model(params=base_params.update(epsilon_w=0.0), t1=t1, E0=E0)
    return I_tot_grid / (yy_base[0,0] - yy_base[-1,0])

def outcome_metrics(tt, yy, params, t1, delta_dep=0.05):
    N = tt.shape[0]
    dt = (tt[-1] - tt[0]) / jnp.maximum(N - 1, 1)
    S = yy[:,0]
    Is = yy[:, -(params.n_W + params.n_B + 2)]
    rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]

    ### final Rt ###
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
    F = jnp.fft.fft(jnp.where(in_window, rt_true - mean_Rt, 0.0), n=2*N)
    acorr = jnp.real(jnp.fft.ifft(F * jnp.conj(F)))[:N]
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

    # other metrics
    time_to_below = jnp.where(jnp.any(rt_true < 1.0), tt[jnp.argmax(rt_true < 1.0)], t1)
    Itot = yy[0,0] - yy[-1,0]
    peak_Is = jnp.max(Is)
    extinction_time = tt[-1]

    # oscillation metrics
    rt_reported = yy[:, -(params.n_B + 1)]
    first_100th = rt_true[:max(rt_true.shape[0] // 100, 1)]
    amplitude = jnp.max(first_100th) - jnp.min(first_100th)
    above = (rt_reported >= params.R_crit).astype(jnp.int32)
    total_time_above = above.mean() * t1
    num_crossings = jnp.sum(jnp.diff(above) > 0)

    return Rt_final, time_to_below, Itot, peak_Is, extinction_time, amplitude, total_time_above, num_crossings

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_metrics(model, base_params, eps_ww, eps_ss, t1, E0, delta_dep=0.05):
    def wrap_metrics(w, s):
            params = base_params.update(epsilon_w=w, epsilon_s=s)
            tt, yy = model(params=params, t1=t1, E0=E0)
            Rt_final, time_to_below, Itot, peak_Is, _, _, _, _ = outcome_metrics(tt, yy, params, t1, delta_dep)
            return Rt_final, time_to_below, Itot, peak_Is
    return jax.vmap(jax.vmap(wrap_metrics, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_delay_metrics_grid(model, base_params, taus_W, taus_B, t1=10000.0, E0=1e-6, delta_dep=0.05):
    def wrap_delay_metrics(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        tt, yy = model(params=params, t1=t1) 
        Rt_final, time_to_below, Itot, peak_Is, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep)
        return Rt_final, time_to_below, Itot, peak_Is, amplitude, total_time_above, num_crossings
    return jax.vmap(jax.vmap(wrap_delay_metrics, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)
