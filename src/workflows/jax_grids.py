"""
Vectorised jax grids over the deterministic models.
"""

from collections.abc import Callable
from functools import partial

import jax

from models.metrics import outcome_metrics
from models.parameters import Params

_STATIC = ['model', 't1', 'n_ts']


@partial(jax.jit, static_argnames=_STATIC)
def compute_R_grid(model: Callable, base_params: Params, eps_ww, eps_ss, t1: float = 100.0, E0: float = 1e-6, n_ts=None):
    """Grid of Rt with warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
        return outcome_metrics(tt, yy, params, t1, delta_dep=0.05)[0]
    return jax.lax.map(lambda s: jax.vmap(final_R, in_axes=(0, None))(eps_ww, s), eps_ss)

@partial(jax.jit, static_argnames=_STATIC)
def compute_I_tot_grid(model: Callable, base_params: Params, eps_ww, eps_ss, t1: float = 100.0, E0: float = 1e-6, n_ts=None):
    """
    Grid of the proportion infected relative to a no-intervention baseline.
    Warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    def I_tot(w, s):
        _, yy, *_ = model(params=base_params.update(epsilon_w=w, epsilon_s=s), t1=t1, E0=E0, n_ts=n_ts)
        return yy[0, 0] - yy[-1, 0]
    grid = jax.lax.map(lambda s: jax.vmap(I_tot, in_axes=(0, None))(eps_ww, s), eps_ss)
    return grid / I_tot(0.0, 0.0)

@partial(jax.jit, static_argnames=_STATIC)
def compute_asymptomatic_grid_Rt(model: Callable, base_params: Params, p, phi_a, t1: float = 50.0, E0: float = 1e-6, n_ts=None):
    """
    Grid of the reproductive number after interventions.
    Asymptomatic proportion p on the x axis and relative infectiousness phi_a on the y axis.
    """
    def final_R(p_, phi_):
        params = base_params.update(p=p_, phi_a=phi_)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
        return outcome_metrics(tt, yy, params, t1, delta_dep=0.05)[0]
    return jax.lax.map(lambda phi_: jax.vmap(final_R, in_axes=(0, None))(p, phi_), phi_a)

@partial(jax.jit, static_argnames=_STATIC)
def compute_asymptomatic_grid_Itot(model: Callable, base_params: Params, p, phi_a, t1: float = 600.0, E0: float = 1e-6, n_ts=None):
    """
    Grid of the proportion infected.
    Asymptomatic proportion p on the x axis and relative infectiousness phi_a on the y axis.
    """
    def I_tot(p_, phi_):
        _, yy, *_ = model(params=base_params.update(p=p_, phi_a=phi_), t1=t1, E0=E0, n_ts=n_ts)
        return yy[0, 0] - yy[-1, 0]
    return jax.lax.map(lambda phi_: jax.vmap(I_tot, in_axes=(0, None))(p, phi_), phi_a)

@partial(jax.jit, static_argnames=_STATIC)
def compute_I_tot_grid_delayed_ww(model: Callable, base_params: Params, taus, I_crit_list, t1: float = 100.0, E0: float = 1e-6, n_ts=None):
    """Grid of the proportion infected relative to baseline across behavioural delays (x) and infection intervention thresholds (y)."""
    def I_tot(tau_B, I_crit):
        _, yy, *_ = model(params=base_params.update(tau_B=tau_B, I_crit=I_crit), t1=t1, E0=E0, n_ts=n_ts)
        return yy[0, 0] - yy[-1, 0]
    _, yy_base, *_ = model(params=base_params.update(epsilon_w=0.0), t1=t1, E0=E0, n_ts=n_ts)
    grid = jax.lax.map(lambda I_crit: jax.vmap(I_tot, in_axes=(0, None))(taus, I_crit), I_crit_list)
    return grid / (yy_base[0, 0] - yy_base[-1, 0])

@partial(jax.jit, static_argnames=_STATIC)
def compute_metrics(model, base_params, eps_ww, eps_ss, t1, E0, delta_dep=0.05, n_ts=None):
    """Grid of (Rt, time_to_below, Itot, peak_Is)."""
    def wrap_metrics(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
        Rt_final, time_to_below, Itot, peak_Is, *_ = outcome_metrics(tt, yy, params, t1, delta_dep)
        return Rt_final, time_to_below, Itot, peak_Is
    return jax.lax.map(lambda s: jax.vmap(wrap_metrics, in_axes=(0, None))(eps_ww, s), eps_ss)

@partial(jax.jit, static_argnames=['model', 't1', 'n_ts', 'n_S'])
def compute_delay_metrics_grid(model, base_params, taus_W, taus_B, t1=10000.0, E0=1e-6, delta_dep=0.05, n_ts=None, n_S=None):
    """Outcome metrics over a (tau_B, tau_W) grid of behavioural (x) and reporting (y) delays."""
    def wrap_delay_metrics(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
        Rt_final, time_to_below, Itot, peak_Is, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep, n_S=n_S)
        return Rt_final, time_to_below, Itot, peak_Is, amplitude, total_time_above, num_crossings, tau_W + tau_B
    return jax.lax.map(lambda tau_W: jax.vmap(wrap_delay_metrics, in_axes=(None, 0))(tau_W, taus_B), taus_W)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_rt_grid(model, base_params, taus_W, taus_B, t1=300.0):
    """True Rt in (tau_W, tau_B)."""
    def _rt(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        _, yy = model(params=params, t1=t1)
        return params.R_0 * params.rho * yy[:,-1] * yy[:,0]
    return jax.vmap(jax.vmap(_rt, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)
