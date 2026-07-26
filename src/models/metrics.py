"""
Outcome metrics and analytical approximations from model runs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from functools import partial
from typing import Callable

from models.parameters import Params


def _get_crossing(tt, fn, level, rising=True, fallback=None):
    """Get the time when a trajectory crosses a specific level."""
    g = (fn - level) if rising else (level - fn)
    crossed = g >= 0.0
    any_crossed = jnp.any(crossed)
    idx_cross = jnp.clip(jnp.argmax(crossed), 1, tt.shape[0] - 1)
    # gradient values before and after the crossing
    g0, g1 = g[idx_cross - 1], g[idx_cross]
    # fraction between g0 and g1 where crossing happens
    dg = g1 - g0
    frac = jnp.clip(jnp.where(dg != 0.0, -g0 / jnp.where(dg != 0.0, dg, 1.0), 0.0), 0.0, 1.0)
    # return linear interpolation of the crossing time or fallback if no crossing
    t_cross = jnp.where(crossed[0], tt[0], tt[idx_cross - 1] + frac * (tt[idx_cross] - tt[idx_cross - 1]))
    fallback = tt[-1] if fallback is None else fallback
    return jnp.where(any_crossed, t_cross, fallback)

def _cumulative_trapezoid(tt, fn):
    """Cumulative trapezoidal integral of fn(tt)."""
    areas = 0.5 * (fn[1:] + fn[:-1]) * jnp.diff(tt)
    return jnp.concatenate([jnp.zeros((1,), fn.dtype), jnp.cumsum(areas)])

def _integral_to(tt, fn, C, t):
    """Cumulative integral C at time t."""
    i = jnp.clip(jnp.searchsorted(tt, jnp.clip(t, tt[0], tt[-1]), side='right'), 1, tt.shape[0] - 1)
    t0, t1 = tt[i - 1], tt[i]
    f0, f1 = fn[i - 1], fn[i]
    f_of_t = f0 + jnp.where(t1 > t0, (t - t0) / (t1 - t0), 0.0) * (f1 - f0)
    return C[i - 1] + 0.5 * (t - t0) * (f0 + f_of_t)

def _window_mean(tt, f, t_a, t_b):
    """Continuous mean of f over [t_a, t_b]."""
    t_a = jnp.clip(t_a, tt[0], tt[-1])
    t_b = jnp.clip(t_b, t_a, tt[-1])
    width = jnp.maximum(t_b - t_a, 1e-12)
    C = _cumulative_trapezoid(tt, f)
    num = _integral_to(tt, f, C, t_b) - _integral_to(tt, f, C, t_a)
    return num / width

def trajectory_indices(n_W, n_B, n_S: int = 1):
    """Return compartment indices dict."""
    R = -(n_W + n_B + 1)
    Is = R - n_S if n_S == 1 else slice(R - n_S, R)
    return {"S": 0, "Is": Is, "R": R, "W_out": -(n_B + 1), "B_out": -1}

def _column(yy, index):
    col = yy[:, index]
    return jnp.sum(col, axis=-1) if col.ndim > 1 else col

def rt_amplitude(tt, rt_true, window: str = "initial"):
    """Maximum Rt amplitude over first 1% or final third."""
    if window == "initial":
        initial = rt_true[:max(rt_true.shape[0] // 100, 1)]
        return jnp.max(initial) - jnp.min(initial)
    if window == "final":
        final = tt >= 2.0 * tt[-1] / 3.0
        return jnp.max(jnp.where(final, rt_true, -jnp.inf)) - jnp.min(jnp.where(final, rt_true, jnp.inf))
    raise ValueError(window)

def _oscillation_period(tt, f, t_a, t_b, T_min=4.0, T_max=200.0, peak_threshold=0.2):
    """First local maximum of the normalised autocorrelation of f - <f> over [t_a, t_b]."""
    dt = tt[1] - tt[0]
    interval = (tt >= t_a) & (tt <= t_b)
    x = jnp.where(interval, f, 0.0)
    x = x - jnp.sum(x) / jnp.maximum(jnp.sum(interval), 1)
    ac = jnp.correlate(x, x, mode="full")[x.shape[0] - 1:]
    ac = ac / jnp.maximum(ac[0], 1e-30)
    lag = jnp.arange(ac.shape[0]) * dt
    peak = (ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]) & (ac[1:-1] > peak_threshold)
    in_range = (lag[1:-1] >= T_min) & (lag[1:-1] <= T_max)
    max_idx = jnp.argmax(peak & in_range)
    return jnp.where(jnp.any(peak & in_range), lag[1:-1][max_idx], jnp.nan)

def calculate_averaged_Rt(params, tt, S, Is, rt_true, delta_dep, max_window=10.0, max_periods=10):
    """Average Rt after interventions take effect but before susceptible depletion."""
    t_I_crit = _get_crossing(tt, Is, params.I_crit, rising=True, fallback=tt[jnp.argmax(Is)])
    sd = jnp.sqrt(params.tau_W**2 / params.n_W + params.tau_B**2 / params.n_B)
    t_0 = jnp.clip(t_I_crit + params.tau_W + params.tau_B + 2.0 * sd, tt[0], tt[-1])
    # only look for depletion after t_0
    t_depleted = _get_crossing(tt, -jnp.where(tt >= t_0, S, S[0]), -(1.0 - delta_dep) * S[0], rising=True, fallback=tt[-1])
    t_1 = jnp.clip(jnp.minimum(t_depleted, max_window * t_0), t_0, tt[-1])
    # average over whole oscillation periods
    T_osc = _oscillation_period(tt, rt_true, t_0, t_1)
    m = jnp.clip(jnp.floor((t_1 - t_0) / T_osc), 0.0, float(max_periods))
    t_end = jnp.where(jnp.isfinite(T_osc) & (m >= 1.0), t_0 + m * T_osc, t_1)
    return _window_mean(tt, rt_true, t_0, t_end)

def outcome_metrics(tt, yy, params, t1, delta_dep=0.05, population_size=1, warning_state=None, amplitude_window="initial", n_S=1):
    """Compute outcome metrics from model trajectories."""
    idx = trajectory_indices(n_W=params.n_W, n_B=params.n_B, n_S=n_S)
    S = _column(yy, idx["S"]) / population_size
    Is = _column(yy, idx["Is"]) / population_size
    R = _column(yy, idx["R"]) / population_size
    rt_true = params.R_0 * params.rho * _column(yy, idx["B_out"]) * S

    # basic metrics
    Rt_final = calculate_averaged_Rt(params, tt, S, Is, rt_true, delta_dep)
    time_to_below = _get_crossing(tt, -rt_true, -1.0, rising=True, fallback=t1)
    Itot = S[0] - S[-1]
    peak_Is = jnp.max(Is)
    amplitude = rt_amplitude(tt, rt_true, amplitude_window)

    # extinction time
    infected = (1.0 - S - R) * population_size
    threshold = 0.5 if population_size > 1 else 1e-6 
    extinct = infected < threshold
    extinction_time = jnp.where(jnp.any(extinct), tt[jnp.argmax(extinct)], tt[-1])

    # warning duration and count
    if warning_state is not None:
        above = (warning_state >= 0.5).astype(jnp.float32)
    else:
        above = (_column(yy, idx["W_out"]) >= params.R_crit).astype(jnp.float32)
    dt_array = jnp.diff(tt)
    total_time_above = jnp.sum(0.5 * (above[1:] + above[:-1]) * dt_array)
    num_crossings = jnp.sum(jnp.diff(above) > 0.0)

    return Rt_final, time_to_below, Itot, peak_Is, extinction_time, amplitude, total_time_above, num_crossings

@partial(jax.jit, static_argnames=['model', 'n_ts'])
def compute_R_grid(model: Callable, base_params: Params, eps_ww: float, eps_ss: float, t1: float = 100.0, E0: float = 1e-6, n_ts: int = 5000):
    """Compute a 2D grid of Rt values with wastewater warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
        Rt = outcome_metrics(tt, yy, params, t1, delta_dep=0.05)[0]
        return Rt
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)

@partial(jax.jit, static_argnames=['model'])
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

@partial(jax.jit, static_argnames=['model'])
def compute_asymptomatic_grid_Rt(model: Callable, base_params: Params, p: float, phi_a: float, t1: float = 50.0, E0: float = 1e-6):
    """
    Compute a 2D grid of the reproductive number after interventions.
    Asymptomatic proportion p on the x axis and relative infectiousness phi_a on the y axis.
    """
    def final_R(p, phi_a):
        params = base_params.update(p=p, phi_a=phi_a)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0)
        return outcome_metrics(tt, yy, params, t1, delta_dep=0.05)[0]
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(p, phi_a)

@partial(jax.jit, static_argnames=['model'])
def compute_asymptomatic_grid_Itot(model: Callable, base_params: Params, p: float, phi_a: float, t1: float = 600.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline.
    Asymptomatic proportion p on the x axis and relative infectiousness phi_a on the y axis.
    """
    def I_tot(p, phi_a):
        _, yy, *_ = model(params=base_params.update(p=p, phi_a=phi_a), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    return jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(p, phi_a)

@partial(jax.jit, static_argnames=['model'])
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

@partial(jax.jit, static_argnames=['model', 'n_ts'])
def compute_metrics(model, base_params, eps_ww, eps_ss, t1, E0, delta_dep=0.05, n_ts=5000):
    """Grid of (Rt, time_to_below, Itot, peak_Is)."""
    def wrap_metrics(w, s):
            params = base_params.update(epsilon_w=w, epsilon_s=s)
            tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
            Rt_final, time_to_below, Itot, peak_Is, _, _, _, _ = outcome_metrics(tt, yy, params, t1, delta_dep)
            return Rt_final, time_to_below, Itot, peak_Is
    return jax.vmap(jax.vmap(wrap_metrics, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)


def contour_boundary(model, base_params, eps_ww, t1, E0, lo=0.0, hi=0.999, level=1.0, metric='Rt', baseline_Itot=None, n_ts=5000):
    """For each eps_w, find eps_s at which the outcome Rt, relative Itot, or peak equals level."""
    def _fn(eps_w, eps_s):
        params = base_params.update(epsilon_w=float(eps_w), epsilon_s=float(eps_s))
        tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
        Rt, _, Itot, peak = outcome_metrics(tt, yy, params, t1)[:4]
        if metric == 'Rt':
            outcome_value = float(Rt)
        elif metric == 'Itot':
            outcome_value = float(Itot) / float(baseline_Itot)
        elif metric == 'peak_Is': 
            outcome_value = float(peak)
        else:
            raise ValueError(metric)
        return outcome_value - level

    boundary = []
    for eps_w in np.asarray(eps_ww):
        lower_contour = _fn(eps_w, lo)
        upper_contour = _fn(eps_w, hi)
        if not np.isfinite(lower_contour) or not np.isfinite(upper_contour) or lower_contour * upper_contour > 0:
            boundary.append(np.nan)
        else:
            boundary.append(brentq(lambda e_s, e_w=eps_w: _fn(e_w, e_s), lo, hi, xtol=1e-6))
    return np.asarray(eps_ww), np.asarray(boundary)

@partial(jax.jit, static_argnames=['model'])
def compute_delay_metrics_grid(model, base_params, taus_W, taus_B, t1=10000.0, E0=1e-6, delta_dep=0.05):
    """Outcome metrics over a (tau_W, tau_B) grid of reporting and behavioural delays."""
    def wrap_delay_metrics(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        tt, yy, *_ = model(params=params, t1=t1, E0=E0)
        Rt_final, time_to_below, Itot, peak_Is, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep)
        return Rt_final, time_to_below, Itot, peak_Is, amplitude, total_time_above, num_crossings
    return jax.vmap(jax.vmap(wrap_delay_metrics, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_delay_metrics_grid_piecewise(model, base_params, taus_W, taus_B, t1=10000.0, E0=1e-6, delta_dep=0.05):
    """As compute_delay_metrics_grid but for piecewise models."""
    def wrap_delay_metrics(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        tt, yy, ms = model(params=params, t1=t1, E0=E0)
        Rt_final, time_to_below, Itot, peak_Is, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep, warning_state=ms)
        return Rt_final, time_to_below, Itot, peak_Is, amplitude, total_time_above, num_crossings
    return jax.vmap(jax.vmap(wrap_delay_metrics, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

@partial(jax.jit, static_argnames=['model', 't1', 'sweep_field'])
def compute_amplitude_duration_piecewise(model, base_params, sweep_values, sweep_field='R_off', t1=10000.0, E0=1e-6, delta_dep=0.05):
    """Get sustained oscillation amplitude, time above threshold, and number of crossings."""
    def wrap(value):
        params = base_params.update(**{sweep_field: value})
        tt, yy, ms = model(params=params, t1=t1, E0=E0)
        _, _, _, _, _, amplitude, total_time_above, num_crossings = outcome_metrics(tt, yy, params, t1, delta_dep, warning_state=ms)
        return amplitude, total_time_above, num_crossings
    return jax.vmap(wrap)(sweep_values)

def R0_decomposition(params: Params, include_isolation: bool = False) -> dict[str, float]:
    """Decomposition of R0 into asymptomatic, presymptomatic and symptomatic contributions."""
    eps = float(params.epsilon_s) if include_isolation else 0.0
    R_a = float(params.beta * params.p * params.phi_a * params.mu_a_inv)
    R_p = float(params.beta * (1.0 - params.p) * params.phi_p * params.sigma_inv)
    R_s = float(params.beta * (1.0 - params.p) * (1.0 - eps) * params.mu_s_inv)
    return {"a": R_a, "p": R_p, "s": R_s}

def transmission_fractions(params: Params) -> dict[str, float]:
    """Fraction of all transmission events of each type."""
    R = R0_decomposition(params)
    total = R["a"] + R["p"] + R["s"]
    return {k: (R[k] / total if total > 0 else 0.0) for k in ("a", "p", "s")}

def _infection_jacobian(params: Params) -> tuple[np.ndarray, list[str]]:
    """Jacobian of the infection subsystem linearised around the disease-free equilibrium."""
    beta = float(params.beta)
    eps_s = float(params.epsilon_s)
    beta_s = beta * (1.0 - eps_s)
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
            [-gamma,        beta * phi_a, beta * phi_p, beta_s],
            [p * gamma,     -mu_a,        0.0,          0.0   ],
            [(1-p) * gamma, 0.0,          -sigma,       0.0   ],
            [0.0,           0.0,          sigma,        -mu_s ],
        ])
        labels = ["a", "p", "s"]
    elif has_asymptomatic: # SEIAR
        mu_a = 1.0 / float(params.mu_a_inv)
        J = np.array([
            [-gamma,          beta * phi_a, beta_s],
            [p * gamma,       -mu_a,        0.0   ],
            [(1 - p) * gamma, 0.0,          -mu_s ],
        ])
        labels = ["a", "s"]
    else: # SEIR
        J = np.array([
            [-gamma, beta_s],
            [gamma,  -mu_s ],
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
    infectious = v[1:] # E is not yet infectious
    total = infectious.sum()
    fractions = infectious / total if total > 0 else infectious
    infectious_fractions = {"a": 0.0, "p": 0.0, "s": 0.0}
    infectious_fractions.update({label: float(value) for label, value in zip(labels, fractions)})
    return infectious_fractions

def mean_warning_multiplier(epsilon_w: float) -> float:
    """1 - epsilon_w/2."""
    return 1.0 - epsilon_w / 2.0

def calculate_mt_branching_q(ps, ew, es):
    """Extinction probability of the multi-type branching process approximation."""
    warn = mean_warning_multiplier(ew)
    def extinction_prob(q):
        asyx = ps.phi_a * ps.beta * ps.mu_a_inv * warn
        presyx = ps.phi_p * ps.beta * ps.sigma_inv * warn
        syx = ps.beta * ps.mu_s_inv * (1-es) * warn
        return ps.p / (1 + asyx * (1-q)) + (1-ps.p) / ((1 + presyx * (1-q)) * (1 + syx * (1-q))) - q
    try:
        return brentq(extinction_prob, 0.0, 1.0-1e-9)
    except ValueError:
        return 1.0

def calculate_mt_branching_q_with_superspreading(k, ps, ew, es):
    """As calculate_mt_branching_q but with overdispersed transmission."""
    warn = mean_warning_multiplier(ew)
    def g_r(q,r):
        return 1-(1+(1-q)/r)**(-r)
    def extinction_prob(q):
        asyx = ps.phi_a * ps.beta * ps.mu_a_inv * warn
        presyx = ps.phi_p * ps.beta * ps.sigma_inv * warn
        syx = ps.beta * ps.mu_s_inv * (1-es) * warn
        return ps.p / (1 + asyx * g_r(q,k)) + (1-ps.p) / ((1 + presyx * g_r(q,k)) * (1 + syx * g_r(q,k))) - q
    try:
        return brentq(extinction_prob, 0.0, 1.0 - 1e-9)
    except ValueError:
        return 1.0

def strategy_metrics(tau_W, tau_B, model, base_params, t1, asymmetric, discrete_eval, check_interval, T_lead_on=False):
    """
    Summary metrics for one piecewise warning strategy given delays (tau_W, tau_B).
    Returns [Rt_final, Itot, peak_Is, amplitude, time_above, cost].
    """
    params = base_params.update(tau_W=tau_W, tau_B=tau_B)
    n_W, n_B = params.n_W, params.n_B
    ts, ys, ms = model(params=params, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval)
    idx = trajectory_indices(n_W, n_B)
    S, B_out = ys[:, idx["S"]], ys[:, idx["B_out"]]
    rt_true = params.R_0 * params.rho * B_out * S

    amplitude = rt_amplitude(ts, rt_true, window="final")
    if asymmetric or discrete_eval:
        time_above = ms.mean() * t1
    else:
        W_out = ys[:, idx["W_out"]]
        if T_lead_on:
            R_est = W_out + base_params.T_lead * (n_W / params.tau_W) * (ys[:, idx["W_out"] - 1] - W_out)
        else:
            R_est = W_out
        time_above = (R_est >= params.R_crit).mean() * t1

    cost = jnp.trapezoid(1.0 - B_out, ts)
    Itot = S[0] - S[-1]
    Is = ys[:, idx["Is"]]
    peak_Is = jnp.max(Is)
    Rt_final = calculate_averaged_Rt(params, ts, S, Is, rt_true, 0.05)
    return jnp.stack([Rt_final, Itot, peak_Is, amplitude, time_above, cost])

@partial(jax.jit, static_argnames=['model', 't1', 'asymmetric', 'discrete_eval', 'check_interval', 'T_lead_on'])
def strategy_metric_grid(model, base_params, taus_W, taus_B, t1, asymmetric, discrete_eval, check_interval, T_lead_on=False):
    """strategy_metrics over the full (tau_W, tau_B) grid."""
    strategy_metrics_vmap = partial(strategy_metrics, model=model, base_params=base_params, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval, T_lead_on=T_lead_on)
    return jax.vmap(jax.vmap(strategy_metrics_vmap, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

def strategy_grid(
    model, base_params, k, eps_w, eps_s, strategies, t1=300.0, 
    taus_W=None, taus_B=None, R_off=0.8, eval_interval=14.0,
):
    """Delay-grid metrics for multiple warning strategies."""
    taus_W = np.linspace(1.0, 30.0, 100) if taus_W is None else taus_W
    taus_B = np.linspace(1.0, 30.0, 100) if taus_B is None else taus_B
    base_params = base_params.update(epsilon_s=eps_s, epsilon_w=eps_w, k=k, R_off=R_off, eval_interval=eval_interval)
    taus_W, taus_B = jnp.asarray(taus_W), jnp.asarray(taus_B)
    data = {}
    for s, (asym, disc, tl, ci) in strategies.items():
        bp = base_params.update(T_lead=tl)
        grid = strategy_metric_grid(model, bp, taus_W, taus_B, t1, asymmetric=asym, discrete_eval=disc, check_interval=ci, T_lead_on=(tl > 0.0))
        data[s] = np.asarray(grid)
    return data, list(np.asarray(taus_W)), list(np.asarray(taus_B))

def R_boundary(theta, eps_s, eps_w):
    """
    Boundary R_t = 1 as a function of the non-symptomatic transmission fraction theta 
    and the intervention efficacies:
        R_0_crit = 1 / ((1 - epsilon_w/2) * (1 - epsilon_s*(1 - theta))).
    """
    return 1.0 / (mean_warning_multiplier(eps_w) * (1.0 - eps_s * (1.0 - theta)))
