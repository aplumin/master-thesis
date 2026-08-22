"""
Outcome metrics and analytical approximations from model runs.
"""
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from models.parameters import Params
from models.stability import logistic


def get_crossing(tt, fn, level, rising=True, fallback=None):
    """Get the time when a trajectory crosses a specific level."""
    g = (fn - level) if rising else (level - fn)
    crossed = g >= 0.0
    idx_cross = jnp.clip(jnp.argmax(crossed), 1, tt.shape[0] - 1)
    # gradient values before and after the crossing
    g0 = g[idx_cross - 1]
    g1 = g[idx_cross]
    # fraction between g0 and g1 where crossing happens
    dg = g1 - g0
    frac = jnp.clip(jnp.where(dg != 0.0, -g0 / jnp.where(dg != 0.0, dg, 1.0), 0.0), 0.0, 1.0)
    # return linear interpolation of the crossing time or fallback if no crossing
    t_cross = jnp.where(crossed[0], tt[0], tt[idx_cross - 1] + frac * (tt[idx_cross] - tt[idx_cross - 1]))
    fallback = tt[-1] if fallback is None else fallback # default: last
    return jnp.where(jnp.any(crossed), t_cross, fallback)

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

def n_Is_compartments(params):
    """Number of symptomatic subcompartments."""
    return int(getattr(params, "nS", 1)) # exponential Params don't have nS attribute

def trajectory_indices(n_W, n_B, n_S: int = 1):
    """Return compartment indices dict."""
    R = -(n_W + n_B + 1)
    Is = R - n_S if n_S == 1 else slice(R - n_S, R)
    return {"S": 0, "Is": Is, "R": R, "W_out": -(n_B + 1), "B_out": -1}

def column(yy, index):
    col = yy[:, index]
    return jnp.sum(col, axis=-1) if col.ndim > 1 else col

def rt_amplitude(tt, rt_true, window: str = "initial", t_alive=None):
    """Maximum Rt amplitude over first 1% or final third."""
    if window == "final":
        return _rt_amp_final(tt, rt_true, tt[-1] if t_alive is None else t_alive)
    return _rt_amp_initial(tt, rt_true)

def _rt_amp_initial(tt, rt_true):
    initial = rt_true[:max(rt_true.shape[0] // 100, 1)]
    return jnp.max(initial) - jnp.min(initial)

def _rt_amp_final(tt, rt_true, t_alive):
    final = (tt >= 2.0 * t_alive / 3.0) & (tt <= t_alive)
    lo, hi = jnp.min(rt_true), jnp.max(rt_true)
    amplitude = jnp.max(jnp.where(final, rt_true, lo)) - jnp.min(jnp.where(final, rt_true, hi))
    return jnp.where(jnp.any(final), amplitude, jnp.nan)

def trapz_to(tt, f, t1):
    """Trapezoidal integral of f over [0, t1]. Interpolated at the endpoint."""
    tt = np.asarray(tt, dtype=float)
    f = np.asarray(f, dtype=float)
    if t1 >= tt[-1]:
        return float(np.trapezoid(f, tt))
    i = int(np.searchsorted(tt, t1))
    t_head, f_head = tt[:i], f[:i]
    f_at = np.interp(t1, tt, f)
    return float(np.trapezoid(np.append(f_head, f_at), np.append(t_head, t1)))

def cost_and_time_above(tt, B_out, warn_state, t_end):
    """Warning cost and time warned integrated over [0, t_end]."""
    return trapz_to(tt, 1.0 - B_out, t_end), trapz_to(tt, warn_state, t_end)

def oscillation_period(tt, f, t_a, t_b, T_min=4.0, T_max=200.0, peak_threshold=0.2, detrend_days=1.0):
    """
    Period of the dominant oscillation of f over [t_a, t_b].
    First local maximum of the normalised autocorrelation of the mean-centred signal with parabolic interpolation. 
    """
    dt = tt[1] - tt[0]
    interval = (tt >= t_a) & (tt <= t_b)
    num_points = jnp.maximum(jnp.sum(interval), 1) # min 1 to avoid division by 0
    f_mean = jnp.sum(jnp.where(interval, f, 0.0)) / num_points
    # centre at 0
    x = jnp.where(interval, f - f_mean, 0.0)
    n = x.shape[0]
    
    nfft = 1 << int(np.ceil(np.log2(2*n-1))) # pad length of the FFT window to the next power of 2
    F = jnp.fft.rfft(x, n=nfft) # Fourier transform

    # # Gaussian high-pass filter: 1 - exp(-2pi * freq * smoothing window)^2
    # freq = jnp.fft.rfftfreq(nfft, d=dt) # sample frequency
    # F = F * (1.0 - jnp.exp(-2.0 * (jnp.pi * freq * detrend_days) ** 2))
    
    # autocorrelation: multiply spectrum by its complex conjugate, then inverse FFT
    ac = jnp.fft.irfft(F * jnp.conj(F), n=nfft)[:n]
    # normalise by number of overlapping 
    ac = ac / jnp.maximum(num_points - jnp.arange(n), 1)
    # normalize to 1
    ac = ac / jnp.maximum(ac[0], 1e-30)
    
    # find peaks
    lag = jnp.arange(ac.shape[0]) * dt
    is_maximum = (ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:])
    is_over_peak_threshold = ac[1:-1] > peak_threshold
    is_in_interval = (lag[1:-1] >= T_min) & (lag[1:-1] <= T_max)
    peak = is_maximum & is_over_peak_threshold & is_in_interval
    i = jnp.argmax(peak) + 1 # index of virst valid peak
    
    # parabolic interpolation
    denom = ac[i-1] - 2.0 * ac[i] + ac[i+1]
    offset = jnp.where(denom != 0.0, 0.5 * (ac[i-1] - ac[i+1]) / jnp.where(denom != 0.0, denom, 1.0), 0.0)
    return jnp.where(jnp.any(peak), (i + offset) * dt, jnp.nan)

def calculate_averaged_Rt(params, tt, S, Is, rt_true, delta_dep, t_alive=None, min_window=1.0, max_window=10.0, max_periods=10):
    """Average Rt after interventions take effect but before susceptible depletion."""
    t_I_crit = jnp.where(
        params.I_crit > 0.0,
        get_crossing(tt, Is, params.I_crit, rising=True, fallback=tt[jnp.argmax(Is)]),
        tt[0],
    )
    t_last = tt[-1] if t_alive is None else jnp.where(jnp.isnan(t_alive), tt[-1], t_alive)
    sd = jnp.sqrt(params.tau_W**2 / params.n_W + params.tau_B**2 / params.n_B)
    t_0 = jnp.clip(t_I_crit + params.tau_W + params.tau_B + 2.0 * sd, tt[0], tt[-1])
    # only look for depletion after t_0
    t_depleted = get_crossing(tt, -jnp.where(tt >= t_0, S, S[0]), -(1.0 - delta_dep) * S[0], rising=True, fallback=tt[-1])
    t_1 = jnp.clip(jnp.minimum(t_depleted, max_window * t_0), t_0, tt[-1])
    # average over whole oscillation periods
    T_osc = oscillation_period(tt, rt_true, t_0, t_1, detrend_days=params.tau_W + params.tau_B)
    m = jnp.clip(jnp.floor((t_1 - t_0) / T_osc), 0.0, float(max_periods))
    t_end = jnp.where(jnp.isfinite(T_osc) & (m >= 1.0), t_0 + m * T_osc, t_1)
    Rt = jnp.where((t_1 - t_0) < 1.0, jnp.interp(t_0, tt, rt_true), _window_mean(tt, rt_true, t_0, t_end))
    if t_alive is None:
        return Rt
    return jnp.where((t_last - t_0) < min_window, jnp.nan, Rt)

def outcome_metrics(tt, yy, params, t1, delta_dep=0.05, population_size=1, warning_state=None, amplitude_window="final", n_S=None, t_alive=None):
    """Compute outcome metrics from model trajectories."""
    idx = trajectory_indices(n_W=params.n_W, n_B=params.n_B, n_S=n_Is_compartments(params) if n_S is None else n_S)
    S = column(yy, idx["S"]) / population_size
    Is = column(yy, idx["Is"]) / population_size
    R = column(yy, idx["R"]) / population_size
    rt_true = params.R_0 * params.rho * column(yy, idx["B_out"]) * S
    t_last = tt[-1] if t_alive is None else jnp.where(jnp.isnan(t_alive), tt[-1], t_alive)

    # basic metrics
    Rt_final = calculate_averaged_Rt(params, tt, S, Is, rt_true, delta_dep, t_alive=t_alive)
    time_to_below = get_crossing(tt, -rt_true, -1.0, rising=True, fallback=np.inf)
    time_to_below = jnp.where(time_to_below <= t_last, time_to_below, np.inf)
    Itot = S[0] - S[-1]
    peak_Is = jnp.max(Is)
    amplitude = rt_amplitude(tt, rt_true, amplitude_window, t_alive=t_alive)

    # extinction time
    infected = (1.0 - S - R) * population_size
    threshold = 0.5 if population_size > 1 else 1e-6 
    peak = jnp.argmax(infected)
    extinct = (infected < threshold) & (jnp.arange(infected.shape[0]) > peak)
    extinction_time = jnp.where(jnp.any(extinct), tt[jnp.argmax(extinct)], jnp.nan)

    # warning duration and count
    if warning_state is not None:
        above = (warning_state >= 0.5).astype(jnp.float32)
    else:
        above = (column(yy, idx["W_out"]) >= params.R_crit).astype(jnp.float32)
    total_time_above = _integral_to(tt, above, _cumulative_trapezoid(tt, above), jnp.clip(jnp.minimum(t1, t_last), tt[0], tt[-1]))
    num_crossings = jnp.sum((jnp.diff(above) > 0.0) & (tt[1:] <= t_last))
    return Rt_final, time_to_below, Itot, peak_Is, extinction_time, amplitude, total_time_above, num_crossings

def R0_decomposition(params: Params, include_isolation: bool = False):
    """Decomposition of R0 into asymptomatic, presymptomatic and symptomatic contributions."""
    eps = float(params.epsilon_s) if include_isolation else 0.0
    R_a = float(params.beta * params.p * params.phi_a * params.mu_a_inv)
    R_p = float(params.beta * (1.0 - params.p) * params.phi_p * params.sigma_inv)
    R_s = float(params.beta * (1.0 - params.p) * (1.0 - eps) * params.mu_s_inv)
    return {"a": R_a, "p": R_p, "s": R_s}

def transmission_fractions(params: Params):
    """Fraction of all transmission events of each type."""
    R = R0_decomposition(params)
    total = R["a"] + R["p"] + R["s"]
    return {k: (R[k] / total if total > 0 else 0.0) for k in ("a", "p", "s")}

def _infection_jacobian(params: Params):
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
    else: # SEIR
        J = np.array([
            [-gamma, beta_s],
            [gamma,  -mu_s ],
        ])
        labels = ["s"]
    return J, labels

def growth_rate(params: Params):
    """Initial exponential growth rate alpha (dominant eigenvalue of the Jacobian)."""
    J, _ = _infection_jacobian(params)
    return float(np.linalg.eig(J)[0].real.max())

def growth_rate_erlang(ps):
    """Initial exponential growth rate of the Erlang/LCT SEIPAR model."""
    rE = ps.nE / ps.gamma_inv
    rA = ps.nA / ps.mu_a_inv
    rP = ps.nP / ps.sigma_inv
    rS = ps.nS / ps.mu_s_inv
    n = ps.nE + ps.nA + ps.nP + ps.nS
    iE, iA, iP, iS = 0, ps.nE, ps.nE + ps.nA, ps.nE + ps.nA + ps.nP
    J = np.zeros((n, n))
    J[iE, iA:iA + ps.nA] = ps.beta * float(ps.phi_a) * ps.w_a
    J[iE, iP:iP + ps.nP] = ps.beta * float(ps.phi_p) * ps.w_p
    J[iE, iS:iS + ps.nS] = ps.beta * (1.0 - ps.epsilon_s) * ps.w_s

    def chain(start, k, rate, inflow_from=None, inflow_rate=0.0, share=1.0):
        for i in range(k):
            J[start + i, start + i] -= rate
            if i > 0:
                J[start + i, start + i - 1] += rate
        if inflow_from is not None:
            J[start, inflow_from] += share * inflow_rate
    chain(iE, ps.nE, rE)
    chain(iA, ps.nA, rA, inflow_from=iE + ps.nE - 1, inflow_rate=rE, share=ps.p)
    chain(iP, ps.nP, rP, inflow_from=iE + ps.nE - 1, inflow_rate=rE, share=1.0 - ps.p)
    chain(iS, ps.nS, rS, inflow_from=iP + ps.nP - 1, inflow_rate=rP, share=1.0)
    return float(np.linalg.eigvals(J).real.max())

def infectious_fractions(params: Params):
    """Fraction of infectious individuals of each type during the initial exponential growth phase."""
    J, labels = _infection_jacobian(params)
    w, V = np.linalg.eig(J)
    v = np.abs(np.real(V[:, int(np.argmax(w.real))]))
    infectious = v[1:] # E is not yet infectious
    total = infectious.sum()
    fractions = infectious / total if total > 0 else infectious
    infectious_fractions = {"a": 0.0, "p": 0.0, "s": 0.0}
    infectious_fractions.update({label: float(value) for label, value in zip(labels, fractions, strict=True)})
    return infectious_fractions

def eps_s_boundary(R_0, theta, eps_w, k=None, R_crit=1.0):
    """eps_s = (1 - 1/(R_0 * kappa * eps_w)) / (1 - theta)."""
    kappa = np.asarray(mean_warning_multiplier(eps_w, k=k, R_crit=R_crit), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_s = (1.0 - 1.0 / (R_0 * kappa)) / (1.0 - theta)
    return np.where(np.isfinite(eps_s), eps_s, np.nan)

def mean_warning_multiplier(epsilon_w: float, k: float | None = None, R_crit: float = 1.0):
    """1 - epsilon_w/2."""
    if k is None or R_crit == 1.0:
        return 1.0 - epsilon_w / 2.0
    return 1.0 - epsilon_w * logistic(k * (1.0 - R_crit))

def R_boundary(theta, eps_s, eps_w, k=None, R_crit=1.0):
    """R_0_crit = 1 / (m(eps_w) * (1 - eps_s*(1 - theta)))."""
    return 1.0 / (mean_warning_multiplier(eps_w, k=k, R_crit=R_crit) * (1.0 - eps_s * (1.0 - theta)))

def theta_from_type_R_values(R_0, R_a, R_p):
    """
    Calculate nonsymptomatic transmission fraction from reproductive numbers per type:
        theta = (R_a + R_p) / R_0.
    """
    return np.where(R_0 > 0, (R_a + R_p) / np.where(R_0 > 0, R_0, 1.0), 0.0)

def generation_time(r, p, phi_a, phi_p, gamma_inv, sigma_inv, mu_a_inv, mu_s_inv):
    """
    Calculate the mean generation time for the exponential SEIPAR model:
        1/gamma + [p*phi_a/mu_a^2 + (1-p)*(phi_p/sigma^2 + 1/(sigma*mu_s) + 1/mu_s^2)] / r
    """
    return gamma_inv + np.where(r > 0, (p * phi_a * mu_a_inv**2 + (1.0 - p) * (phi_p * sigma_inv**2 + mu_s_inv**2 + sigma_inv * mu_s_inv)) / np.where(r > 0, r, 1.0), 0.0)

def critical_isolation_efficacy(R_s, R_a, R_p):
    """
    Calculate the isolation efficacy needed for R_t = 1 when eps_w = 0:
        1 - (1 - (R_a + R_p)) / R_s.
    """
    return np.where(R_s > 0, 1.0 - (1.0 - (R_a + R_p)) / np.where(R_s > 0, R_s, 1.0), np.nan)

def critical_warning_efficacy(R_eps):
    """
    Calculate the warning efficacy needed for R_t = 1, and for R_crit=1:
        2 * (1 - 1 / R_eps).
    """
    return 2.0 * (1.0 - 1.0 / np.where(R_eps > 0, R_eps, np.nan))

def pathogen_RRs(ps):
    """Asymptomatic and presymptomatic risk ratios from parameters."""
    return (ps.phi_a * ps.mu_a_inv / ps.mu_s_inv if ps.mu_a_inv > 0 else 0.0,
            ps.phi_p * ps.sigma_inv / ps.mu_s_inv if ps.sigma_inv > 0 else 0.0)


### STOCHASTIC
def calculate_mt_branching_q(ps, ew, es):
    """Extinction probability of the multi-type branching process approximation."""
    warn = mean_warning_multiplier(ew)
    asyx = ps.phi_a * ps.beta * ps.mu_a_inv * warn
    presyx = ps.phi_p * ps.beta * ps.sigma_inv * warn
    syx = ps.beta * ps.mu_s_inv * (1-es) * warn
    def extinction_prob(q):
        return ps.p / (1 + asyx * (1-q)) + (1-ps.p) / ((1 + presyx * (1-q)) * (1 + syx * (1-q))) - q
    try:
        return brentq(extinction_prob, 0.0, 1.0-1e-9)
    except ValueError:
        return 1.0

def calculate_mt_branching_q_with_superspreading(k, ps, ew, es, ss_a=True, ss_p=True, ss_s=True):
    """As calculate_mt_branching_q but with overdispersed transmission."""
    warn = mean_warning_multiplier(ew)
    asyx = ps.phi_a * ps.beta * ps.mu_a_inv * warn
    presyx = ps.phi_p * ps.beta * ps.sigma_inv * warn
    syx = ps.beta * ps.mu_s_inv * (1-es) * warn
    def g_r(q, r):
        return -np.expm1(-r * np.log1p((1.0 - q) / r))
    def gap(q, superspreads):
        return g_r(q, k) if (superspreads and k > 0) else (1.0 - q)
    def extinction_prob(q):
        return (ps.p / (1 + asyx * gap(q, ss_a)) + (1-ps.p) / ((1 + presyx * gap(q, ss_p)) * (1 + syx * gap(q, ss_s))) - q)
    try:
        return brentq(extinction_prob, 0.0, 1.0 - 1e-9)
    except ValueError:
        return 1.0

def establishment_threshold(q, alpha=0.01):
    """Number of infecteds above which extinction probability < alpha."""
    if not (0.0 < q < 1.0):
        return np.inf
    return float(np.ceil(np.log(alpha) / np.log(q)))

def extinction_time_from_counts(tt, yy, infected_cols=(1, 2, 3, 4)):
    """First time at which E = Ia = Ip = Is = 0."""
    alive = np.asarray(yy)[:, list(infected_cols)].sum(axis=1) == 0
    extinct = np.flatnonzero(alive)
    return float(np.asarray(tt)[extinct[0]]) if extinct.size else float("nan")

def dispersion_from_individual(ps, r_ind):
    """
    Transmission dispersion r from individual-level dispersion r_ind:
        r = R_0 / (R_0^2/r_ind - R_0 - p*m_a^2 - (1-p)*(m_p^2 + m_s^2) - p*(1-p)*(m_a - m_p - m_s)^2).
    """
    m_a = ps.phi_a * ps.beta * ps.mu_a_inv
    m_p = ps.phi_p * ps.beta * ps.sigma_inv
    m_s = ps.beta * ps.mu_s_inv
    within_type = ps.p * m_a**2 + (1.0 - ps.p) * (m_p**2 + m_s**2)
    between_type = ps.p * (1.0 - ps.p) * (m_a - m_p - m_s)**2
    return ps.R_0 / ((ps.R_0**2 / r_ind) - ps.R_0 - within_type - between_type)
