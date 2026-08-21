"""
Stochastic Gillespie versions of the compartmental models.
"""

import numpy as np
from numba import njit

from models.compartmental import get_n_ts
from models.parameters import Params


@njit(cache=True)
def _advance_chain(X, inflow, xi, dt, coeffs, scratch):
    """
    Integrate the linear delay chain over timestep dt:
        X_i(t+dt) = inflow + sum_{j<=i} c_{i-j} * (X_j(t) - inflow),
    with Poisson coefficients c_m = exp(-xi*dt) * (xi*dt)^m / m!.
    """
    n = X.shape[0]
    for i in range(n):
        scratch[i] = X[i]
    xdt = xi * dt
    coeffs[0] = np.exp(-xdt)
    for m in range(1, n):
        coeffs[m] = coeffs[m - 1] * xdt / m
    for i in range(n):
        v = inflow
        for j in range(i + 1):
            v += coeffs[i - j] * (scratch[j] - inflow)
        X[i] = v

@njit(cache=True)
def _exponential_dt(a0):
    """Exponential waiting time with a rate equal to the total propensity a0."""
    return -np.log(1.0 - np.random.random()) / a0

@njit(cache=True)
def _select_event(a, a0, num_reactions):
    """Select event i with probability a[i]/a0."""
    r2 = np.random.random() * a0
    cumsum = 0.0
    for i in range(num_reactions):
        cumsum += a[i]
        if r2 < cumsum:
            return i
    return num_reactions - 1

@njit(cache=True)
def _logistic(x):
    if x > 80.0:
        return 1.0
    if x < -80.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))

@njit(cache=True)
def _response(W_out, Is_frac, epsilon_w, k, R_crit, k_I, I_crit):
    """Logistic response function."""
    if I_crit > 0.0:
        gate_I = _logistic(k_I * np.log10(max(Is_frac, 1e-300) / I_crit))
    else:
        gate_I = 1.0
    gate_W = _logistic(k * (W_out - R_crit))
    return 1.0 - epsilon_w * gate_W * gate_I

@njit(cache=True)
def init_gillespie_state(N, num_mass, n_W, n_B):
    num_states = num_mass + n_W + n_B
    n_rows = int(N * 4) + 20
    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    for i in range(n_B):
        current_state[num_mass + n_W + i] = 1.0
    states = np.zeros((n_rows, num_states), dtype=np.float64)
    states[0] = current_state
    times = np.zeros(n_rows)
    return current_state, states, times, n_rows

@njit(cache=True)
def _run(params: Params, N, t1, num_mass, num_reactions, model, seed):
    """Run Gillespie simulation. model: 0 = SEIPAR, 1 = SEIR (S,E,I,R)."""
    if seed < 0:
        raise ValueError
    np.random.seed(seed)

    n_W = params.n_W
    n_B = params.n_B
    W_start = num_mass
    B_start = num_mass + n_W
    current_state, states, times, n_rows = init_gillespie_state(N, num_mass, n_W, n_B)

    xi_W = float(n_W) / params.tau_W
    xi_B = float(n_B) / params.tau_B

    a = np.zeros(num_reactions, dtype=np.float64)
    coeffs_W = np.zeros(n_W, dtype=np.float64)
    coeffs_B = np.zeros(n_B, dtype=np.float64)
    scratch_W = np.zeros(n_W, dtype=np.float64)
    scratch_B = np.zeros(n_B, dtype=np.float64)

    t = 0.0
    step = 1
    while t < t1:
        S = current_state[0]
        E = current_state[1]
        W_out = current_state[W_start + n_W - 1]
        B_out = current_state[B_start + n_B - 1]

        # calculate propensities
        if model == 0: # SEIPAR
            Ia = current_state[2]
            Ip = current_state[3]
            Is = current_state[4]
            a[0] = B_out * params.beta * (params.phi_a * Ia + params.phi_p * Ip + (1.0 - params.epsilon_s) * Is) * (S / N)  # infection
            a[1] = params.p * E / params.gamma_inv # E -> Ia
            a[2] = (1.0 - params.p) * E / params.gamma_inv # E -> Ip
            a[3] = Ip / params.sigma_inv # Ip -> Is
            a[4] = Ia / params.mu_a_inv # Ia -> R
            a[5] = Is / params.mu_s_inv # Is -> R
        else: # SEIR
            Is = current_state[2]
            a[0] = B_out * params.beta * (1.0 - params.epsilon_s) * Is * (S / N)
            a[1] = E / params.gamma_inv
            a[2] = Is / params.mu_s_inv
        a0 = 0.0
        for i in range(num_reactions):
            a0 += a[i]
        if a0 <= 0.0:
            break

        # advance delay chains
        f_W = _response(W_out, Is / N, params.epsilon_w, params.k, params.R_crit, params.k_I, params.I_crit)
        Rt_in = params.R_0 * params.rho * B_out * (S / N)
        time_step = _exponential_dt(a0)
        _advance_chain(current_state[W_start:W_start + n_W], Rt_in, xi_W, time_step, coeffs_W, scratch_W)
        _advance_chain(current_state[B_start:B_start + n_B], f_W, xi_B, time_step, coeffs_B, scratch_B)

        # execute next event
        event_idx = _select_event(a, a0, num_reactions)
        if model == 0: # SEIPAR
            if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0 # S  -> E
            elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0 # E  -> Ia
            elif event_idx == 2: current_state[1] -= 1.0; current_state[3] += 1.0 # E  -> Ip
            elif event_idx == 3: current_state[3] -= 1.0; current_state[4] += 1.0 # Ip -> Is
            elif event_idx == 4: current_state[2] -= 1.0; current_state[5] += 1.0 # Ia -> R
            else:                current_state[4] -= 1.0; current_state[5] += 1.0 # Is -> R
        else: # SEIR
            if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0 # S -> E
            elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0 # E -> I
            else:                current_state[2] -= 1.0; current_state[3] += 1.0 # I -> R

        t += time_step
        times[step] = t
        states[step, :] = current_state
        step += 1
        if step >= n_rows:
            raise RuntimeError
    return times[:step], states[:step]

def gillespie_SEIPAR_W(params: Params, N: int, t1: float, seed: int = -1):
    """Gillespie SEIPAR algorithm with exact integration of the delay compartments between events."""
    return _run(params=params.concrete(), N=N, t1=t1, num_mass=6, num_reactions=6, model=0, seed=seed)

def gillespie_SEIR_W(params: Params, N: int, t1: float, seed: int = -1):
    """Gillespie SEIR algorithm with exact integration of the delay compartments between events."""
    return _run(params=params.concrete(), N=N, t1=t1, num_mass=4, num_reactions=3, model=1, seed=seed)

def to_uniform_grid(tt, yy, t1, n_ts=None):
    """Resample a Gillespie trace onto the same uniform grid the ODE models save on."""
    grid = np.linspace(0.0, t1, get_n_ts(t1, n_ts))
    return grid, yy[np.clip(np.searchsorted(tt, grid, side="right") - 1, 0, len(tt) - 1)]


### GRIDDED VERSION
@njit(cache=True)
def _fill_grid(grid_states, grid_t, g, t_lo, t_hi, state):
    n = grid_t.shape[0]
    while g < n and grid_t[g] < t_hi:
        if grid_t[g] >= t_lo:
            grid_states[g, :] = state
        g += 1
    return g

@njit(cache=True)
def _run_gridded(params, N, t1, num_mass, num_reactions, model, seed, n_ts):
    if seed < 0:
        raise ValueError
    np.random.seed(seed)
    n_W, n_B = params.n_W, params.n_B
    W_start, B_start = num_mass, num_mass + n_W
    num_states = num_mass + n_W + n_B

    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1)
    current_state[1] = 1.0
    for i in range(n_B):
        current_state[num_mass + n_W + i] = 1.0

    grid_t = np.linspace(0.0, t1, n_ts)
    grid_states = np.zeros((n_ts, num_states), dtype=np.float64)
    g = _fill_grid(grid_states, grid_t, 0, 0.0, np.nextafter(0.0, 1.0), current_state)

    xi_W = float(n_W) / params.tau_W
    xi_B = float(n_B) / params.tau_B
    a = np.zeros(num_reactions, dtype=np.float64)
    coeffs_W = np.zeros(n_W); coeffs_B = np.zeros(n_B)
    scratch_W = np.zeros(n_W); scratch_B = np.zeros(n_B)
    prev_state = np.zeros(num_states, dtype=np.float64)

    t = 0.0
    n_events = 0
    while t < t1:
        prev_state[:] = current_state
        S, E = current_state[0], current_state[1]
        W_out = current_state[W_start + n_W - 1]
        B_out = current_state[B_start + n_B - 1]
        if model == 0: # SEIPAR
            Ia, Ip, Is = current_state[2], current_state[3], current_state[4]
            a[0] = B_out * params.beta * (params.phi_a * Ia + params.phi_p * Ip + (1.0 - params.epsilon_s) * Is) * (S / N)
            a[1] = params.p * E / params.gamma_inv
            a[2] = (1.0 - params.p) * E / params.gamma_inv
            a[3] = Ip / params.sigma_inv
            a[4] = Ia / params.mu_a_inv
            a[5] = Is / params.mu_s_inv
        else: # SEIR
            Is = current_state[2]
            a[0] = B_out * params.beta * (1.0 - params.epsilon_s) * Is * (S / N)
            a[1] = E / params.gamma_inv
            a[2] = Is / params.mu_s_inv
        a0 = 0.0
        for i in range(num_reactions):
            a0 += a[i]
        if a0 <= 0.0:
            break

        f_W = _response(W_out, Is / N, params.epsilon_w, params.k, params.R_crit, params.k_I, params.I_crit)
        Rt_in = params.R_0 * params.rho * B_out * (S / N)
        time_step = _exponential_dt(a0)
        _advance_chain(current_state[W_start:W_start + n_W], Rt_in, xi_W, time_step, coeffs_W, scratch_W)
        _advance_chain(current_state[B_start:B_start + n_B], f_W, xi_B, time_step, coeffs_B, scratch_B)

        t_new = t + time_step
        g = _fill_grid(grid_states, grid_t, g, t, t_new, prev_state)

        event_idx = _select_event(a, a0, num_reactions)
        if model == 0:
            if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0
            elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0
            elif event_idx == 2: current_state[1] -= 1.0; current_state[3] += 1.0
            elif event_idx == 3: current_state[3] -= 1.0; current_state[4] += 1.0
            elif event_idx == 4: current_state[2] -= 1.0; current_state[5] += 1.0
            else:                current_state[4] -= 1.0; current_state[5] += 1.0
        else:
            if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0
            elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0
            else:                current_state[2] -= 1.0; current_state[3] += 1.0
        t = t_new
        n_events += 1

    while g < n_ts:
        grid_states[g, :] = current_state
        g += 1
    return grid_t, grid_states

def gillespie_SEIPAR_W_gridded(params, N, t1, seed, n_ts=None):
    return _run_gridded(params.concrete(), N, t1, 6, 6, 0, seed, get_n_ts(t1, n_ts))

def gillespie_SEIR_W_gridded(params, N, t1, seed, n_ts=None):
    return _run_gridded(params.concrete(), N, t1, 4, 3, 1, seed, get_n_ts(t1, n_ts))
