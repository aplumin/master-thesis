"""
Gillespie SEIPAR model with superspreading.

Superspreading is modelled by drawing the number of secondary cases of a transmission
event from a NB(k_ss, p_nb) distribution.
"""

import numpy as np
import jax
from numba import njit

from models.parameters import Params
from models.metrics import outcome_metrics, calculate_mt_branching_q_with_superspreading, establishment_threshold
from models.gillespie import _advance_chain, _exponential_dt, _select_event, _response, to_uniform_grid


@njit(cache=True, fastmath=True)
def gillespie_SEIPAR_W_superspreading(
    params, N: int, t1: float, k_ss: float, a_ss: bool = False, p_ss: bool = False, s_ss: bool = False, seed: int = -1):
    """Gillespie algorithm with exact integration of the delay compartments and superspreading."""
    if seed >= 0:
        np.random.seed(seed)

    num_mass = 6 # S,E,Ia,Ip,Is,R
    num_reactions = 8 # S(Ia)->E, S(Ip)->E, S(Is)->E, E->Ia, E->Ip, Ip->Is, Ia->R, Is->R
    n_W = params.n_W
    n_B = params.n_B
    W_start = num_mass
    B_start = num_mass + n_W
    num_states = num_mass + n_W + n_B
    n_rows = int(N * 4) + 20

    N_inv = 1.0 / float(N)
    beta = params.beta
    phi_a = params.phi_a
    phi_p = params.phi_p
    eps_s_comp = 1.0 - params.epsilon_s
    eps_w = params.epsilon_w
    I_crit = params.I_crit
    k_I = params.k_I
    R_crit = params.R_crit
    k_W = params.k
    R0_rho = params.R_0 * params.rho

    rate_E_Ia = params.p / params.gamma_inv
    rate_E_Ip = (1.0 - params.p) / params.gamma_inv
    rate_Ip = 1.0 / params.sigma_inv
    rate_Ia = 1.0 / params.mu_a_inv
    rate_Is = 1.0 / params.mu_s_inv

    xi_W = float(n_W) / params.tau_W
    xi_B = float(n_B) / params.tau_B

    # negative binomial s.t. the mean is 1
    p_nb = k_ss / (k_ss + 1.0) if k_ss > 0.0 else 1.0
    can_superspread = k_ss > 0.0
    ss_a = a_ss and can_superspread # asymptomatics can superspread
    ss_p = p_ss and can_superspread # presymptomatics can superspread
    ss_s = s_ss and can_superspread # symptomatics can superspread

    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    for i in range(n_B):
        current_state[B_start + i] = 1.0

    states = np.zeros((n_rows, num_states), dtype=np.float64)
    times = np.zeros(n_rows)
    states[0] = current_state

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
        Ia = current_state[2]
        Ip = current_state[3]
        Is = current_state[4]
        W_out = current_state[W_start + n_W - 1]
        B_out = current_state[B_start + n_B - 1]

        base_infection_rate = B_out * beta * (S * N_inv)
        a[0] = base_infection_rate * phi_a * Ia # infection from asymptomatic
        a[1] = base_infection_rate * phi_p * Ip # infection from presymptomatic
        a[2] = base_infection_rate * eps_s_comp * Is # infection from symptomatic
        a[3] = E * rate_E_Ia # E -> Ia
        a[4] = E * rate_E_Ip # E -> Ip
        a[5] = Ip * rate_Ip # Ip -> Is
        a[6] = Ia * rate_Ia # Ia -> R
        a[7] = Is * rate_Is # Is -> R
        a0 = 0.0
        for i in range(num_reactions):
            a0 += a[i]
        if a0 <= 0.0:
            break

        # advance delay chains
        f_W = _response(W_out, Is * N_inv, eps_w, k_W, R_crit, k_I, I_crit)
        Rt_in = R0_rho * B_out * (S * N_inv)
        time_step = _exponential_dt(a0)
        _advance_chain(current_state[W_start:W_start + n_W], Rt_in, xi_W, time_step, coeffs_W, scratch_W)
        _advance_chain(current_state[B_start:B_start + n_B], f_W, xi_B, time_step, coeffs_B, scratch_B)

        # execute next event
        event_idx = _select_event(a, a0, num_reactions)
        if event_idx <= 2: # superspreading if event is a transmission and the infector can superspread
            Z = 1.0
            if (event_idx == 0 and ss_a) or (event_idx == 1 and ss_p) or (event_idx == 2 and ss_s):
                Z = float(np.random.negative_binomial(k_ss, p_nb))
            if Z > current_state[0]:
                Z = current_state[0]
            current_state[0] -= Z; current_state[1] += Z                       # S  -> E
        elif event_idx == 3: current_state[1] -= 1.0; current_state[2] += 1.0  # E  -> Ia
        elif event_idx == 4: current_state[1] -= 1.0; current_state[3] += 1.0  # E  -> Ip
        elif event_idx == 5: current_state[3] -= 1.0; current_state[4] += 1.0  # Ip -> Is
        elif event_idx == 6: current_state[2] -= 1.0; current_state[5] += 1.0  # Ia -> R
        else:                current_state[4] -= 1.0; current_state[5] += 1.0  # Is -> R

        t += time_step
        times[step] = t
        states[step, :] = current_state
        step += 1
        if step >= n_rows:
            break
    return times[:step], states[:step]


def _summarise(values, reducer=np.mean):
    return float(reducer(values)) if len(values) else float("nan")

def simulate_superspreading_outcomes(eps_ww, kk, eps_s, t1, N, num_simulations, scenario, npz, seed=0, alpha=0.01):
    shape = (len(eps_ww), len(kk))
    keys = ("Rt", "time_to_below", "Itot", "peak_Is", "extinction_time")
    mean_grids = {k: np.full(shape, np.nan) for k in keys}
    var_grids = {k: np.full(shape, np.nan) for k in keys}
    n_kept = np.zeros(shape, dtype=int)

    metrics_fn = jax.jit(outcome_metrics, static_argnames=("population_size",))

    for i, ew in enumerate(eps_ww):
        for j, k_ss in enumerate(kk):
            ps = Params.for_SEIPAR(epsilon_s=float(eps_s), epsilon_w=float(ew))
            q = calculate_mt_branching_q_with_superspreading(k_ss, ps, ew, eps_s)
            Iest = establishment_threshold(q=q, alpha=alpha)
            samples = {k: [] for k in keys}
            for rep in range(num_simulations):
                run_seed = abs(hash((seed, i, j, rep))) % (2**31 - 1)
                tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=k_ss, a_ss=True, p_ss=True, s_ss=True, seed=run_seed)
                tt, yy = to_uniform_grid(tt, yy, t1)
                if scenario == "establishment" and np.max(yy[:, 2] + yy[:, 3] + yy[:, 4]) < Iest:
                    continue
                Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = metrics_fn(tt, yy, ps, t1, population_size=N)
                for k, v in zip(keys, (Rt, time_to_below, Itot, peak_Is, extinction_time)):
                    samples[k].append(float(v))
            n_kept[i, j] = len(samples["Rt"])
            for k in keys:
                reducer = (lambda x: np.percentile(x, 95)) if k == "extinction_time" else np.mean
                mean_grids[k][i, j] = _summarise(samples[k], reducer)
                var_grids[k][i, j] = _summarise(samples[k], np.var)

    np.savez_compressed(npz, 
        Rt_grid=mean_grids["Rt"], Rt_var_grid=var_grids["Rt"],
        time_to_below_grid=mean_grids["time_to_below"], time_to_below_var_grid=var_grids["time_to_below"],
        Itot_grid=mean_grids["Itot"], Itot_var_grid=var_grids["Itot"],
        peak_Is_grid=mean_grids["peak_Is"], peak_Is_var_grid=var_grids["peak_Is"],
        extinction_time_grid=mean_grids["extinction_time"], extinction_time_var_grid=var_grids["extinction_time"],
        n_kept=n_kept)
