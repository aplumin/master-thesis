"""
Stochastic compartmental models with superspreading.
"""

import numpy as np
from numba import njit

from models.parameters import Params
from models.metrics import outcome_metrics, calculate_mt_branching_q_with_superspreading


@njit(fastmath=True)
def gillespie_SEIPAR_W_superspreading(params, N: int, t1: float, k_ss: float, a_ss: bool = False, p_ss: bool = False, s_ss: bool = False):
    """Gillespie algorithm with exact integration of the delay compartments and superspreading."""
    max_events = int(N * 10)
    n_W = params.n_W
    n_B = params.n_B
    num_mass_compartments = 6 # S,E,Ia,Ip,Is,R
    num_reactions = 8 # S(Ia)->E, S(Ip)->E, S(Is)->E, E->Ip, E->Is, Ip->Is, Ia->R, Is->R
    W_start = num_mass_compartments
    B_start = num_mass_compartments + n_W
    num_states = num_mass_compartments + n_W + n_B

    # precompute everything for efficiency
    inv_N = 1.0 / float(N)
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

    p_nb = k_ss / (k_ss + 1.0) if k_ss > 0.0 else 1.0
    can_superspread = k_ss > 0.0
    superspreading_flags = (a_ss and can_superspread, p_ss and can_superspread, s_ss and can_superspread)

    # initialise states
    states = np.zeros((max_events, num_states), dtype=np.float64)
    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    for i in range(n_B):
        current_state[B_start + i] = 1.0
    states[0] = current_state
    a = np.zeros(num_reactions, dtype=np.float64)
    coeffs_W = np.zeros(n_W, dtype=np.float64)
    coeffs_B = np.zeros(n_B, dtype=np.float64)
    W_old = np.zeros(n_W, dtype=np.float64)
    B_old = np.zeros(n_B, dtype=np.float64)

    # main loop
    times = np.zeros(max_events)
    t = 0.0
    step = 1
    while t < t1 and step < max_events:
        # unpack compartments
        S  = current_state[0]
        E  = current_state[1]
        Ia = current_state[2]
        Ip = current_state[3]
        Is = current_state[4]
        W_out = current_state[W_start + n_W - 1]
        B_out = current_state[B_start + n_B - 1]

        # logistic response
        Is_frac = Is * inv_N
        # TODO: this is analogous to the response function in the deterministic version but should probably be changed.
        if I_crit > 0.0:
            gate_I = 1.0 / (1.0 + np.exp(-k_I * (Is_frac - I_crit)))
        else:
            gate_I = 1.0
        gate_W = 1.0 / (1.0 + np.exp(-k_W * (W_out - R_crit)))
        f_W = 1.0 - eps_w * gate_W * gate_I

        # propensities
        base_infection_rate = B_out * beta * (S * inv_N)
        a[0] = base_infection_rate * phi_a * Ia    
        a[1] = base_infection_rate * phi_p * Ip          
        a[2] = base_infection_rate * eps_s_comp * Is 
        a[3] = E * rate_E_Ia                     
        a[4] = E * rate_E_Ip                     
        a[5] = Ip * rate_Ip                      
        a[6] = Ia * rate_Ia                      
        a[7] = Is * rate_Is                      
        a0 = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
        if a0 <= 0.0:
            break

        # draw time step
        r1 = np.random.random()
        time_step = -np.log(r1) / a0

        # save old delay states
        for i in range(n_W):
            W_old[i] = current_state[W_start + i]
        for i in range(n_B):
            B_old[i] = current_state[B_start + i]

        # W delay chain
        Rt_in = R0_rho * B_out * (S * inv_N)
        xi_W_dt = xi_W * time_step
        # W_i = Rt + exp(-xi*dt) Sum_{j≤i} (xi*dt)^(i-j) / (i-j)! * (W_j - Rt)
        # with poisson coeffs (xi*dt)^k / k!
        coeffs_W[0] = np.exp(-xi_W_dt)
        for m in range(1, n_W):
            coeffs_W[m] = coeffs_W[m-1] * xi_W_dt / m
        for i in range(n_W):
            W_i = Rt_in
            for j in range(i + 1):
                W_i += coeffs_W[i-j] * (W_old[j] - Rt_in) 
            current_state[W_start + i] = W_i
        
        # B delay chain
        B_in = f_W
        xi_B_dt = xi_B * time_step
        coeffs_B[0] = np.exp(-xi_B_dt)
        for m in range(1, n_B):
            coeffs_B[m] = coeffs_B[m-1] * xi_B_dt / m
        for i in range(n_B):
            B_i = B_in
            for j in range(i + 1):
                B_i += coeffs_B[i-j] * (B_old[j] - B_in)
            current_state[B_start + i] = B_i

        # draw next event
        r2 = np.random.random() * a0
        cumsum = 0.0
        event_idx = 0
        for i in range(num_reactions):
            cumsum += a[i]
            if r2 < cumsum:
                event_idx = i
                break

        # execute next event
        if event_idx <= 2:
            Z = 1.0
            if superspreading_flags[event_idx]:
                Z = float(np.random.negative_binomial(k_ss, p_nb))
            Z = min(Z, current_state[0]) 
            current_state[0] -= Z; current_state[1] += Z                      # S  -> E
        elif event_idx == 3: current_state[1] -= 1.0; current_state[2] += 1.0 # E  -> Ia
        elif event_idx == 4: current_state[1] -= 1.0; current_state[3] += 1.0 # E  -> Ip
        elif event_idx == 5: current_state[3] -= 1.0; current_state[4] += 1.0 # Ip -> Is
        elif event_idx == 6: current_state[2] -= 1.0; current_state[5] += 1.0 # Ia -> R
        elif event_idx == 7: current_state[4] -= 1.0; current_state[5] += 1.0 # Is -> R

        # advance time and save state
        t += time_step
        times[step] = t
        states[step, :] = current_state 
        step += 1
        
    return times[:step], states[:step]

def simulate_superspreading_outcomes(eps_ww, kk, eps_s, t1, N, num_simulations, scenario, npz):
    Rt_grid = np.zeros((len(eps_ww), len(kk)))
    Rt_var_grid = np.zeros((len(eps_ww), len(kk)))
    time_to_below_grid = np.zeros((len(eps_ww), len(kk)))
    time_to_below_var_grid = np.zeros((len(eps_ww), len(kk)))
    Itot_grid = np.zeros((len(eps_ww), len(kk)))
    Itot_var_grid = np.zeros((len(eps_ww), len(kk)))
    peak_Is_grid = np.zeros((len(eps_ww), len(kk)))
    peak_Is_var_grid = np.zeros((len(eps_ww), len(kk)))
    extinction_time_grid = np.zeros((len(eps_ww), len(kk)))
    extinction_time_var_grid = np.zeros((len(eps_ww), len(kk)))
    
    for i, ew in enumerate(eps_ww):
        for j, k_ss in enumerate(kk):
            Rt_list = []
            time_to_below_list = []
            Itot_list = []
            peak_Is_list = []
            extinction_time_list = []
            ps = Params.for_SEIPAR(epsilon_s=float(eps_s), epsilon_w=float(ew))
            alpha = 0.01
            q = calculate_mt_branching_q_with_superspreading(k_ss, ps, ew, eps_s)
            Iest = np.ceil(np.log(alpha) / np.log(q)) if 0.0 < q < 1.0 else 1.0
            for _ in range(num_simulations):
                tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=k_ss, a_ss=True, p_ss=True, s_ss=False)
                if (scenario == 'establishment') & (np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest): continue
                Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = outcome_metrics(tt, yy, Params.for_SEIPAR(epsilon_s=eps_s, epsilon_w=ew), t1, population_size=N)
                Rt_list.append(Rt)
                time_to_below_list.append(time_to_below)
                Itot_list.append(Itot)
                peak_Is_list.append(peak_Is)
                extinction_time_list.append(extinction_time)
            Rt_grid[i,j] = np.mean(Rt_list) if Rt_list else np.nan
            Rt_var_grid[i,j] = np.var(Rt_list)
            time_to_below_grid[i,j] = np.mean(time_to_below_list)
            time_to_below_var_grid[i,j] = np.var(time_to_below_list)
            Itot_grid[i,j] = np.mean(Itot_list)
            Itot_var_grid[i,j] = np.var(Itot_list)
            peak_Is_grid[i,j] = np.mean(peak_Is_list)
            peak_Is_var_grid[i,j] = np.var(peak_Is_list)
            percentile_95 = np.percentile(extinction_time_list, 95) if extinction_time_list else np.nan
            extinction_time_grid[i,j] = percentile_95
            extinction_time_var_grid[i,j] = np.var(extinction_time_list)
    
    np.savez_compressed(npz, Rt_grid=Rt_grid, Rt_var_grid=Rt_var_grid, time_to_below_grid=time_to_below_grid, time_to_below_var_grid=time_to_below_var_grid, Itot_grid=Itot_grid, Itot_var_grid=Itot_var_grid, peak_Is_grid=peak_Is_grid, peak_Is_var_grid=peak_Is_var_grid, extinction_time_grid=extinction_time_grid, extinction_time_var_grid=extinction_time_var_grid)
