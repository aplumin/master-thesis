"""
Stochastic compartmental models:
    - gillespie_SEIPAR_W with presymptomatic and asymptomatic transmission and wastewater feedback
    - gillespie_SEIAR_W with presymptomatic transmission and wastewater feedback
    - gillespie_SEIR_W with no presymptomatic or asymptomatic transmission and wastewater feedback
"""

import numpy as np
from numba import njit

@njit
def gillespie_SEIPAR_W(params, N: int, t1: float):
    """Gillespie algorithm with exact integration of the delay compartments between events."""
    max_events = int(N * 10)
    n_W = params.n_W
    n_B = params.n_B
    num_mass_compartments = 6 # S,E,Ia,Ip,Is,R
    num_reactions = 6 # S->E, E->Ip, E->Is, Ip->Is, Ia->R, Is->R
    W_start = num_mass_compartments
    B_start = num_mass_compartments + n_W
    num_states = num_mass_compartments + n_W + n_B

    # initialise states
    states = np.zeros((max_events, num_states), dtype=np.float64)
    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    for i in range(n_B):
        current_state[B_start + i] = 1.0
    states[0] = current_state

    xi_W = float(n_W) / params.tau_W
    xi_B = float(n_B) / params.tau_B

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
        Is_frac = Is / N
        # TODO: this is analogous to the response function in the deterministic version but should probably be changed.
        if params.I_crit > 0.0:
            gate_I = 1.0 / (1.0 + np.exp(-params.k_I * (Is_frac - params.I_crit)))
        else:
            gate_I = 1.0
        gate_W = 1.0 / (1.0 + np.exp(-params.k * (W_out - params.R_crit)))
        f_W = 1.0 - params.epsilon_w * gate_W * gate_I

        # propensities
        a[0] = B_out * params.beta * (params.phi*Ia + Ip + (1.0-params.epsilon_s)*Is) * (S/N)
        a[1] = params.p * E / params.gamma_inv
        a[2] = (1.0 - params.p) * E / params.gamma_inv
        a[3] = Ip / params.sigma_inv
        a[4] = Ia / params.mu_a_inv
        a[5] = Is / params.mu_s_inv
        a0 = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
        if a0 == 0.0:
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
        Rt_in = params.R_0 * params.rho * B_out * (S / N)
        xi_W_dt = xi_W * time_step
        exp_neg_xi_W = np.exp(-xi_W_dt)
        # W_i = Rt + exp(-xi*dt) Sum_{j≤i} (xi*dt)^(i-j) / (i-j)! * (W_j - Rt)
        # with poisson coeffs (xi*dt)^k / k!
        coeffs_W[0] = 1.0
        for m in range(1, n_W):
            coeffs_W[m] = coeffs_W[m-1] * xi_W_dt / m
        for i in range(n_W):
            W_i = Rt_in
            for j in range(i + 1):
                W_i += exp_neg_xi_W * coeffs_W[i-j] * (W_old[j] - Rt_in)
            current_state[W_start + i] = W_i
        
        # B delay chain
        B_in = f_W
        xi_B_dt = xi_B * time_step
        exp_neg_xi_B = np.exp(-xi_B_dt)
        coeffs_B[0] = 1.0
        for m in range(1, n_B):
            coeffs_B[m] = coeffs_B[m-1] * xi_B_dt / m
        for i in range(n_B):
            B_i = B_in
            for j in range(i + 1):
                B_i += exp_neg_xi_B * coeffs_B[i-j] * (B_old[j] - B_in)
            current_state[B_start + i] = B_i

        # draw next event
        r2 = np.random.random() * a0
        cumsum = 0.0
        event_idx = 0
        for i in range(num_reactions):
            cumsum += a[i]
            if r2 < cumsum:
                event_idx = i
                break

        # execute next event
        if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0 # S  -> E
        elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0 # E  -> Ia
        elif event_idx == 2: current_state[1] -= 1.0; current_state[3] += 1.0 # E  -> Ip
        elif event_idx == 3: current_state[3] -= 1.0; current_state[4] += 1.0 # Ip -> Is
        elif event_idx == 4: current_state[2] -= 1.0; current_state[5] += 1.0 # Ia -> R
        elif event_idx == 5: current_state[4] -= 1.0; current_state[5] += 1.0 # Is -> R

        # advance time and save state
        t += time_step
        times[step] = t
        states[step, :] = current_state 
        step += 1
        
    return times[:step], states[:step]


@njit
def gillespie_SEIAR_W(params, N: int, t1: float):
    """Gillespie algorithm with exact integration of the delay compartments between events."""
    max_events = int(N * 10)
    n_W = params.n_W
    n_B = params.n_B
    num_mass_compartments = 5 # S,E,Ia,Is,R
    num_reactions = 5 # S->E, E->Ia, E->Is, Ia->R, Is->R
    W_start = num_mass_compartments
    B_start = num_mass_compartments + n_W
    num_states = num_mass_compartments + n_W + n_B

    # initialise states
    states = np.zeros((max_events, num_states), dtype=np.float64)
    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    for i in range(n_B):
        current_state[B_start + i] = 1.0
    states[0] = current_state

    xi_W = float(n_W) / params.tau_W
    xi_B = float(n_B) / params.tau_B

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
        Is = current_state[3]
        W_out = current_state[W_start + n_W - 1]
        B_out = current_state[B_start + n_B - 1]

        # logistic response
        Is_frac = Is / N
        # TODO: this is analogous to the response function in the deterministic version but should probably be changed.
        if params.I_crit > 0.0:
            gate_I = 1.0 / (1.0 + np.exp(-params.k_I * (Is_frac - params.I_crit)))
        else:
            gate_I = 1.0
        gate_W = 1.0 / (1.0 + np.exp(-params.k * (W_out - params.R_crit)))
        f_W = 1.0 - params.epsilon_w * gate_W * gate_I

        # propensities
        a[0] = B_out * params.beta * (params.phi*Ia + (1.0-params.epsilon_s)*Is) * (S/N)
        a[1] = params.p * E / params.gamma_inv
        a[2] = (1.0 - params.p) * E / params.gamma_inv
        a[3] = Ia / params.mu_a_inv
        a[4] = Is / params.mu_s_inv
        a0 = a[0] + a[1] + a[2] + a[3] + a[4]
        if a0 == 0.0:
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
        Rt_in = params.R_0 * params.rho * B_out * (S / N)
        xi_W_dt = xi_W * time_step
        exp_neg_xi_W = np.exp(-xi_W_dt)
        # W_i = Rt + exp(-xi*dt) Sum_{j≤i} (xi*dt)^(i-j) / (i-j)! * (W_j - Rt)
        # with poisson coeffs (xi*dt)^k / k!
        coeffs_W[0] = 1.0
        for m in range(1, n_W):
            coeffs_W[m] = coeffs_W[m-1] * xi_W_dt / m
        for i in range(n_W):
            W_i = Rt_in
            for j in range(i + 1):
                W_i += exp_neg_xi_W * coeffs_W[i-j] * (W_old[j] - Rt_in)
            current_state[W_start + i] = W_i
        
        # B delay chain
        B_in = f_W
        xi_B_dt = xi_B * time_step
        exp_neg_xi_B = np.exp(-xi_B_dt)
        coeffs_B[0] = 1.0
        for m in range(1, n_B):
            coeffs_B[m] = coeffs_B[m-1] * xi_B_dt / m
        for i in range(n_B):
            B_i = B_in
            for j in range(i + 1):
                B_i += exp_neg_xi_B * coeffs_B[i-j] * (B_old[j] - B_in)
            current_state[B_start + i] = B_i

        # draw next event
        r2 = np.random.random() * a0
        cumsum = 0.0
        event_idx = 0
        for i in range(num_reactions):
            cumsum += a[i]
            if r2 < cumsum:
                event_idx = i
                break

        # execute next event
        if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0 # S  -> E
        elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0 # E  -> Ia
        elif event_idx == 2: current_state[1] -= 1.0; current_state[3] += 1.0 # E  -> Is
        elif event_idx == 3: current_state[2] -= 1.0; current_state[4] += 1.0 # Ia -> R
        elif event_idx == 4: current_state[3] -= 1.0; current_state[4] += 1.0 # Is -> R

        # advance time and save state
        t += time_step
        times[step] = t
        states[step, :] = current_state 
        step += 1
        
    return times[:step], states[:step]


@njit
def gillespie_SEIR_W(params, N: int, t1: float):
    """Gillespie algorithm with exact integration of the delay compartments between events."""
    max_events = int(N * 10)
    n_W = params.n_W
    n_B = params.n_B
    num_mass_compartments = 4 # S,E,I,R
    num_reactions = 3 # S->E, E->I, I->R
    W_start = num_mass_compartments
    B_start = num_mass_compartments + n_W
    num_states = num_mass_compartments + n_W + n_B

    # initialise states
    states = np.zeros((max_events, num_states), dtype=np.float64)
    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    for i in range(n_B):
        current_state[B_start + i] = 1.0
    states[0] = current_state

    xi_W = float(n_W) / params.tau_W
    xi_B = float(n_B) / params.tau_B

    a = np.zeros(num_reactions, dtype=np.float64)
    coeffs_W = np.zeros(n_W, dtype=np.float64)
    coeffs_B = np.zeros(n_B, dtype=np.float64)
    W_old = np.zeros(n_W, dtype=np.float64)
    B_old = np.zeros(n_B, dtype=np.float64)

    times = np.zeros(max_events)
    t = 0.0
    step = 1
    while t < t1 and step < max_events:
        S  = current_state[0]
        E  = current_state[1]
        Is = current_state[2]
        W_out = current_state[W_start + n_W - 1]
        B_out = current_state[B_start + n_B - 1]

        Is_frac = Is / N
        if params.I_crit > 0.0:
            gate_I = 1.0 / (1.0 + np.exp(-params.k_I * (Is_frac - params.I_crit)))
        else:
            gate_I = 1.0
        gate_W = 1.0 / (1.0 + np.exp(-params.k * (W_out - params.R_crit)))
        f_W = 1.0 - params.epsilon_w * gate_W * gate_I

        # propensities
        a[0] = B_out * params.beta * (1.0 - params.epsilon_s) * Is * (S / N)
        a[1] = E / params.gamma_inv
        a[2] = Is / params.mu_s_inv
        a0 = a[0] + a[1] + a[2]
        if a0 == 0.0:
            break

        # draw time step
        r1 = np.random.random()
        time_step = -np.log(r1) / a0

        # draw next event
        for i in range(n_W):
            W_old[i] = current_state[W_start + i]
        for i in range(n_B):
            B_old[i] = current_state[B_start + i]

        # W delay chain
        Rt_in = params.R_0 * params.rho * B_out * (S / N)
        xi_W_dt = xi_W * time_step
        exp_neg_xi_W = np.exp(-xi_W_dt)
        coeffs_W[0] = 1.0
        for m in range(1, n_W):
            coeffs_W[m] = coeffs_W[m-1] * xi_W_dt / m
        for i in range(n_W):
            W_i = Rt_in
            for j in range(i + 1):
                W_i += exp_neg_xi_W * coeffs_W[i-j] * (W_old[j] - Rt_in)
            current_state[W_start + i] = W_i

        # B delay chain
        B_in = f_W
        xi_B_dt = xi_B * time_step
        exp_neg_xi_B = np.exp(-xi_B_dt)
        coeffs_B[0] = 1.0
        for m in range(1, n_B):
            coeffs_B[m] = coeffs_B[m-1] * xi_B_dt / m
        for i in range(n_B):
            B_i = B_in
            for j in range(i + 1):
                B_i += exp_neg_xi_B * coeffs_B[i-j] * (B_old[j] - B_in)
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

        if   event_idx == 0: current_state[0] -= 1.0; current_state[1] += 1.0  # S -> E
        elif event_idx == 1: current_state[1] -= 1.0; current_state[2] += 1.0  # E -> I
        elif event_idx == 2: current_state[2] -= 1.0; current_state[3] += 1.0  # I -> R

        t += time_step
        times[step] = t
        states[step, :] = current_state
        step += 1

    return times[:step], states[:step]