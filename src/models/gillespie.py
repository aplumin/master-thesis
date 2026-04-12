"""
TODO: add superspreading (draw from negative binomial)
TODO: add stochastic SEIAR and SEIR models

Stochastic compartmental models:
    - gillespie_SEIPAR_W with presymptomatic and asymptomatic transmission and wastewater feedback
"""

import numpy as np
from numba import njit

@njit
def gillespie_SEIPAR_W(params, N: int, t1: float):
    """Gillespie algorithm with exact integration of the delay compartments between events."""
    max_events = int(N * 10)
    times = np.zeros(max_events)
    t = 0.0

    # initialise states
    num_delay_compartments = params.num_delay_compartments
    num_states = 6 + num_delay_compartments
    states = np.zeros((max_events, num_states), dtype=np.float64)
    current_state = np.zeros(num_states, dtype=np.float64)
    current_state[0] = float(N - 1) # S
    current_state[1] = 1.0 # E
    states[0] = current_state

    r_const = params.p * params.phi * params.mu_a_inv + (1.0 - params.p) * (params.sigma_inv + params.mu_s_inv)
    delay_rate = float(num_delay_compartments) / params.tau
    a = np.zeros(6, dtype=np.float64)
    coeffs = np.zeros(num_delay_compartments, dtype=np.float64)
    step = 1

    while t < t1 and step < max_events:
        S  = current_state[0]
        E  = current_state[1]
        Ia = current_state[2]
        Ip = current_state[3]
        Is = current_state[4]
        W_out = current_state[5 + num_delay_compartments]

        f_W = 1.0 - (params.epsilon_w / (1.0 + np.exp(-params.k * (W_out - params.R_crit))))

        # propensities
        a[0] = f_W * params.beta * (params.phi*Ia + Ip + (1.0-params.epsilon_s)*Is) * (S/N)
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

        # delay compartments W
        Rt = (params.beta * r_const) * params.rho * f_W * (S/N)
        xi_dt = delay_rate * time_step
        exp_neg_xi_dt = np.exp(-xi_dt)

        # W_i = Rt + exp(-xi*dt) Sum_{j≤i} (xi*dt)^(i-j) / (i-j)! * (W_j - Rt)
        # with poisson coeffs (xi*dt)^k / k!
        coeffs[0] = 1.0
        for k in range(1, num_delay_compartments):
            coeffs[k] = coeffs[k-1] * xi_dt / k

        for i in range(num_delay_compartments):
            W_i = Rt
            for j in range(i+1):
                W_i += exp_neg_xi_dt * coeffs[i-j] * (current_state[6+j] - Rt)
            current_state[6+i] = W_i
        
        # draw next event
        r2 = np.random.random() * a0
        cumsum = 0.0
        event_idx = 0
        for i in range(6):
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
