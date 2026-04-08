"""
Stochastic compartmental models:
    - gillespie_SEIPAR_W with presymptomatic and asymptomatic transmission and wastewater feedback
"""

import numpy as np
from numba import njit

@njit
def gillespie_SEIPAR_W(params, N: int, t1: float):
    """Gillespie algorithm combined with continuous deterministic ODEs for the delay compartments."""
    max_events = int(N * 10)
    times = np.zeros(max_events)
    t = 0.0

    states = np.zeros((max_events, 9), dtype=np.float64) 
    current_state = np.array([float(N-1), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    states[0] = current_state

    r_const = params.p * params.phi * params.mu_a_inv + (1.0 - params.p) * (params.sigma_inv + params.mu_s_inv)
    delay_rate = 3.0 / params.tau
    
    a = np.zeros(6, dtype=np.float64)
    step = 1

    while t < t1 and step < max_events:
        S, E, Ia, Ip, Is, R, W1, W2, W3 = current_state
        f_W3 = 1.0 - (params.epsilon_w / (1.0 + np.exp(-params.k * (W3 - params.R_crit))))

        # propensities
        a[0] = f_W3 * params.beta * (params.phi*Ia + Ip + (1.0-params.epsilon_s)*Is) * (S/N)
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
        time_step = (1.0 / a0) * -np.log(r1) 
        
        # delay compartments W
        Rt = (params.beta * r_const) * params.rho * f_W3 * (S/N)
        current_state[6] += delay_rate * (Rt - W1) * time_step
        current_state[7] += delay_rate * (W1 - W2) * time_step
        current_state[8] += delay_rate * (W2 - W3) * time_step
        
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
