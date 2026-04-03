"""
Deterministic compartmental models:
    - SEIPAR_W with presymptomatic and asymptomatic transmission and wastewater feedback
    - SEIAR_W with presymptomatic transmission and wastewater feedback
    - SEIR_W with no presymptomatic or asymptomatic transmission and wastewater feedback
"""

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from functools import partial
from typing import Callable

from models.parameters import Params, f, update_epsilons, update_asymptomatic_params


def SEIPAR_W(t, y, params):
    """Compartmental model with linear feedback chain for delayed wastewater response."""
    S, E, Ia, Ip, Is, R, W1, W2, W3 = y
    f_W3 = f(W3, params)
    lambda_S = f_W3 * params.beta * (params.phi * Ia + Ip + (1.0 - params.epsilon_s) * Is) * S
    become_infectious = E / params.gamma_inv
    become_symptomatic = Ip / params.sigma_inv
    recover_asyx = Ia / params.mu_a_inv
    recover_syx = Is / params.mu_s_inv
    
    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dIa = params.p * become_infectious - recover_asyx
    dIp = (1.0 - params.p) * become_infectious - become_symptomatic
    dIs = become_symptomatic - recover_syx
    dR = recover_asyx + recover_syx

    # delay compartments
    delay_rate = 3.0 / params.tau # TODO: make num_compartments flexible (fit to empirical delay distributions)
    Rt = params.R_0 * params.rho * f_W3 * S # use current Rt as input to the linear chain
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)

    return jnp.array([dS, dE, dIa, dIp, dIs, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIPAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6):
    solution = diffeqsolve(
        ODETerm(SEIPAR_W), Tsit5(),
        t0 = 0.0, t1 = t1,  dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


def SEIAR_W(t, y, params):
    """Simplified compartmental model without presymptomatic transmission."""
    S, E, Ia, Is, R, W1, W2, W3 = y
    f_W3 = f(W3, params)
    lambda_S = f_W3 * params.beta * (params.phi * Ia + (1.0 - params.epsilon_s) * Is) * S
    become_infectious = E / params.gamma_inv
    recover_asyx = Ia / params.mu_a_inv
    recover_syx = Is / params.mu_s_inv

    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dIa = params.p * become_infectious - recover_asyx
    dIs = (1.0 - params.p) * become_infectious - recover_syx
    dR = recover_asyx + recover_syx

    # delay compartments
    delay_rate = 3.0 / params.tau
    Rt = params.R_0 * params.rho * f_W3 * S # use current Rt as input to the linear chain
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)
    return jnp.array([dS, dE, dIa, dIs, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIAR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6):
    solution = diffeqsolve(
        ODETerm(SEIAR_W), Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


def SEIR_W(t, y, params):
    """Simplified compartmental model without presymptomatic or asymptomatic transmission."""
    S, E, II, R, W1, W2, W3 = y
    f_W3 = f(W3, params)
    lambda_S = f_W3 * params.beta * (1.0 - params.epsilon_s) * II * S
    become_infectious = E / params.gamma_inv
    recover = II / params.mu_s_inv

    # flow compartments
    dS = -lambda_S
    dE = lambda_S - become_infectious
    dI = become_infectious - recover
    dR = recover

    # delay compartments
    delay_rate = 3.0 / params.tau
    Rt = params.R_0 * params.rho * f_W3 * S # use current Rt as input to the linear chain
    dW1 = delay_rate * (Rt - W1)
    dW2 = delay_rate * (W1 - W2)
    dW3 = delay_rate * (W2 - W3)
    return jnp.array([dS, dE, dI, dR, dW1, dW2, dW3])

@partial(jax.jit, static_argnames=['t1'])
def simulate_SEIR_W(params: Params = Params.for_SEIPAR(), t1: float = 100.0, E0: float = 1e-6):
    solution = diffeqsolve(
        ODETerm(SEIR_W), Tsit5(),
        t0 = 0.0, t1 = t1, dt0 = 0.1,
        y0 = jnp.array([1.0 - E0, E0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        args = params, 
        saveat = SaveAt(ts=jnp.linspace(0.0, t1, 5000)),
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), max_steps = 50_000
    )
    return solution.ts, solution.ys


# PLOTTING FUNCTIONS
@partial(jax.jit, static_argnames=['model', 't1'])
def compute_R_grid(model: Callable, base_params: Params, eps_ww: float, eps_ss: float, t1: float = 100.0, E0: float = 1e-6):
    """Compute a 2D grid of Rt values with wastewater warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = update_epsilons(base_params, w, s)
        _, yy = model(params=params, t1=t1, E0=E0)
        return params.R_0 * params.rho * f(yy[-1,-1], params) * yy[-1,0]
    return jax.vmap(jax.vmap(final_R, in_axes=(None, 0)), in_axes=(0, None))(eps_ww, eps_ss)

def plot_final_R(model=simulate_SEIPAR_W, params=Params.for_SEIPAR(), t1=100.0, E0=1e-6, title=None):
    """
    Plot a grid of the reproductive number after interventions.
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    eps_ww = jnp.linspace(0.0, 0.999, 100)
    eps_ss = jnp.linspace(0.0, 0.999, 100)
    EPS_W, EPS_S = jnp.meshgrid(eps_ww, eps_ss, indexing='ij')

    R_end_vals = compute_R_grid(model, params, eps_ww, eps_ss, t1, E0)
    fig = plt.figure()
    mesh = plt.pcolormesh(EPS_W, EPS_S, R_end_vals, cmap='RdBu_r', norm=colors.CenteredNorm(vcenter=1.0))
    plt.colorbar(mesh)
    plt.contour(EPS_W, EPS_S, R_end_vals, levels=[1.0], colors='k')
    plt.xlabel('Warning response efficacy')
    plt.ylabel('Isolation efficacy')
    plt.title(title)
    return fig

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid(model: Callable, base_params: Params, eps_ww, eps_ss, t1: float = 100.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline. 
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    def I_tot(w, s):
        params = update_epsilons(base_params, w, s)
        _, yy =  model(params=params, t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(None, 0)), in_axes=(0, None))(eps_ww, eps_ss)
    return I_tot_grid / I_tot(0.0, 0.0)

def plot_I_tot(model=simulate_SEIPAR_W, params=Params.for_SEIPAR(), title=None, t1=600.0, E0=1e-6):
    """
    Plot a grid of the total proportion infected after interventions (compared to baseline without interventions).
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    eps_ww = jnp.linspace(0.0, 0.999, 100)
    eps_ss = jnp.linspace(0.0, 0.999, 100)
    EPS_W, EPS_S = jnp.meshgrid(eps_ww, eps_ss, indexing='ij')

    fig = plt.figure()
    mesh = plt.pcolormesh(EPS_W, EPS_S, compute_I_tot_grid(model, params, eps_ww, eps_ss, t1, E0), cmap='viridis')
    fig.colorbar(mesh)
    plt.xlabel('Warning response efficacy')
    plt.ylabel('Isolation efficacy')
    plt.title(title)
    return fig

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Rt_final(model: Callable, base_params: Params, p: float, phi: float, t1: float = 50.0, E0: float = 1e-6):
    """
    Compute a 2D grid of the reproductive number after interventions.
    Asymptomatic proportion p on the x axis and relative infectiousness phi on the y axis.
    """
    def final_R(p, phi):
        params = update_asymptomatic_params(params=base_params, p=p, phi=phi)
        _, yy = model(params=params, t1=t1, E0=E0)
        return params.R_0 * params.rho * f(yy[-1,-1], params) * yy[-1,0]
    return jax.vmap(jax.vmap(final_R, in_axes=(None, 0)), in_axes=(0, None))(p, phi)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Itot_final(model: Callable, base_params: Params, p: float, phi: float, t1: float = 600.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline.
    Asymptomatic proportion p on the x axis and relative infectiousness phi on the y axis.
    """
    def I_tot(p, phi):
        params = update_asymptomatic_params(params=base_params, p=p, phi=phi)
        _, yy =  model(params=params, t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(None, 0)), in_axes=(0, None))(p, phi)
    return I_tot_grid / I_tot(0.0, 0.0)
