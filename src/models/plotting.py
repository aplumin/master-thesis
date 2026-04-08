"""
Plotting functions.
"""

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt

from models.parameters import Params, f, update_epsilons, update_asymptomatic_params
from models.compartmental import simulate_SEIPAR_W, simulate_SEIPAR_W_with_I_gate


@partial(jax.jit, static_argnames=['model', 't1'])
def compute_R_grid(model: Callable, base_params: Params, eps_ww: float, eps_ss: float, t1: float = 100.0, E0: float = 1e-6):
    """Compute a 2D grid of Rt values with wastewater warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = update_epsilons(base_params, w, s)
        _, yy = model(params=params, t1=t1, E0=E0)
        return params.R_0 * params.rho * f(yy[-1,-1], params) * yy[-1,0]
    return jax.vmap(jax.vmap(final_R, in_axes=(None, 0)), in_axes=(0, None))(eps_ww, eps_ss)

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
        _, yy = model(params=params, t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(None, 0)), in_axes=(0, None))(p, phi)
    return I_tot_grid # return absolute fraction infected

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid_delayed_ww(model: Callable, base_params: Params, taus, I_crit_list, t1: float = 100.0, E0: float = 1e-6):
    def I_tot(tau, I_crit):
        _, yy = model(params=base_params._replace(tau=tau), t1=t1, E0=E0, I_crit=I_crit, k_I=10000.0)
        return yy[0,0] - yy[-1,0]
    
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(None, 0)), in_axes=(0, None))(taus, I_crit_list)
    _, yy_base = model(params=Params.for_SEIPAR(epsilon_s=0.8), t1=t1, E0=E0, I_crit=1.0, k_I=10000.0)
    return I_tot_grid / (yy_base[0,0] - yy_base[-1,0])

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
    mesh = plt.pcolormesh(EPS_W, EPS_S, R_end_vals, cmap='RdBu_r', norm=mpl.colors.CenteredNorm(vcenter=1.0))
    plt.colorbar(mesh)
    plt.contour(EPS_W, EPS_S, R_end_vals, levels=[1.0], colors='k')
    plt.xlabel('Warning response efficacy')
    plt.ylabel('Isolation efficacy')
    plt.title(title)
    return fig

def plot_I_tot_delayed_ww(model=simulate_SEIPAR_W_with_I_gate, parameters=Params.for_SEIPAR(), title=None, t1=600.0, E0=1e-6):
    taus = jnp.linspace(1.0, 30.0, 100)
    I_crit_list = jnp.logspace(-6, 0, 100)
    TAUS, I_CRIT = jnp.meshgrid(taus, I_crit_list, indexing='ij')

    fig = plt.figure()
    mesh = plt.pcolormesh(TAUS, I_CRIT, compute_I_tot_grid_delayed_ww(model=model, base_params=parameters, taus=taus, I_crit_list=I_crit_list, t1=t1, E0=E0), cmap='viridis', shading='auto')
    fig.colorbar(mesh, label='Total infections (relative to baseline)')
        
    plt.xlabel('Wastewater delay [days]')
    plt.ylabel('Infection threshold')
    plt.yscale('log')
    plt.title(title)
    return fig

def plot_trajectory(
    model: Callable = simulate_SEIPAR_W, 
    params: Params = Params.for_SEIPAR(), 
    path: str = "trajectory.png", 
    title: str = "Trajectory",
    t1: float | int = 600.0, 
    image_resolution: int = 900,
    num_delay_compartments: int = 3,
    plot_S: bool = True,
    plot_E: bool = True,
    plot_Ia: bool = True,
    plot_Ip: bool = True,
    plot_Is: bool = True,
    plot_total_I: bool = False,
    plot_R: bool = True,
    semilogy: bool = False,
) -> None:
    """
    Simulate and plot trajectories.
    Assume compartment order: S, E, [I compartments], R, [Delay compartments].
    """
    
    # run the model
    tt, yy = model(params=params, t1=t1)
    compartments = yy.T

    # determine index of R compartment
    R_idx = -(num_delay_compartments + 1) if num_delay_compartments > 0 else -1

    # extract I compartments
    I_compartments = compartments[slice(2, R_idx) if R_idx != -1 else slice(2, None)]
    total_I = np.sum(I_compartments, axis=0)
    
    # Plot
    fig = plt.figure(figsize=(6, 6))
    
    if plot_S: plt.plot(tt, compartments[0], label='$S$')
    if plot_E: plt.plot(tt, compartments[1], label='$E$')
    if plot_Is and len(I_compartments) > 0: plt.plot(tt, I_compartments[-1], label='$I_s$')
    if plot_Ia and len(I_compartments) > 1: plt.plot(tt, I_compartments[0], label='$I_a$')
    if plot_Ip and len(I_compartments) > 2: plt.plot(tt, I_compartments[1], label='$I_p$')
    if plot_total_I: plt.plot(tt, total_I, label='$I_{total}$')
    if plot_R: plt.plot(tt, compartments[R_idx], label='$R$')

    plt.title(title)
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    if semilogy: plt.semilogy()
    plt.legend(loc='best')
    plt.tight_layout()
    
    fig.savefig(path, dpi=image_resolution)
    plt.close(fig)
