"""
Plotting functions.
"""

import jax.numpy as jnp
import numpy as np
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.parameters import Params, update_epsilons
from models.compartmental import simulate_SEIPAR_W
from models.gillespie import gillespie_SEIPAR_W
from models.scenarios import (
    compute_I_tot_grid, compute_R_grid, 
    compute_asymptomatic_grid_Rt, compute_asymptomatic_grid_Itot, 
    compute_I_tot_grid_delayed_ww,
)


def plot_heatmap(
    X, Y, Z, 
    cmap='viridis', shading='auto', norm=None,
    contour_levels=[], contour_colors='black', contour_linestyles=['-'],
    title=None, title_fontsize=18, title_pad=10,
    xlabel=None, ylabel=None,
    xlabelsize=14, ylabelsize=14,
    x_logscale=False, y_logscale=False, 
    cbar_shrink=0.8, cbar_aspect=30, cbar_label=None, cbar_labelsize=14, cbar_labelpad=10,
    cbar_axhlines=[], cbar_axhlines_colors=[], cbar_axhlines_linestyles=[],
):
    """General heatmap plotting function."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_box_aspect(1)

    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, shading=shading, norm=norm)
    
    if contour_levels:
        ax.contour(X, Y, Z, levels=contour_levels, colors=contour_colors, linestyles=contour_linestyles)
    
    cbar = fig.colorbar(mesh, ax=ax, shrink=cbar_shrink, aspect=cbar_aspect)
    cbar.set_label(cbar_label, fontsize=cbar_labelsize, labelpad=cbar_labelpad)
    for i, hline in enumerate(cbar_axhlines):
        cbar.ax.axhline(hline, color=cbar_axhlines_colors[i], linestyle=cbar_axhlines_linestyles[i])
    
    if x_logscale: 
        ax.set_xscale('log')
    if y_logscale: 
        ax.set_yscale('log')
    
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel(xlabel, fontsize=xlabelsize)
    ax.set_ylabel(ylabel, fontsize=ylabelsize)
    
    plt.tight_layout()
    return fig, ax


def plot_I_tot(model=simulate_SEIPAR_W, params=Params.for_SEIPAR(), title=None, t1=600.0, E0=1e-6):
    """
    Plot a grid of the total proportion infected after interventions (compared to baseline without interventions).
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    eps_ww = jnp.linspace(0.0, 0.999, 100)
    eps_ss = jnp.linspace(0.0, 0.999, 100)
    fig, _ = plot_heatmap(
        eps_ww, eps_ss, compute_I_tot_grid(model, params, eps_ww, eps_ss, t1, E0), 
        title=title, xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
    )
    return fig

def plot_final_R(model=simulate_SEIPAR_W, params=Params.for_SEIPAR(), t1=100.0, E0=1e-6, title=None):
    """
    Plot a grid of the reproductive number after interventions.
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    eps_ww = jnp.linspace(0.0, 0.999, 100)
    eps_ss = jnp.linspace(0.0, 0.999, 100)
    fig, _ = plot_heatmap(
        eps_ww, eps_ss, compute_R_grid(model, params, eps_ww, eps_ss, t1, E0), 
        cmap='RdBu_r', norm=mpl.colors.CenteredNorm(vcenter=1.0),
        contour_levels=[1.0],
        title=title, xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
    )
    return fig

def plot_I_tot_delayed_ww(model=simulate_SEIPAR_W, parameters=Params.for_SEIPAR(), title=None, t1=600.0, E0=1e-6):
    taus = jnp.linspace(1.0, 30.0, 100)
    I_crit_list = jnp.logspace(-6, 0, 100)
    fig, _ = plot_heatmap(
        taus, I_crit_list, compute_I_tot_grid_delayed_ww(model=model, base_params=parameters, taus=taus, I_crit_list=I_crit_list, t1=t1, E0=E0),
        contour_levels=[0.25, 0.5, 0.75], contour_colors='red', contour_linestyles=['--', '-', '--'],
        cbar_axhlines=[0.25, 0.5, 0.75], cbar_axhlines_colors=['red', 'red', 'red'], cbar_axhlines_linestyles=['--', '-', '--'],
        title=title, xlabel='Wastewater delay $\\tau$', ylabel='Infection threshold',
        y_logscale=True, cbar_label='Total infections (relative to baseline)',
    )
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


def plot_asymptomatic_effect_for_range_of_intervention_efficacies(
    model: Callable = simulate_SEIPAR_W, 
    params: Params = Params.for_SEIPAR(),
    total_infected: bool = False,
    ps = jnp.linspace(0.0, 0.999, 100),
    phis = jnp.linspace(0.0, 0.999, 100),
    epsilon_s = [0.0, 0.4, 0.8],
    epsilon_w = [0.0, 0.4, 0.8],
    E0: float = 1e-6,
    t1: float = None, 
    image_resolution: int = 900,
    path: str = "asymptomatic_effect.png",
) -> None:
    
    # end time
    if t1 is None: t1 = 600.0 if total_infected else 50.0

    # build dataframe
    p_grid, phi_grid = jnp.meshgrid(ps, phis, indexing="ij")
    df_list = []
    for eps_s in epsilon_s:
        for eps_w in epsilon_w:
            base_params = update_epsilons(params=params, epsilon_s=float(eps_s), epsilon_w=float(eps_w))
            if total_infected:
                Z = compute_asymptomatic_grid_Itot(model=model, base_params=base_params, p=ps, phi=phis, t1=t1, E0=E0)
            else:
                Z = compute_asymptomatic_grid_Rt(model=model, base_params=base_params, p=ps, phi=phis, t1=t1, E0=E0)
            df_list.append(pd.DataFrame({'p': np.array(p_grid.flatten()), 'phi': np.array(phi_grid.flatten()), 'Z': np.array(Z.flatten()), 'eps_s': eps_s, 'eps_w': eps_w}))
    df = pd.concat(df_list, ignore_index=True)

    # color scaling
    global_min, global_max = df['Z'].min(), df['Z'].max()
    cmap = 'viridis' if total_infected else 'RdBu_r'
    if total_infected:
        plot_kwargs = {'cmap': cmap, 'vmin': global_min, 'vmax': global_max, 'shading': 'auto'}
    else:
        max_dev = max(abs(global_min - 1.0), abs(global_max - 1.0))
        plot_kwargs = {'cmap': cmap, 'vmin': 1.0 - max_dev, 'vmax': 1.0 + max_dev, 'shading': 'auto'}

    # FacetGrid
    sns.set_theme(style="white", rc={"axes.grid": False})
    g = sns.FacetGrid(df, row="eps_s", col="eps_w", height=3, aspect=1)

    # mapping function
    def _meshmap(data, **kwargs):
        ax = plt.gca()
        Z_matrix = data.pivot(index='p', columns='phi', values='Z').values
        mesh = ax.pcolormesh(p_grid, phi_grid, Z_matrix, linewidth=0, edgecolors='none', rasterized=True, **kwargs)
        if not total_infected: # R=1 contour
            ax.contour(p_grid, phi_grid, Z_matrix, levels=[1.0], colors='black', linewidths=1.5, linestyles='dashed')
        return mesh
    g.map_dataframe(_meshmap, **plot_kwargs)

    # labels and title
    g.set(xlabel=None, ylabel=None, aspect='equal')
    g.figure.supxlabel("Proportion asymptomatic", fontsize=14, y=0.02)
    g.figure.supylabel("Relative infectiousness", fontsize=14, x=-0.04)

    # colorbar
    mesh = g.axes[-1, -1].collections[0] 
    cbar = g.figure.colorbar(mesh, ax=g.axes.ravel().tolist(), shrink=0.8, aspect=30)
    if not total_infected: cbar.ax.axhline(1.0, color='black', linewidth=1.5)

    # save and close
    plt.savefig(path, dpi=image_resolution, bbox_inches='tight')
    plt.close(g.figure)


def run_gillespie_SEIPAR_W(params: Params = Params.for_SEIPAR(), N: int = 1000, t1: int = 100.0, num_simulations: int = 1000, seed: int = 0):
    """Return two plots: 1. trajectories, 2. histogram of times until extinction."""
    np.random.seed(seed)
    times_list = np.zeros(num_simulations)

    fig_traj, ax_traj = plt.subplots()
    for i in range(num_simulations):
        times, history = gillespie_SEIPAR_W(params=params, N=N, t1=t1)
        times_list[i] = times[-1]
        ax_traj.plot(times, history[:,0], alpha=0.5)
        ax_traj.scatter(times[-1], history[-1,0], marker='X', alpha=0.5)
    
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(times_list, density=True)
    
    return fig_traj, fig_hist
