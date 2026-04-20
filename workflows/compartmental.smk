import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Optional
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

from models.parameters import Params, update_asymptomatic_params, update_epsilons, logistic_response_function
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.prcc import calculate_prcc
from models.plotting import (
    plot_heatmap, plot_trajectory,
    plot_final_R, plot_I_tot, plot_I_tot_delayed_ww, plot_asymptomatic_effect_for_range_of_intervention_efficacies,
    run_gillespie_SEIPAR_W, run_gillespie_SEIAR_W, run_gillespie_SEIR_W, 
)

parameters = {
    "SARS-CoV-2": Params.for_SEIPAR(),
    "Influenza A": Params.for_SEIAR(),
    "Ebola": Params.for_SEIR(),
}
models = {
    "SARS-CoV-2": simulate_SEIPAR_W,
    "Influenza A": simulate_SEIAR_W,
    "Ebola": simulate_SEIR_W,
}
gillespie_models = {
    "SARS-CoV-2": run_gillespie_SEIPAR_W,
    "Influenza A": run_gillespie_SEIAR_W,
    "Ebola": run_gillespie_SEIR_W,
}
Rt_times = {
    "SARS-CoV-2": 50.0,
    "Influenza A": 100.0,
    "Ebola": 100.0,
}
pathogens = list(parameters.keys())
E0 = 1e-6

gillespie_popsizes = [100, 1_000_000]
gillespie_num_simulations = [100]

image_resolution = 300
outdir = "results"


rule plot_efficacy_grid_Rt_final:
    output:
        plot="{outdir}/compartmental/efficacy_grid_Rt_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig = plot_final_R(model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], t1=Rt_times[wildcards.pathogen], E0=E0, title=f"Reproductive number after interventions: {wildcards.pathogen}")
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_efficacy_grid_Itot_final:
    output:
        plot="{outdir}/compartmental/efficacy_grid_Itot_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig = plot_I_tot(model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], t1=600.0, E0=E0, title=f"Total number infected: {wildcards.pathogen}")
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_asymptomatic_grid_Rt_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Rt_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(
            model=models[wildcards.pathogen],
            params=parameters[wildcards.pathogen],
            total_infected=False,
            path=output.plot,
            image_resolution=image_resolution,
            t1=Rt_times[wildcards.pathogen],
        )

rule plot_asymptomatic_grid_Itot_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Itot_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(
            model=models[wildcards.pathogen],
            params=parameters[wildcards.pathogen],
            total_infected=True,
            path=output.plot,
            image_resolution=image_resolution,
            t1=Rt_times[wildcards.pathogen],
        )

rule plot_prcc:
    output:
        plot="{outdir}/compartmental/prcc_{outcome}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        parameters = ['$R_0$', '$\\varphi$', '$1/\\gamma$', '$1/\\sigma$', '$1/\\mu_a$', '$1/\\mu_s$', '$p$', '$\\varepsilon_s$', '$\\varepsilon_w$', '$\\tau_W$', '$\\tau_B$', '$k$', '$k_I$']
        fig, ax = plt.subplots(figsize=(10,6))
        y_pos = np.arange(len(parameters))
        # TODO: oscillations for high k, so need some averaged outcome values
        bars = ax.barh(y_pos, calculate_prcc(params=Params.for_SEIPAR(k=1.0), t1=300.0 if wildcards.outcome=='Itot' else 50.0, E0=1e-6, total_infected=wildcards.outcome=='Itot'), align='center')
        ax.axvline(x=0, color='k')
        ax.set_yticks(y_pos); ax.set_yticklabels(parameters); ax.invert_yaxis()
        ax.set_xlabel('PRCC')
        for bar in bars:
            width = bar.get_width()
            ax.text(width+0.02 if width > 0 else width-0.02, bar.get_y()+bar.get_height()/2, f'{width:.3f}', va='center', ha='left' if width > 0 else 'right')
        plt.tight_layout()
        plt.savefig(output.plot, dpi=image_resolution); plt.close(fig)

# TODO: overlay deterministic trajectory
# TODO: out of memory for SC2 1m
rule gillespie:
    output:
        traj="{outdir}/gillespie/gillespie_traj_{pathogen}_{N}.png",
        hist="{outdir}/gillespie/gillespie_hist_{pathogen}_{N}.png",
    run:
        os.makedirs(os.path.dirname(output.traj), exist_ok=True); os.makedirs(os.path.dirname(output.hist), exist_ok=True)
        traj, hist = gillespie_models[wildcards.pathogen](params=parameters[wildcards.pathogen], N=int(wildcards.N), t1=1000.0)
        traj.savefig(output.traj, dpi=image_resolution); plt.close(traj)
        hist.savefig(output.hist, dpi=image_resolution); plt.close(hist)

rule plot_trajectory:
    output:
        plot="{outdir}/compartmental/trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        params = update_epsilons(parameters[wildcards.pathogen], epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w))
        plot_trajectory(
            model=models[wildcards.pathogen],
            params=params,
            path=output.plot,
            title=f"Trajectory: {wildcards.pathogen} (eps_s={wildcards.epsilon_s}, eps_w={wildcards.epsilon_w})",
            t1=600.0,
            image_resolution=image_resolution,
            plot_total_I=True,
            semilogy=True
        )

rule plot_trajectory_delayed_ww_intervention:
    output:
        plot="{outdir}/compartmental/delayed_ww_intervention_trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}_Icrit{I_crit}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        params = update_epsilons(
            parameters[wildcards.pathogen]._replace(I_crit=float(wildcards.I_crit)), 
            epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w)
        )
        plot_trajectory(
            model=models[wildcards.pathogen],
            params=params,
            path=output.plot,
            title=f"Trajectory: {wildcards.pathogen} (eps_s={wildcards.epsilon_s}, eps_w={wildcards.epsilon_w})",
            t1=600.0,
            image_resolution=image_resolution,
            plot_total_I=True,
            semilogy=True
        )

# TODO: currently vary tau_W delay, tau_B is default
rule delayed_ww_intervention:
    output:
        plot="{outdir}/compartmental/delay_grid_ww_intervention_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        base_parameters = Params.for_SEIPAR(epsilon_s=0.8, epsilon_w=0.8)
        fig = plot_I_tot_delayed_ww(model=simulate_SEIPAR_W, parameters=base_parameters)
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule baseline_trajectories:
    output:
        plot="{outdir}/compartmental/baseline_trajectories_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_trajectory(
            model = models[wildcards.pathogen],
            params = parameters[wildcards.pathogen],
            path = output.plot,
            title = f"Baseline trajectories for {wildcards.pathogen}",
            image_resolution = image_resolution
        )

rule baseline_trajectories_no_asymptomatic:
    output:
        plot="{outdir}/compartmental/trajectories_no_asymptomatic_{pathogen}.png"
    run:    
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_trajectory(
            model = models[wildcards.pathogen],
            params = {
                "SARS-CoV-2": Params.for_SEIPAR(p=0.0, phi=0.0),
                "Influenza A": Params.for_SEIAR(p=0.0, phi=0.0),
                "Ebola": Params.for_SEIR(),
            }[wildcards.pathogen],
            path = output.plot,
            title = f"No asymptomatic transmission for {wildcards.pathogen}",
            image_resolution = image_resolution
        )

rule plot_response_function:
    output:
        plot="{outdir}/compartmental/response_function.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        # vectorise response function over Rt and Is
        parameters = Params.for_SEIPAR(epsilon_w=0.8, epsilon_s=0.8, I_crit=1e-3, k=100)
        Rt_vals = jnp.linspace(0.0, 3.0, 200)
        Is_vals = jnp.linspace(0.0, 0.01, 200)
        def _response(r, i): return logistic_response_function(r, parameters, i)
        Z = jax.vmap(jax.vmap(_response, in_axes=(0, None)), in_axes=(None, 0))(Rt_vals, Is_vals)
        
        # layout
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[4, 1, 0.2], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1,0])
        ax_top = fig.add_subplot(gs[0,0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1,1], sharey=ax_main)
        ax_cbar = fig.add_subplot(gs[1,2])
        
        # heatmap
        mesh = ax_main.pcolormesh(Rt_vals, Is_vals, Z, cmap='magma', shading='auto')
        ax_main.contour(Rt_vals, Is_vals, Z, levels=10, colors='white', alpha=0.3)
        ax_main.axvline(parameters.R_crit, color='red', linestyle='--', alpha=0.8, label=f'$R_{{crit}}={parameters.R_crit}$')
        if parameters.I_crit > 0.0:
            ax_main.axhline(parameters.I_crit, color='orange', linestyle='--', alpha=0.8, label=f'$I_{{crit}}={parameters.I_crit}$')
        # ax_main.legend()
        ax_main.set_xlabel('Delayed wastewater signal ($R_t$)', fontsize=14)
        ax_main.set_ylabel('Symptomatic population ($I_s$)', fontsize=14)
        
        # top marginal
        ax_top.plot(Rt_vals, Z[-1,:], color='black', lw=2)
        ax_top.axvline(parameters.R_crit, color='red', linestyle='--')
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(True, alpha=0.2)

        # right marginal
        ax_right.plot(Z[:,-1], Is_vals, color='black', lw=2)
        if parameters.I_crit > 0.0:
            ax_right.axhline(parameters.I_crit, color='orange', linestyle='--')
        ax_right.tick_params(labelleft=False)
        ax_right.grid(True, alpha=0.2)
        
        # colorbar
        cbar = fig.colorbar(mesh, cax=ax_cbar)
        cbar.set_label('Logistic response function', fontsize=14, labelpad=10)
        
        # save and close
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight')
        plt.close(fig)


@partial(jax.jit, static_argnames=['model', 't1'])
def _compute_delay_metrics_grid(model, base_params, taus_W, taus_B, t1=300.0):
    def metrics(tau_W, tau_B):
        params = base_params._replace(tau_W=tau_W, tau_B=tau_B)
        _, yy = model(params=params, t1=t1)
        rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
        rt_reported = yy[:, -(params.n_B + 1)]
        steady = rt_true[-rt_true.shape[0]//3:] # last third
        amplitude = jnp.max(steady) - jnp.min(steady)
        frac_infected = yy[0,0] - yy[-1,0]
        above = (rt_reported >= params.R_crit).astype(jnp.int32)
        total_time_above = above.mean() * t1
        num_crossings = jnp.sum(jnp.diff(above) > 0)
        return amplitude, frac_infected, total_time_above, num_crossings
    return jax.vmap(jax.vmap(metrics, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

# TODO: response stays < 1: check logic
rule plot_true_vs_reported_Rt_scenarios:
    output:
        plot="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_scenarios.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)

        taus_W = [3.0, 7.0, 14.0, 21.0]
        taus_B = [1.0, 3.0, 7.0, 14.0]
        k = float(wildcards.k)
        t1 = 300.0

        epsilon_s = 0.8 if wildcards.pathogen == "SARS-CoV-2" else 0.4
        base_params = update_epsilons(parameters[wildcards.pathogen], epsilon_s=epsilon_s, epsilon_w=0.8)._replace(k=k)
        model = models[wildcards.pathogen]

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=len(taus_W), ncols=len(taus_B), figsize=(12, 12), sharex=True, sharey=True)

        for i, tau_W in enumerate(taus_W):
            for j, tau_B in enumerate(taus_B):
                params = base_params._replace(tau_W=tau_W, tau_B=tau_B)
                tt, yy = model(params=params, t1=t1)
                rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
                rt_reported = yy[:, -(params.n_B + 1)]
                above = (rt_reported >= params.R_crit).astype(jnp.float32)
                total_time_above = float(above.mean() * t1)
                num_crossings = int(jnp.sum(jnp.diff(above) > 0))

                ax = axs[i,j]
                if j == 0: ax.set_ylabel(f'$\\tau_W={tau_W}$', fontsize=16)
                if i == 0: ax.set_title(f'$\\tau_B={tau_B}$', fontsize=16)
                ax.plot(tt, rt_true, color='black')
                ax.plot(tt, rt_reported, color='red')
                ax.axhline(params.R_crit, color='grey', linestyle='--')
                ax.text(0.97, 0.95, f'{total_time_above:.0f} days above $R_{{crit}}$\n{num_crossings} warnings', transform=ax.transAxes, ha='right', va='top', fontsize=8)

        fig.suptitle(f'{wildcards.pathogen}: $k={k:g}$', fontsize=18, y=0.995)
        fig.legend(
            [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='red', lw=2)],
            ['True $R_t$', 'Reported $R_t$'],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=16,
        )
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close()

rule plot_true_vs_reported_Rt_heatmaps:
    output:
        amplitudes ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_amplitudes.png",
        itot ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_itot.png",
        time_above ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_time_above.png",
        crossings ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_crossings.png",
    run:
        for path in output: os.makedirs(os.path.dirname(path), exist_ok=True)

        taus_W = jnp.linspace(1.0, 31.0, num=100)
        taus_B = jnp.linspace(1.0, 31.0, num=100)
        k = float(wildcards.k)
        base_params = update_epsilons(parameters[wildcards.pathogen], epsilon_s=0.8, epsilon_w=0.8)._replace(k=k)

        amplitudes, fractions, time_above, crossings = _compute_delay_metrics_grid(model=models[wildcards.pathogen], base_params=base_params, taus_W=taus_W, taus_B=taus_B)

        kwargs = dict(x_logscale=False, xlabel='Behavioural delay ($\\tau_B$)', ylabel='Reporting delay ($\\tau_W$)')
        scenario = f'({wildcards.pathogen}, $k={k:g}$)'
        fig, _ = plot_heatmap(taus_B, taus_W, amplitudes, cmap='magma', cbar_label='Amplitude of oscillations', title=f'Stability of delayed response {scenario}', **kwargs)
        fig.savefig(output.amplitudes, dpi=image_resolution); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, fractions, cbar_label='Fraction infected', title=f'Effect of delay on infections {scenario}', **kwargs)
        fig.savefig(output.itot, dpi=image_resolution); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, time_above, cmap='cividis', cbar_label='Days above warning threshold', title=f'Time above warning threshold {scenario}', **kwargs)
        fig.savefig(output.time_above, dpi=image_resolution); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, crossings, cmap='plasma', cbar_label='Total times warned', title=f'Number of warning-threshold crossings {scenario}', **kwargs)
        fig.savefig(output.crossings, dpi=image_resolution); plt.close(fig)


rule all:
    input:
        expand(
            rules.plot_efficacy_grid_Rt_final.output.plot, 
            pathogen=pathogens, outdir=outdir
        ),
        expand(
            rules.plot_efficacy_grid_Itot_final.output.plot, 
            pathogen=pathogens, outdir=outdir
        ),
        expand(
            rules.plot_asymptomatic_grid_Rt_final.output.plot, 
            outdir=outdir, pathogen=["SARS-CoV-2", "Influenza A"], # only pathogens with asymptomatic transmission
        ),
        expand(
            rules.plot_asymptomatic_grid_Itot_final.output.plot, 
            outdir=outdir, pathogen=["SARS-CoV-2", "Influenza A"],
        ),
        expand(
            rules.plot_prcc.output.plot, 
            outdir=outdir, outcome=['Itot', 'Rt']  # TODO: generalise to other pathogens
        ),
        expand(
            rules.gillespie.output, 
            outdir=outdir, pathogen=pathogens,
            N=[100], # N=gillespie_popsizes,
        ),
        expand(
            rules.plot_trajectory.output.plot, 
            outdir=outdir, pathogen=pathogens,
            epsilon_s=[0.0, 0.4, 0.8], epsilon_w=[0.0, 0.4, 0.8],
        ),
        expand(
            rules.plot_trajectory_delayed_ww_intervention.output.plot,
             outdir=outdir, pathogen=["SARS-CoV-2", "Influenza A"],
            epsilon_s=[0.8], epsilon_w=[0.8], 
            I_crit=[1e-5, 1e-4, 1e-3, 1e-2],
        ),
        expand(
            rules.delayed_ww_intervention.output.plot, 
            outdir=outdir, pathogen=["SARS-CoV-2"], # TODO: change main rule for generalisation
        ),
        expand(
            rules.baseline_trajectories.output.plot, 
            pathogen=pathogens, outdir=outdir, 
        ),
        expand(
            rules.baseline_trajectories_no_asymptomatic.output.plot, outdir=outdir,
            pathogen=pathogens
        ),
        expand(
            rules.plot_true_vs_reported_Rt_scenarios.output, 
            pathogen=pathogens, outdir=outdir,
            k=[1, 3, 10, 30],
        ),
        expand(
            rules.plot_true_vs_reported_Rt_heatmaps.output, 
            pathogen=pathogens, outdir=outdir,
            k=[1, 3, 10, 30],
        ),
        expand(rules.plot_response_function.output.plot, outdir=outdir),
