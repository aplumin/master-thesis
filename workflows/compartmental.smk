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
    plot_heatmap,
    plot_final_R, plot_I_tot, plot_I_tot_delayed_ww, 
    plot_trajectory,
    plot_asymptomatic_effect_for_range_of_intervention_efficacies,
    run_gillespie_SEIPAR_W,
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


rule plot_efficacy_grid_Rt_final:
    output:
        plot="results/compartmental/efficacy_grid_Rt_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig = plot_final_R(model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], t1=Rt_times[wildcards.pathogen], E0=E0, title=f"Reproductive number after interventions: {wildcards.pathogen}")
        fig.savefig(output.plot, dpi=image_resolution)
        plt.close(fig)

rule plot_efficacy_grid_Itot_final:
    output:
        plot="results/compartmental/efficacy_grid_Itot_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig = plot_I_tot(model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], t1=600.0, E0=E0, title=f"Total number infected: {wildcards.pathogen}")
        fig.savefig(output.plot, dpi=image_resolution)
        plt.close(fig)

rule plot_asymptomatic_grid_Rt_final:
    output:
        plot="results/compartmental/asymptomatic_grid_Rt_final_{pathogen}.png"
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
        plot="results/compartmental/asymptomatic_grid_Itot_final_{pathogen}.png"
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
        plot="results/compartmental/prcc_{outcome}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        parameters = ['$R_0$', '$\\varphi$', '$1/\\gamma$', '$1/\\sigma$', '$1/\\mu_a$', '$1/\\mu_s$', '$p$', '$\\varepsilon_s$', '$\\varepsilon_w$', '$\\tau$']#, '$k$', '$k_I$']
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
        plt.savefig(output.plot, dpi=image_resolution)
        plt.close(fig)

rule gillespie:
    output:
        traj="results/gillespie/gillespie_traj_{pathogen}_{N}.png",
        hist="results/gillespie/gillespie_hist_{pathogen}_{N}.png",
    run:
        os.makedirs(os.path.dirname(output.traj), exist_ok=True); os.makedirs(os.path.dirname(output.hist), exist_ok=True)
        traj, hist = run_gillespie_SEIPAR_W(
            params=parameters[wildcards.pathogen], 
            N=int(wildcards.N), 
            t1=1000.0, 
        )
        traj.savefig(output.traj, dpi=image_resolution); plt.close(traj)
        hist.savefig(output.hist, dpi=image_resolution); plt.close(hist)

rule plot_trajectory:
    output:
        plot="results/compartmental/trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}.png"
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
            num_delay_compartments=params.num_delay_compartments,
            plot_total_I=True,
            semilogy=True
        )

rule plot_trajectory_delayed_ww_intervention:
    output:
        plot="results/compartmental/delayed_ww_intervention_trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}_Icrit{I_crit}.png"
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
            num_delay_compartments=params.num_delay_compartments,
            plot_total_I=True,
            semilogy=True
        )

rule delayed_ww_intervention:
    output:
        plot="results/compartmental/delay_grid_ww_intervention_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        base_parameters = Params.for_SEIPAR(epsilon_s=0.8, epsilon_w=0.8)
        fig = plot_I_tot_delayed_ww(model=simulate_SEIPAR_W, parameters=base_parameters)
        fig.savefig(output.plot, dpi=image_resolution)
        plt.close(fig)

rule baseline_trajectories:
    output:
        plot="results/compartmental/baseline_trajectories_{pathogen}.png"
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
        plot="results/compartmental/trajectories_no_asymptomatic_{pathogen}.png"
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

rule plot_true_vs_reported_Rt:
    output:
        scenarios="results/compartmental/true_vs_reported_Rt_{pathogen}_scenarios.png",
        heatmap="results/compartmental/true_vs_reported_Rt_{pathogen}_heatmap.png",
    run:
        os.makedirs(os.path.dirname(output.scenarios), exist_ok=True)
        
        taus = [3.0, 7.0, 14.0, 21.0]
        ks = [1.0, 10.0, 20.0, 50.0]

        # SCENARIOS
        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=len(taus), ncols=len(ks), figsize=(12, 12), sharex=True, sharey=True)

        for row_idx, tau in enumerate(taus):
            for col_idx, k in enumerate(ks):
                # simulate
                params = Params.for_SEIPAR(epsilon_w=0.8, epsilon_s=0.8, tau=float(tau), k=k)
                tt, yy = simulate_SEIPAR_W(params)
                compartments = yy.T

                # set up plot
                ax = axs[row_idx, col_idx]
                if row_idx == 0: ax.set_title(f'$k={k}$', fontsize=16)
                if col_idx == 0: ax.set_ylabel(f'$\\tau={tau}$', fontsize=16)

                # true Rt
                f_vals = logistic_response_function(compartments[-1], params, compartments[-(params.num_delay_compartments+2)])
                rt_true_vals = params.R_0 * params.rho * f_vals * compartments[0]
                ax.plot(tt, rt_true_vals, color='black')
                
                # reported Rt
                rt_reported = compartments[-1]
                ax.plot(tt, rt_reported, color='red')
                ax.axhline(params.R_crit, color='grey', linestyle='--')

        fig.legend(
            [Line2D([0],[0], color='black', lw=2), Line2D([0],[0], color='red', lw=2)], 
            ['True $R_t$', 'Reported $R_t$'],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=16
        )

        plt.tight_layout()
        plt.savefig(output.scenarios, dpi=image_resolution, bbox_inches='tight')
        plt.close()

        # HEATMAP
        taus = np.linspace(1.0, 31.0, num=100)
        ks = np.logspace(0, 3, num=100)
        amplitudes = np.zeros((100,100))

        for i, tau in enumerate(taus):
            for j, k in enumerate(ks):
                params = Params.for_SEIPAR(epsilon_w=0.8, epsilon_s=0.8, tau=float(tau), k=float(k))
                _, yy = simulate_SEIPAR_W(params)
                compartments = yy.T
                f_vals = logistic_response_function(compartments[-1], params, compartments[-(params.num_delay_compartments+2)])
                rt_true_vals = params.R_0 * params.rho * f_vals * compartments[0]

                # take last third: steady-state
                steady_state = rt_true_vals[len(rt_true_vals) * 2 // 3:]
                amplitudes[i,j] = float(np.max(steady_state) - np.min(steady_state))

        fig, ax = plot_heatmap(
            ks, taus, amplitudes, 
            cmap = 'Greys', cbar_label = 'Amplitude of $R_t$ oscillations',
            title = 'Stability of delayed response',
            x_logscale = True, xlabel = 'Response strength ($k$)',
            ylabel = 'Delay ($\\tau$)'
        )
        plt.savefig(output.heatmap, dpi=image_resolution)
        plt.close()

rule plot_response_function:
    output:
        plot="results/compartmental/response_function.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        # vectorise response function over Rt and Is
        parameters = Params.for_SEIPAR(epsilon_w=0.8, epsilon_s=0.8, I_crit=1e-3)
        Rt_vals = jnp.linspace(0.0, 3.0, 200)
        Is_vals = jnp.linspace(0.0, 0.01, 200)
        def _response(r, i): return logistic_response_function(r, parameters, i)
        Z = jax.vmap(jax.vmap(_response, in_axes=(None, 0)), in_axes=(0, None))(Rt_vals, Is_vals).T
        
        # layout
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[4, 1, 0.2], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1,0])
        ax_top = fig.add_subplot(gs[0,0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1,1], sharey=ax_main)
        ax_cbar = fig.add_subplot(gs[1,2])
        
        # heatmap
        mesh = ax_main.pcolormesh(Rt_vals, Is_vals, Z, cmap='Greys_r', shading='auto')
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


rule all:
    input:
        expand(rules.plot_efficacy_grid_Rt_final.output.plot, pathogen=pathogens),
        expand(rules.plot_efficacy_grid_Itot_final.output.plot, pathogen=pathogens),
        expand(
            rules.plot_asymptomatic_grid_Rt_final.output.plot, 
            pathogen=["SARS-CoV-2", "Influenza A"], # only pathogens with asymptomatic transmission
            epsilon_s=[0.0, 0.4, 0.8],
            epsilon_w=[0.0, 0.4, 0.8],
        ),
        expand(
            rules.plot_asymptomatic_grid_Itot_final.output.plot, 
            pathogen=["SARS-CoV-2", "Influenza A"],
            epsilon_s=[0.0, 0.4, 0.8],
            epsilon_w=[0.0, 0.4, 0.8],
        ),
        expand(rules.plot_prcc.output.plot, outcome=['Itot', 'Rt']),
        # expand(
        #     rules.gillespie.output,
        #     pathogen=["SARS-CoV-2", "Influenza A"],
        #     N=gillespie_popsizes,
        # ),
        expand(
            rules.plot_trajectory.output.plot, 
            pathogen=pathogens,
            epsilon_s=[0.0, 0.4, 0.8],
            epsilon_w=[0.0, 0.4, 0.8],
        ),
        expand(
            rules.plot_trajectory_delayed_ww_intervention.output.plot, 
            pathogen=["SARS-CoV-2", "Influenza A"],
            epsilon_s=[0.8],
            epsilon_w=[0.8],
            I_crit=[1e-5, 1e-4, 1e-3, 1e-2],
        ),
        expand(
            rules.delayed_ww_intervention.output.plot, 
            pathogen=["SARS-CoV-2", "Influenza A"],
        ),
        expand(
            rules.baseline_trajectories.output.plot, 
            pathogen=pathogens
        ),
        expand(
            rules.baseline_trajectories_no_asymptomatic.output.plot, 
            pathogen=pathogens
        ),
        expand(
            rules.plot_true_vs_reported_Rt.output, 
            pathogen=pathogens,
            tau=[1.0, 7.0, 14.0, 21.0, 28.0],
        ),
        rules.plot_response_function.output.plot,
