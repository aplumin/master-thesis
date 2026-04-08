import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Optional
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from models.parameters import Params, update_asymptomatic_params, update_epsilons
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.prcc import calculate_prcc
from models.plotting import (
    plot_final_R, plot_I_tot,
    run_gillespie_SEIPAR_W,
    plot_I_tot_delayed_ww, 
    plot_trajectory,
    plot_asymptomatic_effect_for_range_of_intervention_efficacies
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
        )

rule plot_prcc:
    output:
        plot="results/compartmental/prcc_{outcome}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        # TODO: create nice plotting function
        parameters = ['R_0', 'phi', 'gamma_inv', 'sigma_inv', 'mu_a_inv', 'mu_s_inv', 'p', 'epsilon_s', 'epsilon_w', 'tau']
        fig, ax = plt.subplots(figsize=(10,6))
        y_pos = np.arange(len(parameters))
        bars = ax.barh(y_pos, calculate_prcc(params=Params.for_SEIPAR(), t1=300.0 if wildcards.outcome=='Itot' else 50.0, E0=1e-6, total_infected=wildcards.outcome=='Itot'), align='center')
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
        )
