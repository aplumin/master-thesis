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
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

from models.parameters import Params, logistic_response_function
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.scenarios import compute_R_grid
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

p_CI = {"SARS-CoV-2": (0.23, 0.399), "Influenza A": (None, None)}
phi_CI = {"SARS-CoV-2": (0.07, 0.28), "Influenza A": (None, None)}

rule plot_asymptomatic_grid_Rt_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Rt_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(
            model=models[wildcards.pathogen],
            params=parameters[wildcards.pathogen],
            p_CI=p_CI[wildcards.pathogen],
            phi_CI=phi_CI[wildcards.pathogen],
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
            p_CI=p_CI[wildcards.pathogen],
            phi_CI=phi_CI[wildcards.pathogen],
            total_infected=True,
            path=output.plot,
            image_resolution=image_resolution,
            t1=600.0,
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
        params = parameters[wildcards.pathogen].update(epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w))
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
        params = parameters[wildcards.pathogen].update(I_crit=float(wildcards.I_crit), epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w))
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
        peak_Is, time_to_peak, total_time = plot_trajectory(
            model = models[wildcards.pathogen],
            params = parameters[wildcards.pathogen].update(I_crit=1e-4), # TODO: 1 in 10_000 infected
            path = output.plot,
            title = f"{wildcards.pathogen}",
            image_resolution = image_resolution,
            plot_total_I = True,
            t1 = 500.0
        )
        print(wildcards.pathogen)
        print(peak_Is, time_to_peak, total_time)

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
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
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
        base_params = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, k=k)
        model = models[wildcards.pathogen]

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=len(taus_W), ncols=len(taus_B), figsize=(12, 12), sharex=True, sharey=True)

        for i, tau_W in enumerate(taus_W):
            for j, tau_B in enumerate(taus_B):
                params = base_params.update(tau_W=tau_W, tau_B=tau_B)
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
        base_params = parameters[wildcards.pathogen].update(epsilon_s=0.8, epsilon_w=0.8, k=k)

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



@partial(jax.jit, static_argnames=['model', 't1', 'tRt'])
def compute_metrics(model, base_params, eps_ww, eps_ss, t1, tRt, E0):
    def metrics(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        tt, yy = model(params=params, t1=t1, E0=E0)
        Is = yy[:, -(params.n_W + params.n_B + 2)]
        rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
        Rt_final = rt_true[jnp.argmin(jnp.abs(tt - tRt))]
        time_to_below = jnp.where(jnp.any(rt_true < 1.0), tt[jnp.argmax(rt_true < 1.0)], t1)
        Itot = yy[0,0] - yy[-1,0]
        peak_Is = jnp.max(Is)
        return Rt_final, time_to_below, Itot, peak_Is
    return jax.vmap(jax.vmap(metrics, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)
      
rule plot_main_intervention_grid:
    output:
        plot="{outdir}/compartmental/main_intervention_grid.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        t1 = 500.0

        # compute per pathogen
        Rt_g, tRt_g, Itot_g, peakIs_g = {}, {}, {}, {}

        for pathogen in pathogens:
            Rt, tRt, It, pk = compute_metrics(models[pathogen], parameters[pathogen], eps_ww, eps_ss, t1, Rt_times[pathogen], E0)
            Rt_g[pathogen] = np.array(Rt)
            tRt_g[pathogen] = np.array(tRt)
            _, yy0 = models[pathogen](params=parameters[pathogen].update(epsilon_s=0.0, epsilon_w=0.0), t1=t1, E0=E0)
            Itot_g[pathogen] = np.array(It) / float(yy0[0,0] - yy0[-1,0])
            peakIs_g[pathogen] = np.array(pk)

        # plot
        rows = [ # label, data, cmap, center_at_one
            ('$\\mathcal{R}_t$', Rt_g, 'RdBu_r', True),
            ('Time to $\\mathcal{R}_t<1$', tRt_g, 'magma', False),
            ('$I_\\text{tot}$ (relative to baseline)', Itot_g, 'viridis', False),
            ('Peak $I_s$', peakIs_g, 'viridis', False),
        ]
        fig, axs = plt.subplots(nrows=len(rows), ncols=len(pathogens), figsize=(13, 16), sharex=True, sharey=True)
        for row_idx, (label, data, cmap, center_at_one) in enumerate(rows):

            # normalisation
            vals = np.concatenate([d.ravel() for d in data.values()])
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if center_at_one:
                d = max(abs(vmin - 1.0), abs(vmax - 1.0))
                norm = mpl.colors.Normalize(vmin=1.0 - d, vmax=1.0 + d)
            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # meshgrid
            meshes = []
            for col_idx, pathogen in enumerate(pathogens):
                ax = axs[row_idx, col_idx]
                mesh = ax.pcolormesh(np.array(eps_ww), np.array(eps_ss), data[pathogen], cmap=cmap, norm=norm, shading='auto', rasterized=True)
                meshes.append(mesh)
                ax.set_aspect('equal')

                # contours, titles, labels
                if center_at_one: ax.contour(np.array(eps_ww), np.array(eps_ss), data[pathogen], levels=[1.0], colors='black', linewidths=1.0, linestyles='--')
                if row_idx == 0: ax.set_title(pathogen, fontsize=14)
                if row_idx == len(rows) - 1: ax.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=11)
                if col_idx == 0: ax.set_ylabel('Isolation efficacy $\\varepsilon_s$', fontsize=11)

            # colorbar
            cbar = fig.colorbar(meshes[-1], ax=axs[row_idx, :].tolist(), shrink=0.85, aspect=25, pad=0.02)
            cbar.set_label(label, fontsize=12, labelpad=8)
            if center_at_one: cbar.ax.axhline(1.0, color='black', linewidth=1.0)

        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight')
        plt.close(fig)


best_params_kwargs = {
    "SARS-CoV-2": dict(R_0=2.40, gamma_inv=5.86, sigma_inv=0.52, mu_s_inv=10.0, mu_a_inv=4.63, p=0.230, phi=0.07),
    "Influenza A": dict(R_0=1.30, gamma_inv=3.12, mu_s_inv=4.69, mu_a_inv=2.06, p=0.33, phi=0.50),
    "Ebola": dict(R_0=1.74, gamma_inv=10.38, mu_s_inv=6.30)
}
worst_params_kwargs = {
    "SARS-CoV-2": dict(R_0=2.98, gamma_inv=5.06, sigma_inv=3.00, mu_s_inv=7.80, mu_a_inv=5.50, p=0.399, phi=0.28),
    "Influenza A": dict(R_0=1.70, gamma_inv=2.28, mu_s_inv=2.06, mu_a_inv=4.69, p=0.33, phi=0.50),
    "Ebola": dict(R_0=2.15, gamma_inv=8.80, mu_s_inv=3.70)
}
colors = {
    "SARS-CoV-2": "tab:blue", 
    "Influenza A": "tab:orange", 
    "Ebola": "tab:green"
}

rule plot_R_1_contours:
    output:
        plot="{outdir}/compartmental/R_1_contours.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        
        for pathogen in pathogens:
            mean_params = parameters[pathogen]
            best_params = mean_params.update(**best_params_kwargs[pathogen])
            worst_params = mean_params.update(**worst_params_kwargs[pathogen])
            t1 = Rt_times[pathogen]
            model = models[pathogen]
            
            Rt_mean = np.array(compute_R_grid(model=model, base_params=mean_params, eps_ww=eps_ww, eps_ss=eps_ss, t1=t1))
            Rt_best = np.array(compute_R_grid(model=model, base_params=best_params, eps_ww=eps_ww, eps_ss=eps_ss, t1=t1))
            Rt_worst = np.array(compute_R_grid(model=model, base_params=worst_params, eps_ww=eps_ww, eps_ss=eps_ss, t1=t1))

            shading_range = np.zeros_like(Rt_mean)
            shading_range[(Rt_best <= 1.0) & (Rt_worst >= 1.0)] = 1
            color = colors[pathogen]
            ax.contourf(eps_ww, eps_ss, shading_range, levels=[0.5, 1.5], colors=[color], alpha=0.2)
            ax.contour(eps_ww, eps_ss, Rt_mean, levels=[1.0], colors=[color], linewidths=2)
            ax.contour(eps_ww, eps_ss, Rt_best, levels=[1.0], colors=[color], linestyles='--', linewidths=1)
            ax.contour(eps_ww, eps_ss, Rt_worst, levels=[1.0], colors=[color], linestyles='--', linewidths=1)

        ax.set_title('Controllability boundaries ($\\mathcal{R}_t=1$) with uncertainty', fontsize=14, pad=15)
        ax.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=12)
        ax.set_ylabel('Isolation efficacy $\\varepsilon_s$', fontsize=12)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[Patch(facecolor=colors[p], alpha=0.5, label=p) for p in pathogens], loc='upper right')
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight')
        plt.close(fig)

  
rule plot_combined_contour_grid_R1_Itot:
    output:
        plot="{outdir}/compartmental/combined_R1_and_Itot_reduction_contours.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        t1 = 500.0
        R_crits = [0.8, 1.0, 1.2, 1.5]
        linestyles = ['-', '--', ':', '-.']

        fig, (ax_R, ax_I) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        for pathogen in pathogens:
            model = models[pathogen]
            base_params = parameters[pathogen]
            tRt = Rt_times[pathogen]
            color = colors[pathogen]
            _, yy0 = model(params=base_params.update(epsilon_s=0.0, epsilon_w=0.0), t1=t1, E0=E0)
            baseline_Itot = yy0[0,0] - yy0[-1,0]
            for i, r_crit in enumerate(R_crits):
                Rt_grid, _, Itot_grid, _ = compute_metrics(model, base_params.update(R_crit=r_crit), eps_ww, eps_ss, t1, tRt, E0)
                ax_R.contour(eps_ww, eps_ss, np.array(Rt_grid), levels=[1.0], colors=[color], linestyles=[linestyles[i]], linewidths=2, alpha=0.8)
                ax_I.contour(eps_ww, eps_ss, np.array(Itot_grid) / float(baseline_Itot), levels=[0.2], colors=[color], linestyles=[linestyles[i]], linewidths=2, alpha=0.8)

        ax_R.set_title('Controllability boundaries ($\\mathcal{R}_t = 1$)', fontsize=14, pad=10)
        ax_I.set_title('80% reduction in total infections', fontsize=14, pad=10)
        ax_R.set_ylabel('Isolation efficacy $\\varepsilon_s$', fontsize=12)
        ax_R.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=12)
        ax_I.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=12)
        ax_R.grid(True, alpha=0.3); ax_I.grid(True, alpha=0.3)
        ax_R.set_aspect('equal'); ax_I.set_aspect('equal')
        ax_I.legend(handles=[Line2D([0],[0],color=colors[p],lw=3,label=p) for p in pathogens] + [Line2D([0],[0],color='gray',lw=2,linestyle=linestyles[i],label=f'$R_{{crit}}={r}$') for i, r in enumerate(R_crits)], loc='upper right', fontsize=11)
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


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
            rules.baseline_trajectories_no_asymptomatic.output.plot, 
            pathogen=pathogens, outdir=outdir,
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
        expand(rules.plot_main_intervention_grid.output.plot, outdir=outdir),
        expand(rules.plot_R_1_contours.output.plot, outdir=outdir),
        expand(rules.plot_combined_contour_grid_R1_Itot.output.plot, outdir=outdir),
