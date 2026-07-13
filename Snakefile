import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc, gamma, erlang
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from functools import partial
import itertools
import os

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm, CenteredNorm
import seaborn as sns

from models.parameters import Params, logistic_response_function
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.compartmental_piecewise import simulate_SEIPAR_W_piecewise, simulate_SEIAR_W_piecewise, simulate_SEIR_W_piecewise
from models.metrics import (
    R0_decomposition, infectious_fractions, transmission_fractions, growth_rate, calculate_generation_time,
    compute_R_grid, compute_asymptomatic_grid_Rt, outcome_metrics, compute_metrics, compute_delay_metrics_grid,
    calculate_mt_branching_q, calculate_mt_branching_q_with_superspreading,
    strategy_metrics, strategy_grid,
)
from models.spatial import (
    simulate_SEIPAR_W_spatial, simulate_SEIAR_W_spatial, simulate_SEIR_W_spatial, 
    SpatialParams, run_spatial, unpack_spatial,
)
from models.sensitivity import SensitivityResults, run_sensitivity_analysis, partial_rank_residuals
from models.stability import arg_L, dominant_pole, compute_rt_grid, period_and_damping, gain_margin, delay_margin
from models.gillespie import gillespie_SEIPAR_W
from models.superspreading import gillespie_SEIPAR_W_superspreading, simulate_superspreading_outcomes
from models.plotting import (
    plot_heatmap, plot_trajectory, plot_final_R, plot_I_tot, plot_I_tot_delayed_ww, 
    plot_asymptomatic_effect_for_range_of_intervention_efficacies,
    plot_extinction_probability_scenario, plot_nonlinear_response_analysis,
    table_scenario_label, table_row_metrics, f_days, f_pct,
)
from models.parameters_erlang import compute_weights
from models.compartmental_erlang import linear_chain


### PARAMETERS ###
parameters = {
    "SARS-CoV-2": Params.for_SEIPAR(R_0=2.69, gamma_inv=3.2, sigma_inv=2.3, mu_s_inv=9.3, mu_a_inv=5.0, p=0.351, phi=0.32),
    "Influenza A": Params.for_SEIAR(R_0=1.46, gamma_inv=1.65, mu_s_inv=3.38, mu_a_inv=3.38, p=0.36, phi=0.57),
    "Ebola": Params.for_SEIR(R_0=1.95, gamma_inv=8.5, mu_s_inv=5.0),
    "Omicron": Params.for_SEIPAR(R_0=7.38, gamma_inv=3.57, sigma_inv=0.52, mu_s_inv=4.94, mu_a_inv=4.94, p=0.351, phi=0.32),
    "Measles": Params.for_SEIPAR(R_0=15.0, gamma_inv=10.5, sigma_inv=3.0, mu_s_inv=4.0, mu_a_inv=4.0, p=0.0, phi=0.0),
    "Dengue": Params.for_SEIPAR(R_0=6.0, gamma_inv=5.0, sigma_inv=1.5, mu_s_inv=4.5, mu_a_inv=5.5, p=0.6, phi=0.5),
    "Rhino": Params.for_SEIPAR(R_0=2.8, gamma_inv=0.75, sigma_inv=0.75, mu_s_inv=11.0, mu_a_inv=11.0, p=22.5, phi=0.1),
}
models = {
    "SARS-CoV-2": simulate_SEIPAR_W,
    "Influenza A": simulate_SEIAR_W,
    "Ebola": simulate_SEIR_W,
    "Omicron": simulate_SEIPAR_W,
    "Measles": simulate_SEIPAR_W,
    "Dengue": simulate_SEIPAR_W,
    "Rhino": simulate_SEIPAR_W,
}
models_piecewise = {
    "SARS-CoV-2": simulate_SEIPAR_W_piecewise,
    "Influenza A": simulate_SEIAR_W_piecewise,
    "Ebola": simulate_SEIR_W_piecewise,
}
spatial_models = {
    "SARS-CoV-2": simulate_SEIPAR_W_spatial,
    "Influenza A": simulate_SEIAR_W_spatial,
    "Ebola": simulate_SEIR_W_spatial,
}
Rt_times = {
    "SARS-CoV-2": 50.0,
    "Influenza A": 100.0,
    "Ebola": 100.0,
    "Omicron": 50.0,
    "Measles": 50.0,
    "Dengue": 50.0,
    "Rhino": 50.0,
}
trajectory_end_times = {"SARS-CoV-2": 530, "Influenza A": 874, "Ebola": 1820} # 5x total wave time, rounded to nearest 10
pathogens = ["SARS-CoV-2", "Influenza A", "Ebola"]
asymptomatic_pathogens = ["SARS-CoV-2", "Influenza A"]
pathogens_full_landscape = list(parameters.keys())

best_params_kwargs = { # low R0, 1/sigma, 1/mu_a, p, phi; high 1/gamma, 1/mu_s
    "SARS-CoV-2": dict(R_0=2.40, gamma_inv=5.0, sigma_inv=0.52, mu_s_inv=10.0, mu_a_inv=4.63, p=0.230, phi=0.16),
    "Influenza A": dict(R_0=1.30, gamma_inv=1.89, mu_s_inv=4.69, mu_a_inv=2.06, p=0.32, phi=0.11),
    "Ebola": dict(R_0=1.74, gamma_inv=9.2, mu_s_inv=6.30),
    "Omicron": dict(R_0=3.5, gamma_inv=4.60, sigma_inv=0.001, mu_s_inv=6.18, mu_a_inv=3.06, p=0.230, phi=0.16),
    "Measles": dict(R_0=12.0, gamma_inv=12.0, sigma_inv=2.0, mu_s_inv=4.0, mu_a_inv=4.0, p=0.0, phi=0.0),
    "Dengue": dict(R_0=2.0, gamma_inv=8.0, sigma_inv=1.0, mu_s_inv=5.0, mu_a_inv=4.0, p=0.4, phi=0.1),
    "Rhino": dict(R_0=2.3, gamma_inv=1.0, sigma_inv=0.5, mu_s_inv=14.0, mu_a_inv=8.0, p=0.1, phi=0.1),
}
worst_params_kwargs = { # low high 1/gamma, 1/mu_s; high R0, 1/sigma, 1/mu_a, p, phi
    "SARS-CoV-2": dict(R_0=2.98, gamma_inv=2.5, sigma_inv=3.00, mu_s_inv=7.80, mu_a_inv=5.50, p=0.399, phi=0.64),
    "Influenza A": dict(R_0=1.70, gamma_inv=1.41, mu_s_inv=2.06, mu_a_inv=4.69, p=0.4, phi=1.54),
    "Ebola": dict(R_0=2.15, gamma_inv=7.70, mu_s_inv=3.70),
    "Omicron": dict(R_0=11.4, gamma_inv=2.51, sigma_inv=1.52, mu_s_inv=3.06, mu_a_inv=7.74, p=0.399, phi=0.64),
    "Measles": dict(R_0=18.0, gamma_inv=9.0, sigma_inv=4.0, mu_s_inv=4.0, mu_a_inv=4.0, p=0.0, phi=0.0),
    "Dengue": dict(R_0=10.0, gamma_inv=3.0, sigma_inv=2.0, mu_s_inv=3.0, mu_a_inv=7.0, p=0.8, phi=1.0),
    "Rhino": dict(R_0=3.0, gamma_inv=0.5, sigma_inv=1.0, mu_s_inv=8.0, mu_a_inv=14.0, p=0.35, phi=0.1),
}
colors = {
    "SARS-CoV-2": "tab:blue", "Influenza A": "tab:orange", "Ebola": "tab:green",
    "Omicron": "skyblue", "Measles": "pink", "Dengue": "red", "Rhino": "yellow",
}
p_CI = {"SARS-CoV-2": (0.23, 0.399), "Influenza A": (0.32, 0.40)}
phi_CI = {"SARS-CoV-2": (0.16, 0.64), "Influenza A": (0.11, 1.54)}
k_sc2 = 0.4

prcc_scenarios = ['start', 'threshold']
prcc_outcomes  = ['Rt', 'Itot']
prcc_scenario_titles = {'start': r'$I_{\text{crit}}=0$', 'threshold': r'$I_{\text{crit}}=10^{-4}$'}
prcc_outcome_titles = {'Rt': r'$\mathcal{R}_t$', 'Itot': r'$I_\text{tot}$'}

PARAM_LABELS: dict[str, str] = {
    "R_0": r"$\mathcal{R}_0$", "phi": r"$\varphi$", "p": r"$p$", "gamma_inv": r"$1/\gamma$", "sigma_inv": r"$1/\sigma$", 
    "mu_a_inv": r"$1/\mu_a$", "mu_s_inv": r"$1/\mu_s$", "epsilon_s": r"$\varepsilon_s$", "epsilon_w": r"$\varepsilon_w$",
    "tau_W": r"$\tau_W$", "tau_B": r"$\tau_B$", "log_k": r"$\log k$", "log_k_I": r"$\log k_I$",
    "R_crit": r"$\mathcal{R}_{\text{crit}}$", "log_I_crit": r"$\log I_{\text{crit}}$",
}

E0 = 1e-6
gillespie_popsizes = [10000] #, 1_000_000]
gillespie_num_simulations = [100]

image_resolution = 300
outdir = "results"

# Alternative warning systems
R_OFF = 0.8
EVAL_INTERVAL = 14.0
T_LEAD = 7.0
STRATEGIES = {
    "baseline": (False, False, 0.0, 1.0), "lead": (False, False, T_LEAD, 1.0),
    "interval": (False, True, 0.0, 1.0), "asymmetric": (True, False, 0.0, 0.1),
}
METRIC_NAMES = ["$\\mathcal{R}_t$", "total number of infections", "symptomatic peak", "steady-state $\\mathcal{R}_t$ amplitude", "time above $\\mathcal{R}_{crit}$", "total contact reduction cost"]
METRIC_BOUNDS = [(0.0, 1.25), (0.0, 300.0), (0.0, 175.0), (0.0, 0.0005), (0.0, 0.000025)]
INTERVENTION_SCENARIOS = {
    "SARS-CoV-2": [("baseline", 0.00, 0.00), ("isolation", 0.50, 0.00), ("warning", 0.00, 0.50), ("weak", 0.25, 0.25), ("combined", 0.50, 0.50)],
    "Influenza A": [("baseline", 0.00, 0.00), ("isolation", 0.50, 0.00), ("warning", 0.00, 0.50), ("weak", 0.25, 0.25), ("combined", 0.50, 0.50)],
    "Ebola": [("baseline", 0.00, 0.00), ("isolation", 0.50, 0.00), ("warning", 0.00, 0.50), ("weak", 0.25, 0.25), ("combined", 0.50, 0.50)],
}


###############################################
# BASELINE
###############################################

rule baseline_trajectories:
    output:
        plot="{outdir}/compartmental/baseline_trajectories_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        peak_Is, time_to_peak, total_time = plot_trajectory(model = models[wildcards.pathogen], params = parameters[wildcards.pathogen].update(I_crit=1e-4), path = output.plot, title = f"{wildcards.pathogen}", image_resolution = image_resolution, plot_total_I = True, t1 = 500.0)

rule calculate_generation_times:
    output:
        txt="{outdir}/compartmental/generation_times.txt"
    run:
        os.makedirs(os.path.dirname(output.txt), exist_ok=True)
        with open(output.txt, 'w') as f:
            for pathogen in pathogens:
                f.write(f"{pathogen}: {calculate_generation_time(parameters[pathogen])}\n")

rule calculate_growth_rates:
    output:
        txt="{outdir}/compartmental/growth_rates.txt"
    run:
        os.makedirs(os.path.dirname(output.txt), exist_ok=True)
        with open(output.txt, 'w') as f:
            for pathogen in pathogens:
                ps = parameters[pathogen]
                f.write(f"{pathogen}: {(ps.R_0-1)/calculate_generation_time(ps)}\n")

###############################################
# RESPONSE FUNCTION
###############################################

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

rule plot_nonlinear_response_analysis:
    output:
        plot="{outdir}/compartmental/nonlinear_response_analysis.png"
    run:
        path = output.plot
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plot_nonlinear_response_analysis(
            n_W = 3.0, tau_W = 14.0, n_B = 1.0, tau_B = 7.0,
            dt = 0.1, eps_w = 1.0, k = 10.0, threshold = 1.0,
            path=path, res=image_resolution, pathogens=pathogens, colors=colors,
            parameters=parameters, best_params_kwargs=best_params_kwargs, worst_params_kwargs=worst_params_kwargs
        )
        

###############################################
# INTERVENTIONS
###############################################
rule plot_trajectory:
    output:
        plot="{outdir}/compartmental/trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        ps = parameters[wildcards.pathogen].update(epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w))
        plot_trajectory(t1=trajectory_end_times[wildcards.pathogen], model=models[wildcards.pathogen], params=ps, path=output.plot, title=f"{wildcards.pathogen} ($\\varepsilon_s={wildcards.epsilon_s}, \\varepsilon_w={wildcards.epsilon_w}$)", image_resolution=image_resolution, plot_total_I=True)

rule delayed_ww_intervention:
    output:
        plot="{outdir}/compartmental/delay_grid_ww_intervention_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig = plot_I_tot_delayed_ww(model=simulate_SEIPAR_W, parameters=Params.for_SEIPAR(epsilon_s=0.0, epsilon_w=0.8))
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_main_intervention_grid:
    output:
        plot="{outdir}/compartmental/main_intervention_grid.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        t1 = 30000.0

        # compute per pathogen
        Rt_g, tRt_g, Itot_g, peakIs_g = {}, {}, {}, {}
        for pathogen in pathogens:
            ps = parameters[pathogen]
            Rt, tRt, It, pk = compute_metrics(model=models[pathogen], base_params=ps, eps_ww=eps_ww, eps_ss=eps_ss, t1=t1, E0=E0)
            Rt_g[pathogen] = np.array(Rt)
            tRt_g[pathogen] = np.array(tRt)
            _, yy0 = models[pathogen](params=ps.update(epsilon_s=0.0, epsilon_w=0.0), t1=t1, E0=E0)
            Itot_g[pathogen] = np.array(It) / float(yy0[0,0] - yy0[-1,0])
            Is0 = yy0[:, -(ps.n_W + ps.n_B + 2)]
            peakIs_g[pathogen] = np.array(pk) / float(np.max(Is0))

        # plot
        rows = [ # label, data, cmap, center_at_one, log
            ('$\\mathcal{R}_t$', Rt_g, 'RdBu_r', True, False),
            ('Time to $\\mathcal{R}_t<1$', tRt_g, 'magma', False, True),
            ('$I_\\text{tot}$ (relative to baseline)', Itot_g, 'viridis', False, False),
            ('Peak $I_s$ (relative to baseline)', peakIs_g, 'viridis', False, False),
        ]
        fig, axs = plt.subplots(nrows=len(rows), ncols=len(pathogens), figsize=(13, 16), sharex=True, sharey=True)
        for row_idx, (label, data, cmap, center_at_one, logscale) in enumerate(rows):

            # normalisation
            vals = np.concatenate([d.ravel() for d in data.values()])
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if center_at_one:
                d = max(abs(vmin - 1.0), abs(vmax - 1.0))
                norm = Normalize(vmin=1.0 - d, vmax=1.0 + d)
            elif logscale:
                norm = LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

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

rule plot_R_1_contours:
    output:
        plot="{outdir}/compartmental/R_1_contours.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        
        for pathogen in pathogens:
            mean_params = parameters[pathogen].update(k=1.0)
            best_params = mean_params.update(**best_params_kwargs[pathogen])
            worst_params = mean_params.update(**worst_params_kwargs[pathogen])
            t1 = 3000 #Rt_times[pathogen]
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

        # ax.set_title('Controllability boundaries ($\\mathcal{R}_t=1$) with uncertainty', fontsize=14, pad=15)
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
        t1 = 1000.0
        R_crits = [0.9, 1.0, 1.1, 1.2]
        linestyles = ['--', '-', '-.', ':']

        fig, (ax_R, ax_I) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        for pathogen in pathogens:
            model = models[pathogen]
            base_params = parameters[pathogen]
            color = colors[pathogen]
            _, yy0 = model(params=base_params, t1=t1, E0=E0)
            baseline_Itot = yy0[0,0] - yy0[-1,0]
            for i, r_crit in enumerate(R_crits):
                Rt_grid, _, Itot_grid, _ = compute_metrics(model=model, base_params=base_params.update(R_crit=r_crit), eps_ww=eps_ww, eps_ss=eps_ss, t1=t1, E0=E0)
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

rule plot_controllability_boundaries:
    output:
        plot="{outdir}/compartmental/controllability_boundaries.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)

        ps = jnp.linspace(0.0, 0.999, 100)
        phis = jnp.linspace(0.0, 0.999, 100)
        eps_s_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        eps_w_levels = [0.0, 0.2, 0.4, 0.6, 0.8]

        fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True) #, gridspec_kw={'width_ratios': [5,5,1]})
        for ax, pathogen in zip(axs, asymptomatic_pathogens):
            base = parameters[pathogen]
            model = models[pathogen]
            t1 = 10000 #Rt_times[pathogen]
            shade_map = {0.0: 'red', 0.2: 'orange', 0.4: 'yellow', 0.6:'lime', 0.8: 'green'}

            # nonsymptomatic fraction heatmap
            P, PHI = np.meshgrid(np.array(ps), np.array(phis), indexing='xy')
            Ra = P * PHI * base.mu_a_inv
            Rp = (1.0 - P) * base.sigma_inv
            Rs = (1.0 - P) * base.mu_s_inv
            mesh = ax.pcolormesh(np.array(ps), np.array(phis), (Ra + Rp) / (Ra + Rp + Rs), cmap='Greys', vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
            if pathogen == "Influenza A": fig.colorbar(mesh)

            # Rt contours
            for eps_s in eps_s_levels:
                for eps_w in eps_w_levels:
                    params_int = base.update(epsilon_s=eps_s, epsilon_w=eps_w)
                    Rt = np.array(compute_asymptomatic_grid_Rt(model=model, base_params=params_int, p=ps, phi=phis, t1=t1, E0=E0))
                    ax.contour(np.array(ps), np.array(phis), Rt, levels=[1.0], colors=[shade_map[eps_s]], linestyles='dotted' if eps_w<0.1 else [(0, (1, 1))] if eps_w<0.3 else 'dashed' if eps_w<0.5 else [(0, (5, 1))] if eps_w<0.7 else '-', linewidths=2.0)

            # literature estimates
            p_lower, p_upper = p_CI.get(pathogen, (None, None))
            phi_lower, phi_upper = phi_CI.get(pathogen, (None, None))
            xerr = np.array([[base.p - p_lower], [p_upper - base.p]]) if p_lower is not None else None
            yerr = np.array([[base.phi - phi_lower], [phi_upper - base.phi]]) if phi_lower is not None else None
            ax.errorbar(base.p, base.phi, xerr=xerr, yerr=yerr, fmt='o', color='white', markeredgecolor='black', ecolor='black', elinewidth=1.2, capsize=3, markersize=6, zorder=5)

            # axes
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xlabel(r'Proportion asymptomatic $p$', fontsize=12)
            ax.set_title(pathogen, fontsize=14, pad=8)
        axs[0].set_ylabel(r'Relative infectiousness $\varphi$', fontsize=12)

        # legend
        legend_handles = [
            # Line2D([0],[0], color='red', lw=2, label=r'$\varepsilon_s = 0.0$'),
            Patch(facecolor='orange', label=r'$\varepsilon_s = 0.2$'),
            Patch(facecolor='yellow', label=r'$\varepsilon_s = 0.4$'),
            Patch(facecolor='lime', label=r'$\varepsilon_s = 0.6$'),
            Patch(facecolor='green', label=r'$\varepsilon_s = 0.8$'),
            Line2D([0],[0], color='gray', lw=2, ls='dotted', label=r'$\varepsilon_w = 0.0$'),
            Line2D([0],[0], color='gray', lw=2, ls=(0, (1, 1)), label=r'$\varepsilon_w = 0.2$'),
            Line2D([0],[0], color='gray', lw=2, ls='dashed', label=r'$\varepsilon_w = 0.4$'),
            Line2D([0],[0], color='gray', lw=2, ls=(0, (5, 1)), label=r'$\varepsilon_w = 0.6$'),
            Line2D([0],[0], color='gray', lw=2, ls='-', label=r'$\varepsilon_w = 0.8$'),
            Line2D([0],[0], marker='o', color='white', markeredgecolor='black', linestyle='None', markersize=6, label=r'literature estimates'),
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=10)

        # fig.suptitle(r'Controllability boundary ($\mathcal{R}_t=1$) for varying asymptomaticity', fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


def _empirical_bounds(pathogen: str) -> dict[str, tuple[float, float]]:
    best, worst = best_params_kwargs[pathogen], worst_params_kwargs[pathogen]
    return {k: (min(best[k], worst[k]), max(best[k], worst[k])) for k in best}

def _transmission_decomposition(ps: Params) -> dict[str, float]:
    R = R0_decomposition(ps)
    t = transmission_fractions(ps)
    f = infectious_fractions(ps)
    decomposition = {"growth_rate": growth_rate(ps)}
    for k in ("a", "p", "s"):
        decomposition[f"R_{k}"] = R[k]
        decomposition[f"trans_{k}"] = t[k]
        decomposition[f"inf_{k}"] = f[k]
    return decomposition

rule transmission_decomposition:
    output:
        csv="{outdir}/compartmental/transmission_decomposition.csv",
    run:
        os.makedirs(os.path.dirname(output.csv), exist_ok=True)
        n_samples=5000
        ci=(2.5, 97.5)
        seed=0
        
        rows = []
        for pathogen in pathogens:
            ps = parameters[pathogen]

            # MC over empirical bounds
            bounds = _empirical_bounds(pathogen)
            names = list(bounds)
            X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=seed).random(n=n_samples), np.array([bounds[k][0] for k in names]), np.array([bounds[k][1] for k in names]))
            samples = [_transmission_decomposition(ps.update(**dict(zip(names, row)))) for row in X]

            def _summarise(key, scale):
                point_val = _transmission_decomposition(ps)[key] * scale
                draws = np.array([s[key] for s in samples]) * scale
                lo_ci, hi_ci = np.percentile(draws, ci)
                return point_val, float(draws.mean()), float(lo_ci), float(hi_ci)
            pv, mn, l, h = _summarise("growth_rate", 1.0)
            rows.append((pathogen, "growth_rate", "-", pv, mn, l, h))
            for qname, (prefix, scale) in {"R": ("R_", 1.0), "transmission_pct":("trans_", 100.0), "infectious_pct": ("inf_", 100.0)}.items():
                for t in ["a", "p", "s"]:
                    pv, mn, l, h = _summarise(f"{prefix}{t}", scale)
                    rows.append((pathogen, qname, t, pv, mn, l, h))

        with open(output.csv, "w") as f:
            f.write("pathogen,quantity,type,point,mean,ci_lo,ci_hi\n")
            for pathogen, q, t, pv, mn, l, h in rows: f.write(f"{pathogen},{q},{t},{pv},{mn},{l},{h}\n")

### characteristics table ###
rule baseline_intervention_table:
    output:
        tex="{outdir}/compartmental/baseline_intervention_table.tex",
    run:
        os.makedirs(os.path.dirname(output.tex), exist_ok=True)
        t1 = 10_000.0
        rows = {p: [] for p in pathogens}
        for pathogen in pathogens:
            base = table_row_metrics(parameters[pathogen], models[pathogen], 0.0, 0.0, t1)
            base["prevented"] = 0.0
            for name, eps_s, eps_w in INTERVENTION_SCENARIOS[pathogen]:
                m = (base if (eps_s == 0.0 and eps_w == 0.0) else table_row_metrics(parameters[pathogen], models[pathogen], eps_s, eps_w, t1, itot_baseline=base["itot"]))
                rows[pathogen].append((name, eps_s, eps_w, m))
        with open(output.tex, "w") as f:
            f.write("\\begin{table}[H]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{scenario} & $\\mathcal{R}_t$ & \\textbf{peak sympt.} & \\textbf{time to peak} & \\textbf{total wave time} & \\textbf{attack rate} & \\textbf{infections prevented} & \\textbf{isolation cost} & \\textbf{warning cost}\\\\\n\\midrule\n")
            for i, pathogen in enumerate(pathogens):
                f.write(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{pathogen}}}}}\\\\\n")
                for name, eps_s, eps_w, m in rows[pathogen]:
                    tp, wt = f_days(m)
                    f.write(" & ".join([f"\\quad {table_scenario_label(name, eps_s, eps_w)}", f"{m['Rt']:.2f}", f_pct(m["peak_Is"], 1), tp, wt, f_pct(m["itot"], 0), f_pct(m["prevented"], 0), f"{m['isolation_cost']:.1f}", f"{m['warning_cost']:.1f}",]) + " \\\\\n")
                if i < len(pathogens)-1: f.write("\\midrule\n")
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption[Baseline and intervention characteristics of epidemic scenarios]{Baseline and intervention characteristics of epidemic scenarios. $\\mathcal{R}_t$ is the effective reproductive number after interventions. Peak symptomatic is the maximum $I_s$ fraction; time to peak and total wave time are measured in days from when $I_s$ crosses $I_\\text{crit}=10^{-4}$ until the peak and until it drops back below. The rows without a peak and wave have an initial $\\mathcal{R}_t(t=0)=\\mathcal{R}_\\varepsilon$ below one and thus cause no outbreak. Infections prevented is measured relative to the no-intervention baseline. The cost is the total contact-reduction in person$\\times$days, $\\int\\varepsilon_s I_s\\mathrm{d}t$ for the isolation cost and $\int(1-B_\\text{out})\\mathrm{d}t$ for the warning cost. \\textsuperscript{$\dagger$}rounded.}\\label{tab:baseline_intervention}\n\\end{table}\n")


###############################################
# ASYMPTOMATIC
###############################################

rule plot_asymptomatic_grid_Rt_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Rt_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], p_CI=p_CI[wildcards.pathogen], phi_CI=phi_CI[wildcards.pathogen], total_infected=False, path=output.plot, image_resolution=image_resolution, t1=Rt_times[wildcards.pathogen])

rule plot_asymptomatic_grid_Itot_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Itot_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(t1=600.0, model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], p_CI=p_CI[wildcards.pathogen], phi_CI=phi_CI[wildcards.pathogen], total_infected=True, path=output.plot,image_resolution=image_resolution)

rule plot_asymptomatic_generation_time:
    output:
        plot="{outdir}/compartmental/asymptomatic_generation_time.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        ps = parameters["SARS-CoV-2"]
        P = jnp.linspace(0.0, 0.999, 100)
        PHI = jnp.linspace(0.0, 0.999, 100)

        def get_generation_time(p, phi):
            nom = p * phi * ps.mu_a_inv**2 +(1-p)*(ps.sigma_inv**2 + ps.mu_s_inv**2 + ps.sigma_inv*ps.mu_s_inv)
            denom = p * phi * ps.mu_a_inv + (1-p)*(ps.sigma_inv + ps.mu_s_inv)
            return ps.gamma_inv + nom / denom
        generation_times = jax.vmap(jax.vmap(get_generation_time, in_axes=(None, 0)), in_axes=(0, None))(P, PHI)

        fig, ax = plot_heatmap(
            X=P, Y=PHI, Z=generation_times, cmap='magma', contour_levels=[11,12,13,14,15],
            title='Asymptomatic transmission and generation intervals',
            xlabel=r'Proportion asymptomatic, $p$', ylabel=r'Relative infectiousness, $\varphi$',
        )
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_asymptomatic_landscape:
    output:
        plot="{outdir}/compartmental/asymptomatic_landscape.png"
    run:
        def sample(best_params, worst_params, n_samples, seed=0):
            rng = np.random.default_rng(seed)
            def _sample(name):
                def _bounds(name):
                    a = getattr(best_params, name)
                    b = getattr(worst_params, name)
                    return (min(a, b), max(a, b))
                lo, hi = _bounds(name)
                if lo == hi: return np.full(n_samples, lo)
                return rng.normal((lo+hi)/2, (hi-lo)/(2*1.96), size=n_samples)
            
            p = _sample('p')
            R_a = p * _sample('phi') * _sample('mu_a_inv')
            R_p = (1.0 - p) * _sample('sigma_inv')
            R_s = (1.0 - p) * _sample('mu_s_inv')
            theta = (R_a + R_p) / (R_a + R_p + R_s)
            return np.asarray(_sample('R_0')), np.asarray(theta)

        def _R_boundary(theta, eps_s, eps_w):
            return 1.0 / ((1.0 - eps_w / 2.0) * (1.0 - eps_s * (1.0 - theta)))

        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,6))

        # asymptomatic landscape
        for pathogen in pathogens_full_landscape:
            mean_params = parameters[pathogen]
            best_params = mean_params.update(**best_params_kwargs[pathogen])
            worst_params = mean_params.update(**worst_params_kwargs[pathogen])
            R0_s, theta_s = sample(best_params=best_params, worst_params=worst_params, n_samples=10000)
            ax.scatter(theta_s, R0_s, s=4, alpha=0.1, color=colors[pathogen], edgecolors='none')

        # controllability boundaries
        thetas = np.linspace(0, 1, 500)
        scenarios = [
            {"label": "Warning ($\epsilon_s=0, \epsilon_w=0.8$)", "eps_s": 0.0, "eps_w": 0.8, "color": "black"},
            {"label": "Isolation ($\epsilon_s=0.8, \epsilon_w=0$)", "eps_s": 0.8, "eps_w": 0.0, "color": "grey"},
            {"label": "Combined ($\epsilon_s=0.8, \epsilon_w=0.8$)", "eps_s": 0.8, "eps_w": 0.8, "color": "green"},
        ]
        for s in scenarios:
            R_boundary = _R_boundary(thetas, s["eps_s"], s["eps_w"])
            ax.plot(thetas, R_boundary, color=s["color"], linewidth=2, linestyle='-', label=s["label"])

        # plotting
        ax.set_xlim([-0.01, 1])
        ax.set_ylim([0, 20])
        ax.set_xlabel('Proportion presymptomatic and asymptomatic $\\theta$', fontsize=12)
        ax.set_ylabel('Basic reproductive number $\\mathcal{R}_0$', fontsize=12)
        ax.add_artist(ax.legend(handles=[Patch(facecolor=colors[p], label=p) for p in pathogens_full_landscape], loc='upper left', title="Pathogens"))
        ax.legend(loc='upper right', title="Controllability ($\\mathcal{R}_t<1$)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output.plot, dpi=image_resolution); plt.close()


###############################################
# SENSITIVITY ANALYSIS
###############################################

rule compute_prcc:
    output:
        npz="{outdir}/compartmental/prcc/data_{pathogen}_{scenario}_{outcome}_{bounds}.npz"
    run:
        os.makedirs(os.path.dirname(output.npz), exist_ok=True)
        p = wildcards.pathogen
        t1 = 1000.0
        best = best_params_kwargs[p]; worst = worst_params_kwargs[p]
        specific_bounds = {}
        for k in best.keys():
            l_val = min(best[k], worst[k]); u_val = max(best[k], worst[k])
            if l_val == u_val: u_val += 1e-5
            specific_bounds[k] = (l_val, u_val)
        around_mean = wildcards.bounds=="symmetric"
        results = run_sensitivity_analysis(
            model=models[wildcards.pathogen], scenario=wildcards.scenario, outcome=wildcards.outcome,
            base_params=parameters[wildcards.pathogen].update(epsilon_s=0.4, epsilon_w=0.8),
            manual_bounds=None if around_mean else specific_bounds,
            t1=t1, E0=E0, n_lhs=5000, n_bootstrap=100, n_sobol_base=1024, avg_frac=0.1, around_mean=around_mean,
        )
        np.savez_compressed(output.npz, param_names=np.array(results.param_names), lower_bounds=np.array([results.bounds[k][0] for k in results.param_names]), upper_bounds=np.array([results.bounds[k][1] for k in results.param_names]), samples=results.samples, outputs=results.outputs, prcc_mean=results.prcc_mean, prcc_lower=results.prcc_lower, prcc_upper=results.prcc_upper, prcc_samples=results.prcc_samples, sobol_S1=results.sobol_S1, sobol_S1_conf=results.sobol_S1_conf, sobol_ST=results.sobol_ST, sobol_ST_conf=results.sobol_ST_conf)

def load_sensitivity_results(path: str) -> SensitivityResults:
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["param_names"]]
    return SensitivityResults(param_names=names, bounds={n:(float(l),float(u)) for n,l,u in zip(names,d["lower_bounds"],d["upper_bounds"])}, samples=d["samples"], outputs=d["outputs"], prcc_mean=d["prcc_mean"], prcc_lower=d["prcc_lower"], prcc_upper=d["prcc_upper"], prcc_samples=d["prcc_samples"], sobol_S1=d["sobol_S1"], sobol_S1_conf=d["sobol_S1_conf"], sobol_ST=d["sobol_ST"], sobol_ST_conf=d["sobol_ST_conf"])

rule plot_prcc_monotonicity:
    input:
        npz=rules.compute_prcc.output.npz
    output:
        plot="{outdir}/compartmental/prcc/monotonicity_{pathogen}_{scenario}_{outcome}_{bounds}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        results = load_sensitivity_results(input.npz)
        
        # subplots
        d = len(results.param_names)
        n_cols = 4
        n_rows = int(np.ceil(d / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        
        def calculate_binned_means(x, y, num_bins=20):
            """Sort data by x and calculate means across bins"""
            order = np.argsort(x)
            x_sorted, y_sorted = x[order], y[order]
            bin_edges = np.linspace(0, len(x_sorted), num_bins + 1).astype(int)
            x_means, y_means = [], []
            for start, end in zip(bin_edges[:-1], bin_edges[1:]):
                if end > start: x_means.append(x_sorted[start:end].mean()); y_means.append(y_sorted[start:end].mean()) 
            return np.array(x_means), np.array(y_means)

        # plot
        for i, name in enumerate(results.param_names):
            ax = axs[i // n_cols, i % n_cols]
            ex, ey = partial_rank_residuals(results.samples, results.outputs, i)
            ax.scatter(ex, ey, s=4, alpha=0.25, edgecolors='none', color=colors[wildcards.pathogen])
            ax.set_title(f'{PARAM_LABELS.get(name, name)} (PRCC $= {results.prcc_mean[i]:+.2f}$)', fontsize=18)
            x_trend, y_trend = calculate_binned_means(ex, ey)
            ax.plot(x_trend, y_trend, color='black', lw=1.4, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color='gray', lw=0.4, alpha=0.5); ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
        for j in range(d, n_rows*n_cols): axs[j//n_cols, j%n_cols].axis('off') # remove unused subplots

        # labels and titles
        fig.supxlabel('partial rank of parameter', fontsize=24)
        fig.supylabel('partial rank of output', fontsize=24)
        fig.suptitle(f'{wildcards.pathogen}, {prcc_scenario_titles[wildcards.scenario]}, {prcc_outcome_titles[wildcards.outcome]}', fontsize=32)
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)

rule plot_prcc_grid:
    input:
        npz=lambda wc: expand("{outdir}/compartmental/prcc/data_{pathogen}_{scenario}_{outcome}_{bounds}.npz", outdir=wc.outdir, pathogen=pathogens, scenario=['start', 'threshold'], outcome=prcc_outcomes, bounds=wc.bounds)
    output:
        plot="{outdir}/compartmental/combined_prcc_grid_{bounds}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        # all params for global x axis
        all_params = []
        for path in input.npz:
            res = load_sensitivity_results(path)
            for p in res.param_names: 
                if p not in all_params: all_params.append(p)  
        labels = [PARAM_LABELS.get(n, n) for n in all_params]
        x = np.arange(len(all_params))

        def params_aligned(res, metric, is_err=False, is_abs=False):
            """Align params to shared x axis: insert nan if not available"""
            a = []
            for p in all_params:
                if p in res.param_names:
                    idx = res.param_names.index(p)
                    if metric == 'prcc_err': val = (res.prcc_upper[idx]-res.prcc_lower[idx])/2
                    else:
                        val = getattr(res, metric)[idx]
                        if is_abs: val = np.abs(val)
                    a.append(val)
                else: a.append(np.nan)
            return np.array(a)

        # plot
        n_rows = len(pathogens)
        n_cols = len(prcc_outcomes)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), sharey=True)
        w = 0.32
        for r, pathogen in enumerate(pathogens):
            color = colors[pathogen]
            for c, outcome in enumerate(prcc_outcomes):
                ax = axs[r, c]
                results_start = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_start_{outcome}_{wildcards.bounds}.npz")
                results_threshold = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_threshold_{outcome}_{wildcards.bounds}.npz")
                # bars
                err_kw = dict(ecolor='k', linewidth=0.6, capsize=1.5)
                edge_kw = dict(edgecolor='k', linewidth=0.3)
                ax.bar(x - 0.5*w, params_aligned(results_start, 'prcc_mean', is_abs=False), yerr=params_aligned(results_start, 'prcc_err'), width=w, color=color, **edge_kw, error_kw=err_kw)
                ax.bar(x + 0.5*w, params_aligned(results_threshold, 'prcc_mean', is_abs=False), yerr=params_aligned(results_threshold, 'prcc_err'), width=w, color=color, alpha=0.5, hatch='////', **edge_kw, error_kw=err_kw)                
                # ticks
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
                # grid
                ax.set_ylim(-1.05, 1.05)
                ax.grid(axis='y', alpha=0.3)
                # labels
                if c == 0: ax.set_ylabel(f'{pathogen}', fontsize=20)
                if r == 0: ax.set_title(prcc_outcome_titles.get(outcome, outcome), fontsize=20)
        # legend
        fig.legend(
            handles=[Patch(facecolor='gray', label=r'PRCC ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', hatch='////', label=r'PRCC ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3, alpha=0.5)],
            loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=14, frameon=True)
        # save and close
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)

rule plot_combined_sensitivity_grid:
    input:
        npz=lambda wc: expand("{outdir}/compartmental/prcc/data_{pathogen}_{scenario}_{outcome}_{bounds}.npz", outdir=wc.outdir, pathogen=pathogens, scenario=['start', 'threshold'], outcome=prcc_outcomes, bounds=["empirical"])
    output:
        plot="{outdir}/compartmental/combined_sensitivity_grid.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        
        # all params for global x axis
        all_params = []
        for path in input.npz:
            res = load_sensitivity_results(path)
            for p in res.param_names: 
                if p not in all_params: all_params.append(p)  
        labels = [PARAM_LABELS.get(n, n) for n in all_params]
        x = np.arange(len(all_params))

        def params_aligned(res, metric, is_err=False, is_abs=False):
            """Align params to shared x axis: insert nan if not available"""
            a = []
            for p in all_params:
                if p in res.param_names:
                    idx = res.param_names.index(p)
                    if metric == 'prcc_err': val = (res.prcc_upper[idx]-res.prcc_lower[idx])/2
                    else:
                        val = getattr(res, metric)[idx]
                        if is_abs: val = np.abs(val)
                    a.append(val)
                else: a.append(np.nan)
            return np.array(a)

        # plot
        n_rows = len(pathogens)
        n_cols = len(prcc_outcomes)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 5*n_rows), sharey=True)
        w = 0.14
        for r, pathogen in enumerate(pathogens):
            color = colors[pathogen]
            for c, outcome in enumerate(prcc_outcomes):
                ax = axs[r, c]
                results_start = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_start_{outcome}_empirical.npz")
                results_threshold = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_threshold_{outcome}_empirical.npz")
                # bars
                err_kw = dict(ecolor='k', linewidth=0.6, capsize=1.5)
                edge_kw = dict(edgecolor='k', linewidth=0.3)
                ax.bar(x - 2.5*w, params_aligned(results_start, 'prcc_mean', is_abs=True), yerr=params_aligned(results_start, 'prcc_err'), width=w, color=color, **edge_kw, error_kw=err_kw)
                ax.bar(x - 1.5*w, params_aligned(results_threshold, 'prcc_mean', is_abs=True), yerr=params_aligned(results_threshold, 'prcc_err'), width=w, color=color, hatch='////', **edge_kw, error_kw=err_kw)
                ax.bar(x - 0.5*w, params_aligned(results_start, 'sobol_S1'), yerr=params_aligned(results_start, 'sobol_S1_conf', is_err=True), width=w, color=color, alpha=0.7, **edge_kw, error_kw=err_kw)
                ax.bar(x + 0.5*w, params_aligned(results_threshold, 'sobol_S1'), yerr=params_aligned(results_threshold, 'sobol_S1_conf', is_err=True), width=w, color=color, alpha=0.7, hatch='////', **edge_kw, error_kw=err_kw)
                ax.bar(x + 1.5*w, params_aligned(results_start, 'sobol_ST'), yerr=params_aligned(results_start, 'sobol_ST_conf', is_err=True), width=w, color=color, alpha=0.35, **edge_kw, error_kw=err_kw)
                ax.bar(x + 2.5*w, params_aligned(results_threshold, 'sobol_ST'), yerr=params_aligned(results_threshold, 'sobol_ST_conf', is_err=True), width=w, color=color, alpha=0.35, hatch='////', **edge_kw, error_kw=err_kw)
                # ticks
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
                # grid
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                # labels
                if c == 0: ax.set_ylabel(f'{pathogen}', fontsize=20)
                if r == 0: ax.set_title(prcc_outcome_titles.get(outcome, outcome), fontsize=20)
        # legend
        fig.legend(
            handles=[
                Patch(facecolor='gray', label=r'|PRCC| ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', hatch='////', label=r'|PRCC| ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.7, label=r'$S_1$ ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.7, hatch='////', label=r'$S_1$ ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.35, label=r'$S_T$ ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.35, hatch='////', label=r'$S_T$ ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3)], 
            loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=14, frameon=True)
        # save and close
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)

rule export_param_bounds:
    input:
        npzs=lambda wc: expand(rules.compute_prcc.output.npz, pathogen=pathogens, scenario="threshold", outcome="Itot", outdir=outdir, bounds=wc.bounds)
    output:
        tex="{outdir}/compartmental/prcc/sensitivity_bounds_table_{bounds}.tex"
    run:
        os.makedirs(os.path.dirname(output.tex), exist_ok=True)
        combinations = list(itertools.product(pathogens, ["threshold"]))
        col_mapping = {("SARS-CoV-2", "threshold"): "SARS-CoV-2", ("Influenza A", "threshold"): "H1N1", ("Ebola", "threshold"): "Ebola"}
        param_defs = {
            "R_0": ("$\\mathcal{R}_0$", "basic reproductive number"), "gamma_inv": ("$1/\\gamma$", "latent period"), "mu_s_inv": ("$1/\\mu_s$", "symptomatic period"), 
            "sigma_inv": ("$1/\\sigma$", "presymptomatic period"), 
            "mu_a_inv": ("$1/\\mu_a$", "asymptomatic period"), "p": ("$p$", "proportion asymptomatic"), "phi": ("$\\varphi$", "relative asympt. infectiousness"), 
            "epsilon_s": ("$\\varepsilon_s$", "isolation efficacy"), "epsilon_w": ("$\\varepsilon_w$", "warning response efficacy"), "tau_W": ("$\\tau_W$", "reporting delay"), "tau_B": ("$\\tau_B$", "behavioural delay"), 
            "log_k": ("$\\log_{10} k$", "warning gate sharpness"), "R_crit": ("$\\mathcal{R}_{\\text{crit}}$", "warning threshold"), 
            "log_k_I": ("$\\log_{10} k_I$", "prevalence gate sharpness (*)"), "log_I_crit": ("$\\log_{10} I_{\\text{crit}}$", "prevalence threshold (*)"), 
        }
        bounds_data = {}
        for combination, fpath in zip(combinations, input.npzs): bounds_data[combination] = load_sensitivity_results(fpath).bounds
        with open(output.tex, 'w') as f:
            f.write("\\begin{table}[H]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{llccc}\n\\toprule\n")
            header = ["Parameter", ""] + [col_mapping[c] for c in combinations]
            f.write(" & ".join(header) + " \\\\\n\\midrule\n")
            for p_key, (symbol, desc) in param_defs.items():
                row = [desc, symbol]
                for combination in combinations: 
                    if p_key in bounds_data[combination]: row.append(f"$[{bounds_data[combination][p_key][0]:g}, {bounds_data[combination][p_key][1]:g}]$")
                    else: row.append("---")
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption[Parameter ranges used for sensitivity analysis]{Parameter ranges used for Latin hypercube sampling in the global sensitivity analysis. Parameters marked with an asterisk (*) are only included when the symptomatic threshold $I_\\text{crit}=10^{-4}$ is active.}\n\\label{tab:prcc-bounds}\n\\end{table}\n")


###############################################
# STABILITY ANALYSIS
###############################################

rule plot_true_vs_reported_Rt_scenarios:
    output:
        plot="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_scenarios.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)

        taus_W = [7.0, 14.0, 21.0]
        taus_B = [3.0, 7.0, 14.0]
        ks = [5.0, 10.0, 50.0]
        k = 10.0
        t1 = 300.0

        epsilon_s = 0.5 if wildcards.pathogen == "SARS-CoV-2" else 0.0
        base_params = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, k=k)
        model = models[wildcards.pathogen]

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=len(taus_W)+1, ncols=len(taus_B), figsize=(12, 12), sharex=True, sharey=True)

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
                ax.text(0.97, 0.15, f'{total_time_above:.0f} days above $R_{{crit}}$\n{num_crossings} warnings', transform=ax.transAxes, ha='right', va='top', fontsize=8)

        base_params = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, tau_W=14.0, tau_B=7.0)

        for i, k in enumerate(ks):
            params = base_params.update(k=k)
            tt, yy = model(params=params, t1=t1)
            rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
            rt_reported = yy[:, -(params.n_B + 1)]
            above = (rt_reported >= params.R_crit).astype(jnp.float32)
            total_time_above = float(above.mean() * t1)
            num_crossings = int(jnp.sum(jnp.diff(above) > 0))

            ax = axs[len(taus_W),i]
            ax.set_title(f'$k={k}$', fontsize=16)
            ax.set_ylabel(f'$\\tau_W={params.tau_W},\\tau_B={params.tau_B}$', fontsize=16)
            ax.plot(tt, rt_true, color='black')
            ax.plot(tt, rt_reported, color='red')
            ax.axhline(params.R_crit, color='grey', linestyle='--')
            ax.text(0.97, 0.15, f'{total_time_above:.0f} days above $R_{{crit}}$\n{num_crossings} warnings', transform=ax.transAxes, ha='right', va='top', fontsize=8)

        fig.legend(
            [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='red', lw=2)],
            ['True $R_t$', 'Reported $R_t$'],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=16,
        )
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close()


rule plot_true_vs_reported_Rt_heatmaps:
    output:
        Rt_final ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_Rt_final.png",
        time_to_below ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_time_to_below.png",
        Itot ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_Itot.png",
        peak_Is ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_peak_Is.png",
        amplitudes ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_amplitudes.png",
        time_above ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_time_above.png",
        crossings ="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_heatmap_crossings.png",
    run:
        for path in output: os.makedirs(os.path.dirname(path), exist_ok=True)
        pathogen = wildcards.pathogen
        taus_W = jnp.linspace(1.0, 31.0, num=100)
        taus_B = jnp.linspace(1.0, 31.0, num=100)
        k = float(wildcards.k)
        epsilon_s = 0.5 if wildcards.pathogen == "SARS-CoV-2" else 0.0
        base_params = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, k=k)

        Rt_final, time_to_below, Itot, peak_Is, amplitudes, time_above, crossings = compute_delay_metrics_grid(model=models[wildcards.pathogen], base_params=base_params, taus_W=taus_W, taus_B=taus_B)

        plt.figure(figsize=(6,6))
        plt.scatter(amplitudes, Itot, c='k', alpha=0.2, s=2) # TODO: color for total delay
        plt.title('Effect of oscillations on the number of infections')
        plt.xlabel('Oscillation amplitudes')
        plt.ylabel('Total fraction infected')
        plt.savefig(f"{wildcards.outdir}/compartmental/true_vs_reported_Rt_{wildcards.pathogen}_k{wildcards.k}_scatter_amplitudes_vs_fractions.png", dpi=image_resolution); plt.close()

        kwargs = dict(x_logscale=False, xlabel='Behavioural delay ($\\tau_B$)', ylabel='Reporting delay ($\\tau_W$)')
        scenario = f'({wildcards.pathogen}, $k={k:g}$)'
        fig, _ = plot_heatmap(taus_B, taus_W, amplitudes, cmap='magma', cbar_label='Amplitude of oscillations', title=f'Stability of delayed response {scenario}', **kwargs)
        fig.savefig(output.amplitudes, dpi=image_resolution); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, time_above, cmap='cividis', cbar_label='Days above warning threshold', title=f'Time above warning threshold {scenario}', **kwargs)
        fig.savefig(output.time_above, dpi=image_resolution); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, crossings, cmap='plasma', cbar_label='Total times warned', title=f'Number of warning-threshold crossings {scenario}', **kwargs)
        fig.savefig(output.crossings, dpi=image_resolution); plt.close(fig)

        t1 = 10000.0
        # compute per pathogen
        Rt_g, tRt_g, Itot_g, peakIs_g = {}, {}, {}, {}
        ps = parameters[pathogen]
        Rt, tRt, It, pk = Rt_final, time_to_below, Itot, peak_Is
        Rt_g[pathogen] = np.array(Rt)
        tRt_g[pathogen] = np.array(tRt)
        _, yy0 = models[pathogen](params=ps.update(epsilon_s=0.0, epsilon_w=0.0))
        Itot_g[pathogen] = np.array(It) / float(yy0[0,0] - yy0[-1,0])
        Is0 = yy0[:, -(ps.n_W + ps.n_B + 2)]
        peakIs_g[pathogen] = np.array(pk) / float(np.max(Is0))

        # plot
        cols = [ # title, label, data, cmap, center_at_one, log
            ('Rt_final', '$\\mathcal{R}_t$', Rt_g, 'RdBu_r', True, False),
            ('time_to_below', 'Time to $\\mathcal{R}_t<1$', tRt_g, 'magma', False, True),
            ('Itot', '$I_\\text{tot}$ (relative to baseline)', Itot_g, 'viridis', False, False),
            ('peak_Is', 'Peak $I_s$', peakIs_g, 'viridis', False, False),
        ]
        for col, (title, label, data, cmap, center_at_one, logscale) in enumerate(cols):

            # normalisation
            vals = np.concatenate([d.ravel() for d in data.values()])
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if center_at_one:
                d = max(abs(vmin - 1.0), abs(vmax - 1.0))
                norm = Normalize(vmin=1.0 - d, vmax=1.0 + d)
            elif logscale:
                norm = LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

            # meshgrid
            fig, ax = plt.subplots(figsize=(6,6))
            mesh = ax.pcolormesh(np.array(taus_B), np.array(taus_W), data[pathogen], cmap=cmap, norm=norm, shading='auto', rasterized=True)
            ax.set_aspect('equal')
            # contours, titles, labels
            if center_at_one: ax.contour(np.array(taus_B), np.array(taus_W), data[pathogen], levels=[1.0], colors='black', linewidths=1.0, linestyles='--')
            ax.set_title(f'{label} ({pathogen})', fontsize=14)
            ax.set_xlabel('Behavioural delay ($\\tau_B$)', fontsize=11)
            ax.set_ylabel('Reporting delay ($\\tau_W$)', fontsize=11)
            # colorbar
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, aspect=25, pad=0.02)
            cbar.set_label(label, fontsize=12, labelpad=8)
            if center_at_one: cbar.ax.axhline(1.0, color='black', linewidth=1.0)
            #save and close
            if title=='Rt_final': fig.savefig(output.Rt_final, dpi=image_resolution, bbox_inches='tight')
            elif title=='time_to_below': fig.savefig(output.time_to_below, dpi=image_resolution, bbox_inches='tight')
            elif title=='Itot': fig.savefig(output.Itot, dpi=image_resolution, bbox_inches='tight')
            elif title=='peak_Is': fig.savefig(output.peak_Is, dpi=image_resolution, bbox_inches='tight')
            plt.close(fig)


### CONTROL THEORY ###
rule plot_gain_margins:
    output:
        plot="{outdir}/compartmental/gain_margins.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_w = 0.8
        taus_W = np.linspace(1.0, 31.0, 100)
        taus_B = np.linspace(1.0, 31.0, 100)
        MG = np.array([[gain_margin(eps_w, tw, tb) for tb in taus_B] for tw in taus_W])
        vmin, vmax = float(np.nanmin(MG)), float(np.nanmax(MG))
        norm = LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
        ticks = [1,2,3,4,5,10,20]
        fig, ax = plot_heatmap(taus_B, taus_W, MG, norm=norm, cmap='magma_r', cbar_ticks=ticks, contour_levels=[1.0], xlabel=r'Behavioural delay ($\tau_B$)', ylabel=r'Reporting delay ($\tau_W$)', title='Gain margin')
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_delay_margins:
    output:
        plot="{outdir}/compartmental/delay_margins.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_w = 0.8
        taus_W = np.linspace(1.0, 31.0, 100)
        taus_B = np.linspace(1.0, 31.0, 100)
        MD = np.array([[delay_margin(eps_w, tw, tb) for tb in taus_B] for tw in taus_W])
        fig, ax = plot_heatmap(taus_B, taus_W, MD, cmap='magma_r', contour_levels=[0.0], xlabel=r'Behavioural delay ($\tau_B$)', ylabel=r'Reporting delay ($\tau_W$)', title='Delay margin')
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_period_and_damping_scatter:
    output:
        period ="{outdir}/compartmental/period_scatter_{pathogen}_k{k}.png",
        damping="{outdir}/compartmental/damping_scatter_{pathogen}_k{k}.png",
    run:
        for path in output: os.makedirs(os.path.dirname(path), exist_ok=True)
        N = 50
        eps_w = 0.8
        k = float(wildcards.k)
        t1 = 300.0
        epsilon_s = 0.5 if wildcards.pathogen == "SARS-CoV-2" else 0.0
        base_params = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=eps_w, k=k)
        model = models[wildcards.pathogen]
        n_W = base_params.n_W
        n_B = base_params.n_B
        taus_W = jnp.linspace(3.0, 31.0, N)
        taus_B = jnp.linspace(1.0, 31.0, N)

        # analytical
        analytical_period = np.full((N, N), np.nan)
        analytical_damping = np.full((N, N), np.nan)
        for i, tw in enumerate(np.array(taus_W)):
            for j, tb in enumerate(np.array(taus_B)):
                pole = dominant_pole(float(tw), float(tb), eps_w, k, n_W, n_B)
                if not np.isnan(pole):
                    analytical_period[i, j] = 2*np.pi / abs(pole.imag)
                    analytical_damping[i, j] = -pole.real

        # simulation grid
        rt_grid = np.array(compute_rt_grid(model, base_params, taus_W, taus_B, t1=t1))
        tt = np.linspace(0.0, t1, rt_grid.shape[-1])
        simulation_period = np.full((N, N), np.nan)
        simulation_damping = np.full((N, N), np.nan)
        for i in range(N):
            for j in range(N):
                simulation_period[i, j], simulation_damping[i, j] = period_and_damping(tt, rt_grid[i, j], smoothing_days = max(20, 2*analytical_period[i,j]))

        TW, TB = np.meshgrid(np.array(taus_W), np.array(taus_B), indexing='ij')
        total_delay = TW + TB

        # period scatterplot
        valid = np.isfinite(simulation_period) & np.isfinite(analytical_period)
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(analytical_period[valid], simulation_period[valid], c=total_delay[valid], cmap='viridis', s=10, alpha=0.8)
        if valid.any():
            lim = [0.9*min(analytical_period[valid].min(), simulation_period[valid].min()), 1.05*max(analytical_period[valid].max(), simulation_period[valid].max())]
            ax.plot(lim, lim, 'k--', lw=1)
            ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_xlabel('analytical')
        ax.set_ylabel('simulated')
        ax.set_title(f'Oscillation periods ({wildcards.pathogen}, $k={k:g}$)')
        plt.colorbar(sc, ax=ax, label=r'total delay $\tau_W + \tau_B$ (days)', shrink=0.7)
        plt.tight_layout()
        plt.savefig(output.period, dpi=image_resolution); plt.close(fig)

        # damping scatterplot
        valid = np.isfinite(simulation_damping) & np.isfinite(analytical_damping)
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(analytical_damping[valid], simulation_damping[valid], c=total_delay[valid], cmap='viridis', s=10, alpha=0.8)
        if valid.any():
            lim = [0, 1.05*max(analytical_damping[valid].max(), simulation_damping[valid].max())] #[min(analytical_damping[valid].min(), simulation_damping[valid].min()), 
            ax.plot(lim, lim, 'k--', lw=1)
            ax.axhline(0, color='k', lw=0.5, ls='--'); ax.axvline(0, color='k', lw=0.5, ls='--')
            ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_xlabel('analytical')
        ax.set_ylabel('simulated')
        ax.set_title(f'Decay rates ({wildcards.pathogen}, $k={k:g}$)')
        plt.colorbar(sc, ax=ax, label=r'total delay $\tau_W + \tau_B$ (days)', shrink=0.7)
        plt.tight_layout()
        plt.savefig(output.damping, dpi=image_resolution); plt.close(fig)


###############################################
# STOCHASTIC
###############################################

rule plot_stochastic_baseline_trajectories:
    output:
        plot ="{outdir}/gillespie/stochastic_baseline_trajectories_{pathogen}_N{N}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        t1 = 1000.0
        num_simulations = 100
        ps = Params.for_SEIPAR()
        E0 = 1/N
        alpha = 0.01
        Iest = np.ceil(np.log(alpha)/np.log(calculate_mt_branching_q(ps,0,0)))

        fig, (ax_traj, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(6,6), sharex=True, height_ratios=[2,1])

        initial_fadeout_times = []
        established_extinction_times = []

        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
            initial_fadeout = np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest
            if initial_fadeout: initial_fadeout_times.append((tt[-1]))
            else: established_extinction_times.append((tt[-1]))
            ax_traj.plot(tt, yy.T[0], alpha=0.05, color='grey' if initial_fadeout else colors[wildcards.pathogen])

        ymax = max(max(initial_fadeout_times), max(established_extinction_times))
        tt_det, yy_det = models[wildcards.pathogen](params=ps, t1=t1, E0=E0)
        S_det = yy_det.T[0] * N
        ax_traj.plot(tt_det, S_det, color=colors[wildcards.pathogen])
        ax_traj.set_xlim([0, ymax])
        ax_traj.set_title('Number of susceptibles')

        ax_hist.hist([initial_fadeout_times, established_extinction_times], density=True, stacked=True, color=['grey', colors[wildcards.pathogen]], bins=int(ymax//10))
        ax_hist.set_title('Extinction times')
        ax_hist.set_xlabel('days')
        fig.suptitle(f'Stochastic susceptible trajectories (N={N})')
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


rule plot_linearised_branching_process_extinction_probabilities:
    output:
        plot="{outdir}/gillespie/linearised_branching_process_extinction_probabilities_{pathogen}.png",
        I_establishment="{outdir}/gillespie/I_establishment_{pathogen}.png",

    run:
        for path in output: os.makedirs(os.path.dirname(path), exist_ok=True)

        ps = parameters[wildcards.pathogen]
        eps_ww = np.linspace(0.0, 0.999, 100)
        eps_ss = np.linspace(0.0, 0.999, 100)
        alpha = 0.01
        Iest = np.zeros((len(eps_ww), len(eps_ss)))
        qs = np.zeros((len(eps_ww), len(eps_ss)))

        for i, ew in enumerate(eps_ww):
            for j, es in enumerate(eps_ss):
                q = calculate_mt_branching_q(ps, ew, es)
                qs[j,i] = q
                Iest[j,i] = np.ceil(np.log(alpha)/np.log(q))

        fig, ax = plot_heatmap(eps_ww, eps_ss, qs, cmap='magma_r', 
            contour_metric=compute_R_grid(models[wildcards.pathogen], parameters[wildcards.pathogen]._replace(k=1), eps_ww, eps_ss, Rt_times[wildcards.pathogen]), 
            contour_levels=[1.0], contour_colors='white',
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title='Linearised branching process extinction probabilities')
        plt.savefig(output.plot, dpi=image_resolution); plt.close()

        fig, ax = plot_heatmap(eps_ww, eps_ss, Iest, cmap='plasma', 
            norm=LogNorm(vmin=np.max([float(np.nanmin(Iest)),1]), vmax=np.max([float(np.nanmax(Iest)),1])),
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title='$I_\\text{establishment}$')
        plt.savefig(output.I_establishment, dpi=image_resolution); plt.close()

rule simulate_stochastic_outcomes:
    output:
        npz="{outdir}/gillespie/{scenario}_stochastic_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    run:
        os.makedirs(os.path.dirname(output.npz), exist_ok=True)
        num_simulations = int(wildcards.num_simulations)
        N = int(wildcards.N)
        res = int(wildcards.resolution)
        t1 = 2000.0
        eps_ww = np.linspace(0.0, 0.999, res)
        eps_ss = np.linspace(0.0, 0.999, res)
        Rt_grid = np.zeros((len(eps_ww), len(eps_ss)))
        Rt_var_grid = np.zeros((len(eps_ww), len(eps_ss)))
        time_to_below_grid = np.zeros((len(eps_ww), len(eps_ss)))
        time_to_below_var_grid = np.zeros((len(eps_ww), len(eps_ss)))
        Itot_grid = np.zeros((len(eps_ww), len(eps_ss)))
        Itot_var_grid = np.zeros((len(eps_ww), len(eps_ss)))
        peak_Is_grid = np.zeros((len(eps_ww), len(eps_ss)))
        peak_Is_var_grid = np.zeros((len(eps_ww), len(eps_ss)))
        extinction_time_grid = np.zeros((len(eps_ww), len(eps_ss)))
        extinction_time_var_grid = np.zeros((len(eps_ww), len(eps_ss)))

        for i, ew in enumerate(eps_ww):
            for j, es in enumerate(eps_ss):
                Rt_list = []
                time_to_below_list = []
                Itot_list = []
                peak_Is_list = []
                extinction_time_list = []

                ps = Params.for_SEIPAR(epsilon_s=float(es), epsilon_w=float(ew))
                alpha = 0.01
                Iest = np.ceil(np.log(alpha)/np.log(calculate_mt_branching_q(ps, ew, es)))

                for k in range(num_simulations):
                    tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
                    if (wildcards.scenario == 'establishment') & (np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest):
                        continue

                    Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = outcome_metrics(tt, yy, Params.for_SEIPAR(epsilon_s=es, epsilon_w=ew), t1, population_size=N)
                    Rt_list.append(Rt)
                    time_to_below_list.append(time_to_below)
                    Itot_list.append(Itot)
                    peak_Is_list.append(peak_Is)
                    extinction_time_list.append(extinction_time)

                Rt_grid[i,j] = np.mean(Rt_list)
                Rt_var_grid[i,j] = np.var(Rt_list)
                time_to_below_grid[i,j] = np.mean(time_to_below_list)
                time_to_below_var_grid[i,j] = np.var(time_to_below_list)
                Itot_grid[i,j] = np.mean(Itot_list)
                Itot_var_grid[i,j] = np.var(Itot_list)
                peak_Is_grid[i,j] = np.mean(peak_Is_list)
                peak_Is_var_grid[i,j] = np.var(peak_Is_list)
                percentile_95 = np.nan
                try: percentile_95 = np.percentile(extinction_time_list, 95)
                except: pass
                extinction_time_grid[i,j] = percentile_95
                extinction_time_var_grid[i,j] = np.var(extinction_time_list)
        
        np.savez_compressed(
            output.npz,
            Rt_grid=Rt_grid, Rt_var_grid=Rt_var_grid,
            time_to_below_grid=time_to_below_grid, time_to_below_var_grid=time_to_below_var_grid,
            Itot_grid=Itot_grid, Itot_var_grid=Itot_var_grid,
            peak_Is_grid=peak_Is_grid, peak_Is_var_grid=peak_Is_var_grid,
            extinction_time_grid=extinction_time_grid, extinction_time_var_grid=extinction_time_var_grid
        )

rule plot_stochastic_intervention_grid:
    input:
        npz="{outdir}/gillespie/{scenario}_stochastic_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    output:
        plot="{outdir}/gillespie/stochastic_{scenario}_pathogen{pathogen}_N{N}_sims{num_simulations}_res{resolution}_outcome{metric}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        res = int(wildcards.resolution)
        metric = wildcards.metric
        eps_ww = np.linspace(0.0, 0.999, res)
        eps_ss = np.linspace(0.0, 0.999, res)
        data = np.load(input.npz)[f"{metric}_grid"]
        
        fig, ax = plot_heatmap(
            eps_ww, eps_ss, data.T, 
            cmap='magma' if metric.startswith('extinction_time') else 'RdBu_r' if metric == 'Rt' else 'viridis', 
            norm=CenteredNorm(vcenter=1.0) if metric == 'Rt' else None,
            contour_levels=[1.0], contour_colors='black' if metric == 'Rt' else 'white',
            contour_metric=compute_R_grid(models[wildcards.pathogen], parameters[wildcards.pathogen]._replace(k=1), eps_ww, eps_ss, Rt_times[wildcards.pathogen], E0=1/N), 
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title={"Rt": "Average Final $R_t$", "Rt_var": "Variance of Final $R_t$", "time_to_below": "Average Time to $R_t < 1$", "time_to_below_var": "Variance of Time to $R_t < 1$", "Itot": "Average Proportion Infected", "Itot_var": "Variance of Proportion Infected", "peak_Is": "Average Peak Symptomatic Proportion", "peak_Is_var": "Variance of Peak Symptomatic Proportion", "extinction_time": "95th Percentile Extinction Time", "extinction_time_var": "Variance of Extinction Time"}.get(metric, metric)
        )
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)

rule plot_stochastic_cumulative_extinction_probability:
    output:
        plot ="{outdir}/gillespie/cumulative_extinction_probability_{pathogen}_N{N}_epsS_{eps_s}_epsW{eps_w}_combined.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = float(wildcards.N)
        eps_s = float(wildcards.eps_s)
        eps_w = float(wildcards.eps_w)
        t1 = 5000.0
        num_simulations = 1000
        initial_fadeout_times = []
        established_extinction_times = []

        ps = Params.for_SEIPAR(epsilon_s=eps_s, epsilon_w=eps_w)
        alpha = 0.01
        Iest = np.ceil(np.log(alpha)/np.log(calculate_mt_branching_q(ps, eps_w, eps_s)))

        # simulations
        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
            I = yy[:,2] + yy[:,3] + yy[:,4]
            if np.max(I) < Iest:
                initial_fadeout_times.append(tt[-1])
            else:
                established_extinction_times.append(tt[-1])
        extinction_times_est = np.array(established_extinction_times)
        extinction_times_all = np.concatenate([established_extinction_times, initial_fadeout_times])

        # deterministic susceptible trajectory
        model = models[wildcards.pathogen]
        ps_det = parameters[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w)
        tt_det, yy_det = model(params=ps_det, t1=t1, E0=1/N)
        S_det = yy_det.T[0]

        # subplots with shared x axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10), sharex=True)

        # global plotting
        plot_extinction_probability_scenario(ax1, extinction_times_all, "All introductions", tt_det, S_det)
        plot_extinction_probability_scenario(ax2, extinction_times_est, "Established outbreaks", tt_det, S_det)
        ax2.set_xlabel('days', fontsize=12)
        if len(extinction_times_all) > 0: ax2.set_xlim(-50, max(extinction_times_all))
        plt.suptitle(f'Cumulative Extinction Probability ({wildcards.pathogen}, $\\varepsilon_s={eps_s}$, $\\varepsilon_w={eps_w}$)', fontsize=14, y=0.98)
        fig.tight_layout()
        plt.savefig(output.plot, dpi=image_resolution); plt.close()


### SUPERSPREADING ###
rule plot_superspreading_baseline_trajectories:
    output:
        plot ="{outdir}/gillespie/superspreading_baseline_trajectories_{pathogen}_N{N}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        t1 = 1000.0
        num_simulations = 100
        ps = Params.for_SEIPAR()
        E0 = 1/N
        alpha = 0.01
        q = calculate_mt_branching_q_with_superspreading(k_sc2,ps,0,0)
        Iest = np.ceil(np.log(alpha)/np.log(q))

        fig, (ax_traj, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=(6,6), sharex=True, height_ratios=[2,1])

        initial_fadeout_times = []
        established_extinction_times = []

        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=k_sc2, a_ss=True, p_ss=True, s_ss=True)
            initial_fadeout = np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest
            if initial_fadeout: initial_fadeout_times.append((tt[-1]))
            else: established_extinction_times.append((tt[-1]))
            ax_traj.plot(tt, yy.T[0], alpha=0.05, color='grey' if initial_fadeout else colors[wildcards.pathogen])

        try: 
            ymax = max(max(initial_fadeout_times, default=0), max(established_extinction_times, default=0))
            if ymax > 0:
                ax_traj.set_xlim([0, ymax])
                num_bins = max(1, int(ymax // 10))
                ax_hist.hist(
                    [initial_fadeout_times, established_extinction_times], 
                    density=True, stacked=True, color=['grey', colors[wildcards.pathogen]], bins=num_bins
                )
        except Exception as e: 
            print(e)
        tt_det, yy_det = models[wildcards.pathogen](params=ps, t1=t1, E0=E0)
        S_det = yy_det.T[0] * N
        ax_traj.plot(tt_det, S_det, color=colors[wildcards.pathogen])

        ax_traj.set_title('Number of susceptibles')
        ax_hist.set_title('Extinction times')
        ax_hist.set_xlabel('days')
        fig.suptitle(f'Stochastic susceptible trajectories ({wildcards.pathogen}, N={N})')
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


rule plot_stochastic_cumulative_extinction_probability_superspreading:
    output:
        plot ="{outdir}/gillespie/superspreading_cumulative_extinction_probability_{pathogen}_N{N}_epsS_{eps_s}_epsW{eps_w}_scenario_{scenario}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = float(wildcards.N)
        eps_s = float(wildcards.eps_s)
        eps_w = float(wildcards.eps_w)
        t1 = 5000.0
        num_simulations = 1000
        initial_fadeout_times = []
        established_extinction_times = []

        ps = Params.for_SEIPAR(epsilon_s=eps_s, epsilon_w=eps_w)

        alpha = 0.01
        Iest = np.ceil(np.log(alpha)/np.log(calculate_mt_branching_q_with_superspreading(k_sc2, ps, eps_w, eps_s)))

        # simulations
        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=k_sc2, a_ss=True, p_ss=True, s_ss=True)
            I = yy[:,2] + yy[:,3] + yy[:,4]
            if np.max(I) < Iest:
                initial_fadeout_times.append((tt[-1]))
            else:
                established_extinction_times.append((tt[-1]))

        if wildcards.scenario == 'establishment':
            extinction_times = established_extinction_times
        else:
            extinction_times = np.concatenate([established_extinction_times, initial_fadeout_times])

        # cumulative extinction times and CIs
        sorted_times = np.sort(extinction_times)
        cumulative_prob = np.arange(1, sorted_times.shape[0]+1) / sorted_times.shape[0]
        z_score = 1.96
        std_error = np.sqrt(cumulative_prob * (1-cumulative_prob) / sorted_times.shape[0])
        ci_lower = np.maximum(0, cumulative_prob - z_score*std_error)
        ci_upper = np.minimum(1, cumulative_prob + z_score*std_error)

        # plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.step(sorted_times, cumulative_prob, where='post', label='Cumulative extinction probability', color='blue', linewidth=2)
        ax1.fill_between(sorted_times, ci_lower, ci_upper, step='post', color='blue', alpha=0.25)

        # median time
        median_time = np.median(extinction_times)
        median_time_ci_lower = sorted_times[np.argmax(ci_upper >= 0.5)]
        median_time_ci_upper = sorted_times[np.argmax(ci_lower >= 0.5)]
        ax1.axvline(median_time, color='red', label=f'Median: {median_time:.2f} [{median_time_ci_lower:.2f}, {median_time_ci_upper:.2f}]')
        ax1.axvspan(median_time_ci_lower, median_time_ci_upper, color='red', alpha=0.2)

        # 95% time
        time_95 = np.percentile(extinction_times, 95)
        time_95_ci_lower = sorted_times[np.argmax(ci_upper >= 0.95)]
        time_95_ci_upper = sorted_times[np.argmax(ci_lower >= 0.95)]
        ax1.axvline(time_95, color='orange', label=f'95%: {time_95:.2f} [{time_95_ci_lower:.2f}, {time_95_ci_upper:.2f}]')
        ax1.axvspan(time_95_ci_lower, time_95_ci_upper, color='orange', alpha=0.2)

        # deterministic susceptible trajectory
        model = models[wildcards.pathogen]
        ps = parameters[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w)
        tt, yy = model(params=ps, t1=t1, E0=1/N)
        ax1.plot(tt, yy.T[0], color='green', label='Deterministic susceptible trajectory')

        # histogram
        ax2 = ax1.twinx()
        ax2.hist(extinction_times, bins=100, density=True, color='gray', alpha=0.3, label='Extinction times histogram')
        ax2.set_ylabel('Density', color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')

        # styling
        plt.title(f'Cumulative extinction probability ({wildcards.pathogen}, $k={k_sc2}$, $\\varepsilon_s={eps_s}$, $\\varepsilon_w={eps_w}$, {wildcards.scenario})', fontsize=14)
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(-50, max(extinction_times))
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1+lines_2, labels_1+labels_2, loc='best')
        ax1.grid(True, alpha=0.5)
        fig.tight_layout()
        plt.savefig(output.plot, dpi=image_resolution); plt.close()

rule simulate_superspreading_outcomes:
    output:
        npz="{outdir}/gillespie/{scenario}_superspreading_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    run:
        os.makedirs(os.path.dirname(output.npz), exist_ok=True)
        res = int(wildcards.resolution)
        simulate_superspreading_outcomes(
            eps_ww = np.linspace(0.0, 0.999, res), 
            kk = np.logspace(-4, 4, res), 
            eps_s = 0.5, 
            t1 = 2000.0,
            N = int(wildcards.N),
            num_simulations = int(wildcards.num_simulations), 
            scenario = wildcards.scenario, 
            npz=output.npz
        )


rule plot_superspreading_intervention_grid:
    input:
        npz="{outdir}/gillespie/{scenario}_superspreading_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    output:
        plot="{outdir}/gillespie/superspreading_{scenario}_pathogen{pathogen}_N{N}_sims{num_simulations}_res{resolution}_outcome{metric}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        res = int(wildcards.resolution)
        metric = wildcards.metric
        es = 0.5
        eps_ww = np.linspace(0.0, 0.999, res)
        kk = np.linspace(0.01, 1.0, res)
        data = np.load(input.npz)[f"{metric}_grid"]
        
        fig, ax = plot_heatmap(
            eps_ww, kk, data.T, 
            cmap='magma' if metric.startswith('extinction_time') else 'RdBu_r' if metric == 'Rt' else 'viridis', 
            norm=CenteredNorm(vcenter=1.0) if metric == 'Rt' else None,
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Dispersion parameter $r$',
            title={"Rt": "Average Final $R_t$", "Rt_var": "Variance of Final $R_t$", "time_to_below": "Average Time to $R_t < 1$", "time_to_below_var": "Variance of Time to $R_t < 1$", "Itot": "Average Proportion Infected", "Itot_var": "Variance of Proportion Infected", "peak_Is": "Average Peak Symptomatic Proportion", "peak_Is_var": "Variance of Peak Symptomatic Proportion", "extinction_time": "95th Percentile Extinction Time", "extinction_time_var": "Variance of Extinction Time"}.get(metric, metric)
        )
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


rule simulate_superspreading_outcomes_all_superspreading:
    output:
        npz="{outdir}/gillespie/{scenario}_superspreading_all_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    run:
        os.makedirs(os.path.dirname(output.npz), exist_ok=True)
        num_simulations = int(wildcards.num_simulations)
        N = int(wildcards.N)
        res = int(wildcards.resolution)
        t1 = 2000.0

        es = 0.5
        eps_ww = np.linspace(0.0, 0.999, res)
        kk = np.linspace(0.01, 1.0, res)

        Rt_grid = np.zeros((len(eps_ww), len(kk)))
        Rt_var_grid = np.zeros((len(eps_ww), len(kk)))
        time_to_below_grid = np.zeros((len(eps_ww), len(kk)))
        time_to_below_var_grid = np.zeros((len(eps_ww), len(kk)))
        Itot_grid = np.zeros((len(eps_ww), len(kk)))
        Itot_var_grid = np.zeros((len(eps_ww), len(kk)))
        peak_Is_grid = np.zeros((len(eps_ww), len(kk)))
        peak_Is_var_grid = np.zeros((len(eps_ww), len(kk)))
        extinction_time_grid = np.zeros((len(eps_ww), len(kk)))
        extinction_time_var_grid = np.zeros((len(eps_ww), len(kk)))

        for i, ew in enumerate(eps_ww):
            for j, k_ss in enumerate(kk):
                Rt_list = []
                time_to_below_list = []
                Itot_list = []
                peak_Is_list = []
                extinction_time_list = []

                ps = Params.for_SEIPAR(epsilon_s=float(es), epsilon_w=float(ew))
                alpha = 0.01
                Iest = np.ceil(np.log(alpha)/np.log(calculate_mt_branching_q_with_superspreading(k_ss, ps, ew, es)))

                for _ in range(num_simulations):
                    tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=k_ss, a_ss=True, p_ss=True, s_ss=False)
                    if (wildcards.scenario == 'establishment') & (np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest):
                        continue

                    Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = outcome_metrics(tt, yy, Params.for_SEIPAR(epsilon_s=es, epsilon_w=ew), t1, population_size=N)
                    Rt_list.append(Rt)
                    time_to_below_list.append(time_to_below)
                    Itot_list.append(Itot)
                    peak_Is_list.append(peak_Is)
                    extinction_time_list.append(extinction_time)

                Rt_grid[i,j] = np.mean(Rt_list)
                Rt_var_grid[i,j] = np.var(Rt_list)
                time_to_below_grid[i,j] = np.mean(time_to_below_list)
                time_to_below_var_grid[i,j] = np.var(time_to_below_list)
                Itot_grid[i,j] = np.mean(Itot_list)
                Itot_var_grid[i,j] = np.var(Itot_list)
                peak_Is_grid[i,j] = np.mean(peak_Is_list)
                peak_Is_var_grid[i,j] = np.var(peak_Is_list)
                percentile_95 = np.nan
                try: percentile_95 = np.percentile(extinction_time_list, 95)
                except: pass
                extinction_time_grid[i,j] = percentile_95
                extinction_time_var_grid[i,j] = np.var(extinction_time_list)
        np.savez_compressed(output.npz, Rt_grid=Rt_grid, Rt_var_grid=Rt_var_grid, time_to_below_grid=time_to_below_grid, time_to_below_var_grid=time_to_below_var_grid, Itot_grid=Itot_grid, Itot_var_grid=Itot_var_grid, peak_Is_grid=peak_Is_grid, peak_Is_var_grid=peak_Is_var_grid, extinction_time_grid=extinction_time_grid, extinction_time_var_grid=extinction_time_var_grid)

rule plot_superspreading_intervention_grid_all_superspreading:
    input:
        npz="{outdir}/gillespie/{scenario}_superspreading_all_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    output:
        plot="{outdir}/gillespie/ss_all_{scenario}_pathogen{pathogen}_N{N}_sims{num_simulations}_res{resolution}_outcome{metric}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        res = int(wildcards.resolution)
        metric = wildcards.metric
        es = 0.5
        eps_ww = np.linspace(0.0, 0.999, res)
        kk = np.linspace(0.01, 1.0, res)
        data = np.load(input.npz)[f"{metric}_grid"]
        
        fig, ax = plot_heatmap(
            eps_ww, kk, data.T, 
            cmap='magma' if metric.startswith('extinction_time') else 'RdBu_r' if metric == 'Rt' else 'viridis', 
            norm=CenteredNorm(vcenter=1.0) if metric == 'Rt' else None,
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Dispersion parameter $r$',
            title={"Rt": "Average Final $R_t$", "Rt_var": "Variance of Final $R_t$", "time_to_below": "Average Time to $R_t < 1$", "time_to_below_var": "Variance of Time to $R_t < 1$", "Itot": "Average Proportion Infected", "Itot_var": "Variance of Proportion Infected", "peak_Is": "Average Peak Symptomatic Proportion", "peak_Is_var": "Variance of Peak Symptomatic Proportion", "extinction_time": "95th Percentile Extinction Time", "extinction_time_var": "Variance of Extinction Time"}.get(metric, metric))
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


### ALTERNATIVE WARNING SYSTEMS
rule plot_alternative_warning_strategies_eps_w:
    output:
        plot="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_epsW.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)

        pathogen = wildcards.pathogen
        model = models_piecewise[pathogen]
        t1 = 10000.0
        eps_ww = np.linspace(0.0, 1.0, 100)
        eps_s = 0.5 if pathogen=="SARS-CoV-2" else 0.0
        nE = len(eps_ww)
        nM = len(METRIC_NAMES)
        base_params = parameters[pathogen].update(epsilon_s=eps_s, R_off=R_OFF, eval_interval=EVAL_INTERVAL)
        baseline_metrics = strategy_metrics(model, base_params, t1, False, False, 1.0)
        baseline = np.array([1.0, baseline_metrics[1], baseline_metrics[2], 1.0, t1, t1])

        data = {s: np.zeros((nM, nE)) for s in STRATEGIES}
        for s, (asym, disc, tl, ci) in STRATEGIES.items():
            for i, ew in enumerate(eps_ww):
                ps = base_params.update(epsilon_w=float(ew), T_lead=tl)
                data[s][:, i] = strategy_metrics(model, ps, t1, asym, disc, ci)

        strat_colors = dict(zip(STRATEGIES, sns.color_palette("colorblind", len(STRATEGIES))))
        linestyles = dict(zip(STRATEGIES, ['-','--', '-.', ':']))
        fig, axs = plt.subplots(ncols=nM, nrows=1, figsize=(20, 4), sharex=True)
        for r in range(nM):
            ax = axs[r]
            for s in STRATEGIES: 
                y = data[s][r] / baseline[r]
                if r==3 and s=="asymmetric":
                    print(np.argmax(y > 0.1))
                ax.plot(eps_ww, y, linestyle=linestyles[s], label=s, color=strat_colors[s], lw=2, alpha=0.8)
            if r==0: ax.axhline(1.0, c='k', lw=1)
            ax.set_title(METRIC_NAMES[r], fontsize=12)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel(r"$\varepsilon_w$", fontsize=16)
        axs[0].legend()
        fig.suptitle(f"Warning strategies for varying response strengths $\\varepsilon_w$ ({pathogen}, $\\varepsilon_s={eps_s}$)", fontsize=16, y=0.95)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches="tight"); plt.close(fig)


rule compute_alternative_warning_strategies_grid:
    output:
        data="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_k{k}_epsW{eps_w}_epsS{eps_s}.npz",
    run:
        os.makedirs(os.path.dirname(output.data), exist_ok=True)
        pathogen = wildcards.pathogen
        data, taus_W, taus_B = strategy_grid(
            model=models_piecewise[pathogen], base_params=parameters[pathogen], k=float(wildcards.k), 
            eps_w=float(wildcards.eps_w), eps_s=float(wildcards.eps_s), strategies=STRATEGIES,
            t1=300.0, taus_W=np.linspace(1.0, 30.0, 100), taus_B=np.linspace(1.0, 30.0, 100), R_off=0.8, eval_interval=EVAL_INTERVAL,
        )
        strategies = list(STRATEGIES)
        np.savez_compressed(output.data, grid=np.stack([data[s] for s in strategies]), taus_W=np.asarray(taus_W), taus_B=np.asarray(taus_B), strategies=np.asarray(strategies), metrics=np.asarray(METRIC_NAMES))

rule alternative_warning_strategies_table:
    output:
        tex="{outdir}/compartmental/alternative_warning_strategies_table_{pathogen}.tex",
    run:
        os.makedirs(os.path.dirname(output.tex), exist_ok=True)
        t1 = 10_000.0
        pathogen = wildcards.pathogen
        ps = parameters[pathogen].update(R_off=R_OFF, eval_interval=EVAL_INTERVAL)
        model = models_piecewise[pathogen]
        warning_scenarios = {
            "SARS-CoV-2": [("baseline", 0.00, 0.00), ("uncontrolled", 0.50, 0.40), ("barely controlled", 0.50, 0.80), ("controlled", 0.50, 1.00)],
            "Influenza A": [("baseline", 0.00, 0.00), ("uncontrolled", 0.00, 0.40), ("barely controlled", 0.00, 0.80), ("controlled", 0.00, 1.00)],
            "Ebola": [("baseline", 0.00, 0.00), ("uncontrolled", 0.00, 0.40), ("barely controlled", 0.00, 0.80), ("controlled", 0.00, 1.00)],
        }
        scenarios = warning_scenarios[pathogen]
        
        rows = {s: [] for s,_,_ in scenarios}
        base = table_row_metrics(ps, model, 0.0, 0.0, t1)
        base["prevented"] = 0.0
        for scenario, eps_s, eps_w in scenarios:
            for strategy in list(STRATEGIES):
                strategy_params = ps.update(T_lead=7.0) if strategy=="lead" else ps
                m = (base if (eps_s == 0.0 and eps_w == 0.0) else table_row_metrics(strategy_params, model, eps_s, eps_w, t1, itot_baseline=base["itot"], strategy=strategy))
                rows[scenario].append((strategy, eps_s, eps_w, m))
        with open(output.tex, "w") as f:
            f.write("\\begin{table}[H]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{scenario} & $\\mathcal{R}_t$ & \\textbf{peak sympt.} & \\textbf{time to peak} & \\textbf{total wave time} & \\textbf{attack rate} & \\textbf{infections prevented} & \\textbf{isolation cost} & \\textbf{warning cost}\\\\\n\\midrule\n")
            for i, (scenario, eps_s, eps_w) in enumerate(scenarios):
                f.write(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{table_scenario_label(scenario, eps_s, eps_w, bold=True)}}}}}\\\\\n")
                for strategy, _, _, m in rows[scenario]:
                    tp, wt = f_days(m)
                    # f.write(" & ".join([f"\\quad {strategy}", f"{m['Rt']}", f"{m['peak_Is']}", tp, wt, f"{m['itot']}", f"{m['prevented']}", f"{m['isolation_cost']}", f"{m['warning_cost']}"]) + " \\\\\n")
                    f.write(" & ".join([f"\\quad {strategy}", f"{m['Rt']:.2f}", f_pct(m["peak_Is"], 1), tp, wt, f_pct(m["itot"], 0), f_pct(m["prevented"], 0), f"{m['isolation_cost']:.1f}", f"{m['warning_cost']:.1f}",]) + " \\\\\n")
                if i < len(scenarios)-1: f.write("\\midrule\n")
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption[Characteristics of epidemic scenarios under different warning strategies]{Characteristics of epidemic scenarios under different warning strategies}\\label{tab:alternative_strategies}\n\\end{table}\n")


rule plot_alternative_warning_strategies:
    input:
        data="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_k{k}_epsW{eps_w}_epsS{eps_s}.npz",
    output:
        plot="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_k{k}_epsW{eps_w}_epsS{eps_s}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        k = float(wildcards.k); eps_w = float(wildcards.eps_w); eps_s = float(wildcards.eps_s)
        t1 = 300.0

        npz = np.load(input.data)
        grid = npz["grid"]
        taus_W = npz["taus_W"]
        taus_B = npz["taus_B"]
        strategies = [str(s) for s in npz["strategies"]]
        metric_names = [str(m) for m in npz["metrics"]]
        nS = len(strategies)
        nM = len(metric_names)
        nrows = nM + 1

        model = models_piecewise[wildcards.pathogen]
        base = parameters[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w, k=k, R_off=0.8, eval_interval=EVAL_INTERVAL)

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig = plt.figure(figsize=(3.05 * nS + 1.4, 3.05 * nrows))
        gs = fig.add_gridspec(nrows, nS + 2, width_ratios=[1.0] * nS + [0.16, 0.16], height_ratios=[1.0] * nM + [1.0], hspace=0.18, wspace=0.12)
        axs = np.empty((nrows, nS), dtype=object)
        for r in range(nrows):
            for c in range(nS):
                axs[r, c] = fig.add_subplot(gs[r, c])
                axs[r, c].set_box_aspect(1)

        # heatmaps
        ims = [None] * nM
        for c, s in enumerate(strategies):
            for r in range(nM):
                ax = axs[r, c]
                vmin, vmax = METRIC_BOUNDS[r] 
                im = ax.imshow(grid[c, r], origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, extent=[taus_B[0], taus_B[-1], taus_W[0], taus_W[-1]])
                if r == 0: ax.set_title(strategies[c], fontsize=10)
                if c == 0:
                    ims[r] = im
                    ax.set_ylabel(f"{metric_names[r]}\n$\\tau_W$", fontsize=12)
                else: ax.tick_params(labelleft=False)
                if r == nM - 1: ax.set_xlabel("$\\tau_B$")
                else: ax.tick_params(labelbottom=False)

        # oscillations
        for c, s in enumerate(strategies):
            asym, disc, tl, ci = STRATEGIES[s]
            params = base.update(tau_W=14.0, tau_B=7.0, T_lead=tl)
            tt, yy, mm = model(params=params, t1=t1, asymmetric=asym, discrete_eval=disc, check_interval=ci)
            rt_true = params.R_0 * params.rho * yy[:, -1] * yy[:, 0]
            rt_reported = yy[:, -(params.n_B + 1)]
            above = (rt_reported >= params.R_crit).astype(jnp.float32)

            ax = axs[-1, c]
            ax.plot(tt, rt_true, color="black")
            ax.plot(tt, rt_reported, color="red")
            ax.axhline(params.R_crit, color="grey", linestyle="--")
            ax.set_ylim(0, 1.75); ax.set_xlim(0, t1)
            if c == 0: ax.set_ylabel("true vs reported $R_t$\n($\\tau_W=14$, $\\tau_B=7$)", fontsize=11)
            ax.set_xlabel("Time (days)")
            ax.text(0.96, 0.94, f'{float(above.mean() * t1):.0f} days above $R_{{crit}}$\n{int(jnp.sum(jnp.diff(above) > 0))} warnings', transform=ax.transAxes, ha="right", va="top", fontsize=7)

        # legends
        for r in range(nM):
            cax = fig.add_subplot(gs[r, nS + 1])
            fig.colorbar(ims[r], cax=cax, orientation="vertical")
        lax = fig.add_subplot(gs[nrows - 1, nS + 1]); lax.axis("off")
        lax.legend([Line2D([0], [0], color="black", lw=2), Line2D([0], [0], color="red", lw=2)], ["True $R_t$", "Reported $R_t$"], loc="center", ncol=1, fontsize=11, frameon=False)
        fig.suptitle(f"Warning strategy comparison ({wildcards.pathogen}, $\\varepsilon_s={eps_s:g}$, $\\varepsilon_w={eps_w:g}$, $k={k:g}$)", fontsize=18, y=0.94)
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches="tight"); plt.close()

rule plot_oscillations:
    output:
        plot="{outdir}/compartmental/oscillations.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_s = 0.5
        eps_w = 0.8
        params = Params.for_SEIPAR().update(tau_W=14.0, tau_B=7.0, epsilon_s=eps_s, epsilon_w=eps_w)
        model = simulate_SEIPAR_W_piecewise
        t1 = 1000.0
        tt, yy, mm = model(params=params, t1=t1, asymmetric=True)
        rt_true = params.R_0 * params.rho * yy[:, -1] * yy[:, 0]
        rt_reported = yy[:, -(params.n_B + 1)]

        plt.plot(tt, rt_true, color="black")
        plt.plot(tt, rt_reported, color="red")
        plt.axhline(params.R_crit, color="grey", linestyle="--")
        plt.ylim(0, 1.75); plt.xlim(0, t1)
        plt.ylabel("true vs reported $R_t$\n($\\tau_W=14$, $\\tau_B=7$)", fontsize=11)
        plt.xlabel("Time (days)")
        plt.title(f"SARS-CoV-2 (asymmetric, $\\varepsilon_s=${eps_s}, $\\varepsilon_w=${eps_w})")
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches="tight"); plt.close()


rule plot_true_vs_reported_Rt_scenarios_piecewise:
    output:
        plot="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_scenarios_{scenario}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)

        k = float(wildcards.k)
        t1 = 300.0
        tau_W = 14.0
        tau_B = 7.0
        epsilon_s = 0.5
        eps_w_values = [0.0, 0.4, 0.8, 1.0]

        asymmetric = wildcards.scenario=="asymmetric"
        discrete_eval = wildcards.scenario=="interval"
        lead = wildcards.scenario=="lead"
        T_lead = 7.0 if lead else 0.0

        model = models_piecewise[wildcards.pathogen]

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=1, ncols=len(eps_w_values), figsize=(16,4), sharex=True, sharey=True)

        for j, eps_w in enumerate(eps_w_values):
            ps = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=eps_w, k=k, R_off=0.8, eval_interval=28.0, T_lead=T_lead)
            tt, yy, mm = model(params=ps, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval)
            rt_true = ps.R_0 * ps.rho * yy[:, -1] * yy[:, 0]
            rt_reported = yy[:, -(ps.n_B + 1)]
            above = (rt_reported >= ps.R_crit).astype(jnp.float32)
            total_time_above = float(above.mean() * t1)
            num_crossings = int(jnp.sum(jnp.diff(above) > 0))

            ax = axs[j]
            ax.set_title(f'$\\varepsilon_w={eps_w:g}$', fontsize=16)
            if j == 0: ax.set_ylabel('$R_t$', fontsize=16)
            ax.set_xlabel('time (days)', fontsize=12)
            ax.plot(tt, rt_true, color='black')
            ax.plot(tt, rt_reported, color='red')
            ax.axhline(ps.R_crit, color='grey', linestyle='--')
            ax.text(0.97, 0.95, f'{total_time_above:.0f} days above $R_{{crit}}$\n{num_crossings} warnings', transform=ax.transAxes, ha='right', va='top', fontsize=8)
        fig.suptitle(f'{wildcards.pathogen}: $k={k:g}$, {wildcards.scenario} ($\\tau_W={ps.tau_W:g}$, $\\tau_B={ps.tau_B:g}$, $\\varepsilon_s={epsilon_s:g}$)', fontsize=15, y=1.03,)
        fig.legend(
            [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='red', lw=2)],
            ['True $R_t$', 'Reported $R_t$'],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12), fontsize=14,
        )
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close()


rule plot_spatial_heatmaps:
    output:
        plot="{outdir}/spatial/spatial_heatmap_split_vs_migration.png",
    run:
        pathogen = "SARS-CoV-2"
        model_name = "SEIPAR"
        model = spatial_models[pathogen]
        response_in_B_to_A=True
        eps_s = 0.5
        eps_w = 0.8
        epi_params = parameters[pathogen].update(epsilon_w=eps_w, epsilon_s=eps_s)
        t1 = 1000.0
        E0 = 1e-6

        N_A_grid = jnp.linspace(0.01, 0.99, 100)
        m_grid = jnp.concatenate([jnp.array([0.0]), jnp.logspace(-6, 0, 100)])
        run_grid = jax.vmap(jax.vmap(partial(run_spatial, epi_params=epi_params, response_in_B_to_A=response_in_B_to_A, model=model_name), in_axes=(None, 0)), in_axes=(0, None))
        Itot_A, Itot_B, peak_Is_A, peak_Is_B, total_infections = run_grid(N_A_grid, m_grid)

        # baseline without ww
        ts_bl, ys_bl = model(SpatialParams(epi_params=epi_params.update(epsilon_w=0.0), N_A=1.0, m=0.0), t1=t1, E0=E0)
        d_ref = unpack_spatial(ys_bl, n_W=3, n_B=1, ww_in_B=False, model=model_name)
        itot_baseline = d_ref["R_A"][-1] + d_ref["R_B"][-1]

        def heatmap(ax, Z, title, cmap="viridis", vmin=None, vmax=None):
            im = ax.pcolormesh(np.array(m_grid), np.array(N_A_grid), np.array(Z), cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
            ax.set_xscale("symlog", linthresh=1e-4)
            ax.set_xlabel("migration rate"); ax.set_ylabel("population fraction in A")
            ax.set_title(title, fontsize=11)
            return im

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        im0 = heatmap(axs[0], Itot_B, "Infections in unsurveilled deme B", vmin=0.0, vmax=1.0)
        fig.colorbar(im0, ax=axs[0])
        im1 = heatmap(axs[1], Itot_A, "Infections in surveilled deme A", vmin=0.0, vmax=1.0)
        fig.colorbar(im1, ax=axs[1])
        im2 = heatmap(axs[2], 1.0 - (np.array(total_infections)/itot_baseline), "Total reduction", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        fig.colorbar(im2, ax=axs[2])
        fig.suptitle(f"Spatial model ({pathogen}, $\\varepsilon_s={epi_params.epsilon_s}$, $\\varepsilon_w={epi_params.epsilon_w}$, $\\tau_W={epi_params.tau_W}$, $\\tau_B={epi_params.tau_B:g}$)", y=1.00, fontsize=16)
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches="tight")
        plt.close()

rule plot_spatial_trajectories:
    output:
        migration="{outdir}/spatial/spatial_trajectories_migration.png",
        split="{outdir}/spatial/spatial_trajectories_split.png",
    run:
        pathogen = "SARS-CoV-2"
        model_name = "SEIPAR"
        model = spatial_models[pathogen]
        epi_params = parameters[pathogen].update(epsilon_w=0.4)
        t1 = 1000.0
        E0 = 1e-6

        # effect of migration rate at fixed 50/50 split
        m_values = [0.0, 0.001, 0.01, 0.05]
        N_A = 0.5
        fig, axs = plt.subplots(1, len(m_values), figsize=(4*len(m_values), 3.5), sharey=True)
        for ax, m in zip(axs, m_values):
            sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
            ts, ys = model(sp, t1=t1, E0=E0, primary_in_A=False)
            d = unpack_spatial(ys, model=model_name)
            ax.plot(ts, d["Is_A"] / N_A, label="surveilled deme A", color="blue", alpha=0.7)
            ax.plot(ts, d["Is_B"] / (1-N_A), label="unsurveilled deme B", color="red", alpha=0.7)
            ax.set_title(f"m = {m}/day")
            ax.set_xlabel("time (days)")
        axs[0].set_ylabel("symptomatic prevalence")
        axs[0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Spatial model: effect of migration rate ($N_A=N_B=0.5$)", y=1.04, fontsize=16)
        plt.savefig(f"{output.migration}", dpi=image_resolution, bbox_inches="tight")
        plt.close()

        # effect of population split at fixed migration rate
        N_A_values = [0.0, 0.1, 0.5, 0.9]
        m = 0.01
        fig, axs = plt.subplots(1, len(m_values), figsize=(4*len(m_values), 3.5), sharey=True)
        for ax, N_A in zip(axs, N_A_values):
            sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
            ts, ys = model(sp, t1=t1, E0=E0, primary_in_A=False)
            d = unpack_spatial(ys, model=model_name)
            ax.plot(ts, d["Is_A"] / N_A, label="surveilled deme A", color="blue", alpha=0.7)
            ax.plot(ts, d["Is_B"] / (1-N_A), label="unsurveilled deme B", color="red", alpha=0.7)
            ax.set_title(f"$N_A$ = {N_A}")
            ax.set_xlabel("time (days)")
        axs[0].set_ylabel("symptomatic prevalence")
        axs[0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Spatial model: effect of population split (m = {m}/day)", y=1.04, fontsize=16)
        plt.savefig(f"{output.split}", dpi=image_resolution, bbox_inches="tight")
        plt.close()

rule plot_Rt_spatial:
    output:
        plot="{outdir}/spatial/Rt_spatial.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        pathogen = "SARS-CoV-2"
        model_name = "SEIPAR"
        model = spatial_models[pathogen]
        epi_params = parameters[pathogen].update(epsilon_w=0.4)
        t1 = 700.0
        m_values = [0.0, 0.001, 0.01]
        N_A_values = [0.0, 0.1, 0.5, 0.9]
        response_in_B_to_A = True

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=len(m_values), ncols=len(N_A_values), figsize=(12, 8), sharex=True, sharey=True)

        for i, m in enumerate(m_values):
            for j, N_A in enumerate(N_A_values):
                spatial_params = SpatialParams(epi_params=epi_params, m=m, N_A=N_A)
                tt, yy = model(spatial_params=spatial_params, t1=t1, primary_in_A=False, response_in_B_to_A=response_in_B_to_A)
                c = unpack_spatial(yy, model=model_name, response_in_B_to_A=response_in_B_to_A)
                N_B = 1.0 - N_A
                rt_A = jnp.where(N_A > 0.0, epi_params.R_0 * epi_params.rho * c["B_A"][:, -1] * c["S_A"] / N_A, jnp.nan)
                rt_B = jnp.where(N_B > 0.0, epi_params.R_0 * epi_params.rho * c["B_B"][:, -1] * c["S_B"] / N_B, jnp.nan)
                rt_reported = c["W_A"][:, -1]

                ax = axs[i,j]
                if j == 0: ax.set_ylabel(f'$m={m}$', fontsize=16)
                if i == 0: ax.set_title(f'$N_A={N_A}$', fontsize=16)
                ax.set_xticks([])
                ax.plot(tt, rt_A, color='blue', alpha=0.8)
                ax.plot(tt, rt_B, color='red', alpha=0.8)
                ax.plot(tt, rt_reported, color='black', alpha=0.8)
                ax.axhline(epi_params.R_crit, color='grey', linestyle='--')

        fig.legend(
            [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='black', lw=2)],
            ['True $R_t$ in A', 'True $R_t$ in B', 'Reported $R_t$'],
            loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=16,
        )
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close()

rule plot_Rt_divergence_heatmap:
    output:
        plot="{outdir}/spatial/spatial_heatmap_Rt_divergence_{pathogen}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        pathogen = wildcards.pathogen
        model_name = "SEIPAR" if pathogen=="SARS-CoV-2" else "SEIAR" if pathogen=="Influenza A" else "SEIR"
        model = spatial_models[pathogen]
        epi_params = parameters[pathogen].update(epsilon_w=0.4)
        t1 = 1000.0
        E0 = 1e-6
        response_in_B_to_A = True

        N_A_grid = jnp.linspace(0.01, 0.99, 100)
        m_grid = jnp.logspace(-6, 0, 100)

        def rt_diff(N_A, m):
            sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
            _, yy = model(sp, t1=t1, E0=E0, primary_in_A=False, response_in_B_to_A=response_in_B_to_A)
            c = unpack_spatial(yy, model=model_name, response_in_B_to_A=response_in_B_to_A)
            N_B = 1.0 - N_A
            rt_A = epi_params.R_0 * epi_params.rho * c["B_A"][:, -1] * c["S_A"] / N_A
            rt_B = epi_params.R_0 * epi_params.rho * c["B_B"][:, -1] * c["S_B"] / N_B
            return jnp.mean(jnp.abs(rt_A - rt_B))

        run_grid = jax.vmap(jax.vmap(rt_diff, in_axes=(None, 0)), in_axes=(0, None))
        Z = np.array(run_grid(N_A_grid, m_grid))

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, ax = plt.subplots(figsize=(6.5, 5))
        im = ax.pcolormesh(np.array(m_grid), np.array(N_A_grid), Z, cmap="magma", shading="auto", vmin=0.0)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("migration rate")
        ax.set_ylabel("population fraction in A")
        ax.set_title("Mean difference in $\\mathcal{R}_t$ between demes", fontsize=12)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("average $\\mathcal{R}_t$ difference")
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches="tight")
        plt.close()


rule plot_infectiousness_distributions:
    output:
        plot="{outdir}/compartmental/infectiousness_distributions.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        ps = parameters["SARS-CoV-2"]
        nE, nP, nS, nA = 10,10,10,10

        def generation_time(nE, nP, nS, w_p=None, w_s=None):
            def _infected_subsystem(t, y):
                E, Ip, Is = y[0:0+nE], y[nE:nE+nP], y[nE+nP:nE+nP+nS]
                dE, E_out = linear_chain(X=E, inflow=0.0, rate=nE/ps.gamma_inv)
                dIp, Ip_out = linear_chain(X=Ip, inflow=(1.0-ps.p)*E_out, rate=nP/ps.sigma_inv)
                dIs, Is_out = linear_chain(X=Is, inflow=Ip_out, rate=nS/ps.mu_s_inv)
                return np.concatenate([dE, dIp, dIs])

            sol = solve_ivp(_infected_subsystem, [0, tt[-1]], y0=np.concatenate([[1.0], np.zeros(nE+nP+nS-1)]), t_eval=tt, rtol=1e-10, atol=1e-13)
            Ip, Is = sol.y[nE:nE+nP], sol.y[nE+nP:nE+nP+nS]

            if w_p is None: b = Ip.sum(0) + Is.sum(0)
            else: b = (w_p[:, None] * Ip).sum(0) + (w_s[:, None] * Is).sum(0)
            gt = b / np.trapezoid(b, tt)
            mean = np.trapezoid(tt*gt, tt)
            return gt, mean

        tt = np.linspace(0, 1000, 100_000)
        colors = sns.color_palette("colorblind", 4)
        plt.figure(figsize=(8, 4))

        g_flat, m_flat = generation_time(1, 1, 1)
        plt.plot(tt, g_flat, lw=2.2, color=colors[1], label=f'SEIPAR (mean {m_flat:.1f})')
        plt.fill_between(tt, g_flat, color=colors[1], alpha=0.2)

        g_chain_flat, m_chain_flat = generation_time(nE, nP, nS)
        plt.plot(tt, g_chain_flat, lw=2.2, color=colors[2], label=f'SEIPAR-LCT flat (mean {m_chain_flat:.1f})')
        plt.fill_between(tt, g_chain_flat, color=colors[2], alpha=0.15)

        shape = 8
        scale = 5.5/8.0
        w_p, w_s = compute_weights(ps.gamma_inv, ps.sigma_inv, ps.mu_s_inv, shape, scale, nP, nS)
        
        g_weighted, m_weighted = generation_time(nE, nP, nS, w_p, w_s)
        plt.plot(tt, g_weighted, lw=2.2, color=colors[0], label=f'SEIPAR-LCT weighted (mean {m_weighted:.1f})')
        plt.fill_between(tt, g_weighted, color=colors[0], alpha=0.2)

        # Empirical Erlang-8 approximation
        mean_generation_time = 5.5
        pdf = erlang.pdf(tt, shape, scale=mean_generation_time/shape)
        plt.plot(tt, pdf, label=f'Erlang-{shape} (mean {mean_generation_time})', color='black')
        plt.fill_between(tt, pdf, alpha=0.2, color='black')

        # mean periods
        plt.axvline(3.0, ymin=0, c='k', alpha=0.2)
        plt.axvline(5.5, ymin=0, c='k', alpha=0.2)
        plt.axvline(14.8, ymin=0, c='k', alpha=0.2)
        plt.axvspan(3.0, 5.5, color='skyblue', alpha=0.1, label='presymptomatic')
        plt.axvspan(5.5, 14.8, color='blue', alpha=0.05, label='symptomatic')

        plt.title('Infectiousness distributions')
        plt.xlabel('Time since infection')
        plt.ylabel('Density')
        plt.legend()
        plt.xlim(0, 30)
        plt.ylim(bottom=0)
        plt.savefig(output.plot, dpi=image_resolution, bbox_inches="tight")
        plt.close()



###############################################
# ALL
###############################################
rule all:
    input:
        expand(rules.plot_infectiousness_distributions.output, outdir=outdir),
        expand(rules.plot_oscillations.output, outdir=outdir),
        expand(rules.alternative_warning_strategies_table.output, outdir=outdir, pathogen=pathogens),
        expand(rules.transmission_decomposition.output, outdir=outdir),
        expand(rules.baseline_intervention_table.output, outdir=outdir),
        expand(rules.plot_Rt_divergence_heatmap.output, outdir=outdir, pathogen=pathogens),
        expand(rules.plot_Rt_spatial.output, outdir=outdir),
        expand(rules.plot_spatial_heatmaps.output, outdir=outdir),
        expand(rules.plot_spatial_trajectories.output, outdir=outdir),
        expand(rules.plot_alternative_warning_strategies_eps_w.output, outdir=outdir, pathogen=pathogens),
        expand(rules.plot_alternative_warning_strategies.output.plot, 
            outdir=outdir, pathogen=["SARS-CoV-2"], k=[10], eps_s=[0.5], eps_w=[0.0, 0.4, 0.8, 1.0]
        ),
        expand(
            rules.plot_stochastic_cumulative_extinction_probability.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], eps_s=[0.4], eps_w=[0.4, 0.8],
        ),
        expand(rules.plot_superspreading_baseline_trajectories.output, outdir=outdir, pathogen=["SARS-CoV-2"],
            N=[100, 50_000, 500_000],
        ),
        expand(
            rules.plot_stochastic_cumulative_extinction_probability_superspreading.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], eps_s=[0.4], eps_w=[0.4, 0.8], scenario=['establishment', 'all'],
        ),
        expand(
            rules.plot_stochastic_intervention_grid.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[10], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"],
        ),
        expand(
            rules.plot_superspreading_intervention_grid.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[10], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"],
        ),
        expand(
            rules.plot_superspreading_intervention_grid_all_superspreading.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[10], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"],
        ),
        expand(rules.plot_linearised_branching_process_extinction_probabilities.output, outdir=outdir, pathogen=["SARS-CoV-2"]),
        expand(rules.plot_asymptomatic_grid_Rt_final.output.plot, outdir=outdir, pathogen=asymptomatic_pathogens),
        expand(rules.plot_asymptomatic_grid_Itot_final.output.plot, outdir=outdir, pathogen=asymptomatic_pathogens),
        expand(rules.plot_prcc_monotonicity.output.plot, outdir=outdir, pathogen=pathogens, scenario=prcc_scenarios, outcome=prcc_outcomes, bounds=["empirical", "symmetric"]),
        expand(rules.plot_prcc_grid.output.plot, outdir=outdir, bounds=["empirical", "symmetric"]),
        expand(rules.plot_combined_sensitivity_grid.output.plot, outdir=outdir),
        expand(rules.plot_stochastic_baseline_trajectories.output, outdir=outdir, pathogen=["SARS-CoV-2"], N=[100, 50_000, 500_000]),
        expand(rules.plot_trajectory.output.plot, outdir=outdir, pathogen=pathogens, epsilon_s=[0.0, 0.4, 0.8], epsilon_w=[0.0, 0.4, 0.8]),
        expand(rules.delayed_ww_intervention.output.plot, outdir=outdir, pathogen=["SARS-CoV-2"]),
        expand(rules.baseline_trajectories.output.plot, pathogen=pathogens, outdir=outdir),
        expand(rules.plot_true_vs_reported_Rt_scenarios.output, pathogen=pathogens, outdir=outdir),
        expand(rules.plot_true_vs_reported_Rt_scenarios_piecewise.output, pathogen=["SARS-CoV-2"], outdir=outdir, k=[10], scenario=list(STRATEGIES)),
        expand(rules.plot_true_vs_reported_Rt_heatmaps.output, pathogen=["SARS-CoV-2"], outdir=outdir, k=[10]), #k=[1, 3, 10, 30],),
        expand(rules.plot_response_function.output.plot, outdir=outdir),
        expand(rules.plot_main_intervention_grid.output.plot, outdir=outdir),
        expand(rules.plot_R_1_contours.output.plot, outdir=outdir),
        expand(rules.plot_combined_contour_grid_R1_Itot.output.plot, outdir=outdir),
        expand(rules.export_param_bounds.output.tex, outdir=outdir, bounds=["empirical", "symmetric"]),
        expand(rules.plot_controllability_boundaries.output.plot, outdir=outdir),
        expand(rules.plot_asymptomatic_generation_time.output.plot, outdir=outdir),
        expand(rules.plot_nonlinear_response_analysis.output.plot, outdir=outdir),
        expand(rules.plot_asymptomatic_landscape.output.plot, outdir=outdir),
        expand(rules.calculate_generation_times.output.txt, outdir=outdir),
        expand(rules.calculate_growth_rates.output.txt, outdir=outdir),
        expand(rules.plot_gain_margins.output.plot, outdir=outdir),
        expand(rules.plot_delay_margins.output.plot, outdir=outdir),
        expand(rules.plot_period_and_damping_scatter.output, outdir=outdir, pathogen=pathogens, k=[10, 30]),
