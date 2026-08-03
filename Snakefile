import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc, gamma, erlang, norm, truncnorm, triang, lognorm
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from functools import partial
import itertools
import csv as _csv

import pandas as pd
import seaborn as sns

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from models.compartmental import linear_chain, simulate_SEIPAR_W, simulate_SEIR_W
from models.compartmental_erlang import simulate_SEIPAR_W_Erlang, simulate_SEIR_W_Erlang
from models.compartmental_piecewise import simulate_SEIPAR_W_piecewise, simulate_SEIR_W_piecewise
from models.compartmental_piecewise_erlang import simulate_SEIPAR_W_piecewise_Erlang, simulate_SEIR_W_piecewise_Erlang
from models.gillespie import gillespie_SEIPAR_W, to_uniform_grid
from models.metrics import (
    R_boundary, calculate_mt_branching_q, calculate_mt_branching_q_with_superspreading,
    compute_asymptomatic_grid_Rt, compute_delay_metrics_grid, compute_I_tot_grid_delayed_ww,
    compute_metrics, compute_R_grid, eps_s_boundary, establishment_threshold, growth_rate, 
    growth_rate_erlang, mean_warning_multiplier, outcome_metrics, strategy_grid, strategy_metrics,
)
from models.parameters import Params, calculate_r, logistic_response_function
from models.parameters_erlang import ParamsErlang, compute_weights
from models.plotting import (
    f_days, f_pct, plot_asymptomatic_effect_for_range_of_intervention_efficacies,
    plot_extinction_probability_scenario, plot_final_R, plot_heatmap, plot_I_tot,
    plot_I_tot_delayed_ww, plot_nonlinear_response_analysis, plot_trajectory,
    table_row_metrics, table_scenario_label,
)
from models.sensitivity import (
    DUMMY, ELASTICITY_NAMES, SensitivityResults, Rt_elasticities, 
    elasticities, elasticity_symbol, export_sensitivity_bounds, 
    load_sensitivity_results, ordered_params, param_description, param_symbol, 
    partial_rank_residuals, run_sensitivity_analysis,
)
from models.spatial import (
    ALL_GE_PAIRS, ALL_ZH_PAIRS, SpatialParams, get_canton_pair_stats, load_and_preprocess_phone_data, 
    plot_flows, rescale_migration_rate, run_spatial, simulate_SEIPAR_W_spatial, 
    simulate_SEIR_W_spatial, unpack_spatial,
)
from models.stability import (
    arg_L, compute_rt_grid, critical_loop_gain, delay_margin, dominant_pole, eps_w_crit, 
    gain_margin, k_crit, loop_gain, period_and_damping, phase_margin,
)
from models.superspreading import gillespie_SEIPAR_W_superspreading, simulate_superspreading_outcomes
from models.uncertainty import (
    Marginal, Priors, analytic_boundary, as_uniform, cached_sample_derived, epi_quantities,
    get_epi_characteristics_dict, get_model_prior_list, joint_ci, params_from_priors,
    params_from_sample, probability_uncontrollable, pushforward, sample_derived, simulated_boundary,
)

### PARAMETERS ###
PATHOGENS = ["SARS-CoV-2", "H1N1", "Ebola"]
ASYMPTOMATIC_PATHOGENS = ["SARS-CoV-2", "H1N1"]
PATHOGENS_FULL_LANDSCAPE = ["SARS-CoV-2", "H1N1", "Ebola", "Omicron", "Measles", "Dengue", "Rhino"]
_zero = Marginal(0.0, 0.0, "uniform", mean=0.0)
PRIORS = {
    "SARS-CoV-2": Priors(marginals={
        "R_0": Marginal(2.40, 2.98, "lognormal", mean=2.69),
        "gamma_inv": Marginal(1.5, 4.5, "lognormal", mean=3.0),
        "sigma_inv": Marginal(1.0, 4.0, "lognormal", mean=2.5),
        "mu_s_inv": Marginal(7.8, 10.0, "lognormal", mean=9.3),
        "p": Marginal(0.23, 0.399, "beta", mean=0.351),
        "RR_p": Marginal(0.37, 2.71, "lognormal", mean=1.0),
        "RR_a": Marginal(0.16, 0.64, "lognormal", mean=0.32),
    }, presymptomatic=True, asymptomatic=True),
    "H1N1": Priors(marginals={
        "R_0": Marginal(1.30, 1.70, "lognormal", mean=1.46, quant_lo=0.25, quant_hi=0.75), 
        "gamma_inv": Marginal(1.41, 1.89, "lognormal", mean=1.65),
        "sigma_inv": Marginal(1.0, 4.0, "lognormal", mean=2.0),
        "mu_s_inv": Marginal(2.06, 4.69, "lognormal", mean=3.38),
        "p": Marginal(0.32, 0.4, "beta", mean=0.36),
        "RR_p": Marginal(0.04, 0.13, "lognormal", mean=0.08),
        "RR_a": Marginal(0.11, 1.54, "lognormal", mean=0.57),
    }, presymptomatic=True, asymptomatic=True),
    "Ebola": Priors(marginals={
        "R_0": Marginal(1.74, 2.15, "lognormal", mean=1.95),
        "gamma_inv": Marginal(7.7, 9.2, "lognormal", mean=8.5),
        "mu_s_inv": Marginal(3.7, 6.3, "lognormal", mean=5.0),
        "sigma_inv": _zero, "p": _zero, "RR_p": _zero, "RR_a": _zero,
    }, presymptomatic=False, asymptomatic=False),
    "Omicron": Priors(marginals={
        "R_0": Marginal(3.5, 11.4, "lognormal", mean=7.38), 
        "gamma_inv": Marginal(2.51, 4.6, "lognormal", mean=3.57),
        "sigma_inv": Marginal(0.02, 1.27, "lognormal", mean=0.69), 
        "mu_s_inv": Marginal(3.06, 6.18, "lognormal", mean=4.42),
        "p": Marginal(0.23, 0.399, "beta", mean=0.351),
        "RR_p": Marginal(0.37, 2.71, "lognormal", mean=1.0),
        "RR_a": Marginal(0.16, 0.64, "lognormal", mean=0.32),
    }, presymptomatic=True, asymptomatic=True),
    "Measles": Priors(marginals={
        "R_0": Marginal(12.0, 18.0, "lognormal", mean=15.0), 
        "gamma_inv": Marginal(7.0, 10.0, "lognormal", mean=8.5),
        "sigma_inv": Marginal(2.0, 4.0, "lognormal", mean=3.0), 
        "mu_s_inv": Marginal(4.0, 4.0, "uniform", mean=4.0),
        "RR_p": Marginal(0.5, 2.0, "lognormal", mean=1.0), 
        "RR_a": _zero, "p": _zero,
    }, presymptomatic=True, asymptomatic=False),
    "Dengue": Priors(marginals={
        "R_0": Marginal(2.0, 10.0, "lognormal", mean=6.0), 
        "gamma_inv": Marginal(3.0, 8.0, "uniform", mean=5.0),
        "sigma_inv": Marginal(1.0, 2.0, "lognormal", mean=1.5), 
        "mu_s_inv": Marginal(4.0, 5.0, "lognormal", mean=4.5),
        "p": Marginal(0.4, 0.8, "lognormal", mean=0.6), 
        "RR_a": Marginal(0.1, 1.0, "lognormal", mean=0.5),
        "RR_p": _zero, 
    }, presymptomatic=False, asymptomatic=True),
    "Rhino": Priors(marginals={
        "R_0": Marginal(2.3, 3.0, "lognormal", mean=2.8, quant_lo=0.25, quant_hi=0.75), 
        "gamma_inv": Marginal(0.5, 1.0, "lognormal", mean=0.75),
        "sigma_inv": Marginal(0.5, 1.0, "lognormal", mean=0.75), 
        "mu_s_inv": Marginal(8.0, 14.0, "lognormal", mean=11.0),
        "p": Marginal(0.1, 0.7, "beta", mean=0.58), 
        "RR_p": Marginal(0.1, 0.5, "lognormal", mean=0.2),
        "RR_a": Marginal(0.05, 0.2, "lognormal", mean=0.1),
    }, presymptomatic=True, asymptomatic=True),
}
_joint = {p: joint_ci(PRIORS[p], n=20000, seed=0)[0] for p in ASYMPTOMATIC_PATHOGENS}
P_CI = {p: _joint[p]["p"][1:] for p in ASYMPTOMATIC_PATHOGENS}
PHI_A_CI = {p: _joint[p]["phi_a"][1:] for p in ASYMPTOMATIC_PATHOGENS}
DISPERSION_SC2 = 0.4
ALPHA = 0.01
EPSILON_S = 0.8
EPSILON_W = 0.8
E0 = 1e-6
ICRIT = 1e-6
T1 = 10_000.0
PARAMETERS = {p: params_from_priors(PRIORS[p]) for p in PATHOGENS}
TRAJECTORY_END_TIMES = {"SARS-CoV-2": 530, "H1N1": 874, "Ebola": 1820} # 5x total wave time, rounded to nearest 10

# Models
MODELS = {"SARS-CoV-2": simulate_SEIPAR_W, "H1N1": simulate_SEIPAR_W, "Ebola": simulate_SEIR_W, "Omicron": simulate_SEIPAR_W, "Measles": simulate_SEIPAR_W, "Dengue": simulate_SEIPAR_W, "Rhino": simulate_SEIPAR_W}
MODELS_PIECEWISE = {"SARS-CoV-2": simulate_SEIPAR_W_piecewise, "H1N1": simulate_SEIPAR_W_piecewise, "Ebola": simulate_SEIR_W_piecewise}
SPATIAL_MODELS = {"SARS-CoV-2": simulate_SEIPAR_W_spatial, "H1N1": simulate_SEIPAR_W_spatial, "Ebola": simulate_SEIR_W_spatial}
MODEL_NAMES = {"SARS-CoV-2": "SEIPAR", "H1N1": "SEIPAR", "Ebola": "SEIR"}

# Sensitivity analysis
SENSITIVITY_RANGES = {p: as_uniform(PRIORS[p]) for p in PATHOGENS}
PRCC_SCENARIOS = ['start', 'threshold']
PRCC_OUTCOMES = ['Rt', 'Itot']
PRCC_SCENARIO_TITLES = {'start': r'$I_{\text{crit}}=0$', 'threshold': r'$I_{\text{crit}}=10^{-4}$'}
PRCC_OUTCOME_TITLES = {'Rt': r'$\mathcal{R}_t$', 'Itot': r'$I_\text{tot}$'}
SENSITIVITY_BLOCKS = ["Pathogen", "Model"]

# Alternative warning systems
R_OFF = 0.8
EVAL_INTERVAL = 14.0
T_LEAD = 7.0
CHECK_INTERVAL = 0.1
STRATEGIES = {"baseline": (False, False, 0.0, CHECK_INTERVAL), "lead": (False, False, T_LEAD, CHECK_INTERVAL), "interval": (False, True,  0.0, CHECK_INTERVAL), "asymmetric": (True,  False, 0.0, CHECK_INTERVAL), "lockdown": (True, True, 0.0, CHECK_INTERVAL)}
METRIC_NAMES = ["$\\mathcal{R}_t$", "total number of infections", "symptomatic peak", "steady-state $\\mathcal{R}_t$ amplitude", "time above $\\mathcal{R}_{crit}$", "total cost"]
METRIC_BOUNDS = [(0.0, 3.0), (0.0, 1.25), (0.0, 1.25), (0.0, 300.0), (0.0, 175.0), (0.0, 175.0)]
INTERVENTION_SCENARIOS = {pathogen: {"baseline": (0.0, 0.0), "isolation": (0.5, 0.0), "warning": (0.0, 0.5), "weak": (0.25, 0.25), "combined": (0.5, 0.5)} for pathogen in PATHOGENS}
def get_ew(Rt, ps, eps_s):
    ps = ps.update(epsilon_s=eps_s)
    R = calculate_r(ps.p, ps.phi_a, ps.phi_p, ps.mu_a_inv, ps.sigma_inv, ps.mu_s_inv, eps_s) * ps.beta
    return brentq(lambda ew: R * logistic_response_function(Rt, ps.update(epsilon_w=ew)) - Rt, 0, 1)
WARNING_SCENARIOS = {
    "SARS-CoV-2": {"baseline": (0.0, 0.0), "uncontrolled": (0.8, get_ew(1.2, PARAMETERS["SARS-CoV-2"], 0.8)), "barely uncontrolled": (0.8, get_ew(1.05, PARAMETERS["SARS-CoV-2"], 0.8)), "critical": (0.8, get_ew(1.0, PARAMETERS["SARS-CoV-2"], 0.8)), "controlled": (1.0, get_ew(0.95, PARAMETERS["SARS-CoV-2"], 1.0))},
    "H1N1": {"baseline": (0.0, 0.0), "uncontrolled": (0.0, get_ew(1.2, PARAMETERS["H1N1"], 0.0)), "barely uncontrolled": (0.0, get_ew(1.05, PARAMETERS["H1N1"], 0.0)), "critical": (0.0, get_ew(1.0, PARAMETERS["H1N1"], 0.0)), "controlled": (0.6, get_ew(0.8, PARAMETERS["H1N1"], 0.6))},
    "Ebola": {"baseline": (0.0, 0.0), "uncontrolled": (0.2, get_ew(1.2, PARAMETERS["Ebola"], 0.2)), "barely uncontrolled": (0.2, get_ew(1.05, PARAMETERS["Ebola"], 0.2)), "critical": (0.2, get_ew(1.0, PARAMETERS["Ebola"], 0.2)), "controlled": (0.55, get_ew(0.8, PARAMETERS["Ebola"], 0.55))},
}
K_RANGE = (3.2, 16.0)
R_CRITS = [0.9, 1.0, 1.1]

# LCT / Erlang model
PARAMETERS_LCT = {p: params_from_priors(PRIORS[p], model="Erlang") for p in PATHOGENS}
MODELS_LCT = {"SARS-CoV-2": simulate_SEIPAR_W_Erlang, "H1N1": simulate_SEIPAR_W_Erlang, "Ebola": simulate_SEIR_W_Erlang, "Omicron": simulate_SEIPAR_W_Erlang, "Measles": simulate_SEIPAR_W_Erlang, "Dengue": simulate_SEIPAR_W_Erlang, "Rhino": simulate_SEIPAR_W_Erlang}
MODELS_PIECEWISE_LCT = {"SARS-CoV-2": simulate_SEIPAR_W_piecewise_Erlang, "H1N1": simulate_SEIPAR_W_piecewise_Erlang, "Ebola": simulate_SEIR_W_piecewise_Erlang}

# Spatial
CANTON_PAIRS = "ZHAG_ZHBS_ZHGR"
HOLIDAY_TYPES = ['workday', 'summer', 'christmas']
MEAN_TRIP_DURATION = 8.0 # hours

# Styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = True
LINESTYLES = ['-','--', '-.', ':', (0,(1,2))]
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "P"]
PALETTE = sns.color_palette("colorblind", 1024)
COLORS = {pathogen: PALETTE[i] for i, pathogen in enumerate(PATHOGENS_FULL_LANDSCAPE)}
FIGSIZE = (6, 6)

# Files
OUTDIR = "results"
DATADIR = "data"


###############################################
# BASELINE
###############################################

rule plot_baseline_trajectories:
    output:
        plot="{outdir}/compartmental/baseline_trajectories_{pathogen}.png"
    run:
        peak_Is, time_to_peak, total_time = plot_trajectory(model=MODELS_LCT[wildcards.pathogen], params=PARAMETERS_LCT[wildcards.pathogen], path=output.plot, title=f"{wildcards.pathogen}", plot_total_I=True, t1=T1, icrit=ICRIT)

rule plot_nonlinear_response_analysis:
    output:
        plot="{outdir}/compartmental/nonlinear_response_analysis.png"
    run:
        path = output.plot
        plot_nonlinear_response_analysis(
            n_W = 3.0, tau_W = 14.0, n_B = 1.0, tau_B = 7.0, dt = 0.1, eps_w = 1.0, k = 10.0, threshold = 1.0,
            path=path, pathogens=PATHOGENS, colors=COLORS, parameters=PARAMETERS, 
            R0_lo={pathogen: PRIORS[pathogen].marginals["R_0"].lo for pathogen in PATHOGENS}, 
            R0_hi={pathogen: PRIORS[pathogen].marginals["R_0"].hi for pathogen in PATHOGENS},
        )


###############################################
# INTERVENTIONS
###############################################
rule plot_trajectory:
    output:
        plot="{outdir}/compartmental/no_decomp_trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}.png"
    run:
        plot_trajectory(
            no_decomp=True, t1=TRAJECTORY_END_TIMES[wildcards.pathogen], model=MODELS[wildcards.pathogen], plot_total_I=True,
            params=PARAMETERS[wildcards.pathogen].update(epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w)), 
            path=output.plot, title=f"{wildcards.pathogen} ($\\varepsilon_s={wildcards.epsilon_s}, \\varepsilon_w={wildcards.epsilon_w}$)"
        )

rule delayed_ww_intervention:
    output:
        plot="{outdir}/compartmental/delay_grid_ww_intervention_{pathogen}.png"
    run:
        fig = plot_I_tot_delayed_ww(model=simulate_SEIPAR_W_Erlang, parameters=PARAMETERS_LCT[wildcards.pathogen].update(epsilon_s=0.0, epsilon_w=EPSILON_W), title='Symptomatic prevalence threshold')
        fig.savefig(output.plot); plt.close(fig)

rule plot_main_intervention_grid:
    output:
        plot="{outdir}/compartmental/main_intervention_grid.png"
    run:
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        t1 = 30000.0

        # compute per pathogen
        Rt_g, tRt_g, Itot_g, peakIs_g, Rt_k1 = {}, {}, {}, {}, {}
        for pathogen in PATHOGENS:
            ps = PARAMETERS[pathogen]
            Rt, tRt, It, pk = compute_metrics(model=MODELS[pathogen], base_params=ps, eps_ww=eps_ww, eps_ss=eps_ss, t1=t1, E0=E0)
            Rt_g[pathogen] = np.array(Rt)
            tRt_g[pathogen] = np.array(tRt)
            _, yy0 = MODELS[pathogen](params=ps.update(epsilon_s=0.0, epsilon_w=0.0), t1=t1, E0=E0)
            Itot_g[pathogen] = np.array(It) / float(yy0[0,0] - yy0[-1,0])
            Is0 = yy0[:, -(ps.n_W + ps.n_B + 2)]
            peakIs_g[pathogen] = np.array(pk) / float(np.max(Is0))
            Rt_k1[pathogen], _, _, _ = compute_metrics(model=MODELS[pathogen], base_params=ps.update(k=1), eps_ww=eps_ww, eps_ss=eps_ss, t1=500.0, E0=E0)

        # plot
        rows = [ # label, data, cmap, center_at_one, log
            ('$\\mathcal{R}_t$', Rt_g, 'RdBu_r', True, False),
            ('Time to $\\mathcal{R}_t<1$', tRt_g, 'magma', False, True),
            ('$I_\\text{tot}$ (relative to baseline)', Itot_g, 'viridis', False, False),
            ('Peak $I_s$ (relative to baseline)', peakIs_g, 'viridis', False, False),
        ]
        fig, axs = plt.subplots(nrows=len(rows), ncols=len(PATHOGENS), figsize=(13, 16), sharex=True, sharey=True)
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
            for col_idx, pathogen in enumerate(PATHOGENS):
                ax = axs[row_idx, col_idx]
                mesh = ax.pcolormesh(np.array(eps_ww), np.array(eps_ss), data[pathogen], cmap=cmap, norm=norm, shading='auto', rasterized=True)
                meshes.append(mesh)
                ax.set_aspect('equal')
                # contours, titles, labels
                if center_at_one: ax.contour(np.array(eps_ww), np.array(eps_ss), Rt_k1[pathogen], levels=[1.0], colors='black', linewidths=1.0, linestyles='--')
                if row_idx == 0: ax.set_title(pathogen, fontsize=14)
                if row_idx == len(rows) - 1: ax.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=11)
                if col_idx == 0: ax.set_ylabel('Isolation efficacy $\\varepsilon_s$', fontsize=11)
            # colorbar
            cbar = fig.colorbar(meshes[-1], ax=axs[row_idx, :].tolist(), shrink=0.85, aspect=25, pad=0.02)
            cbar.set_label(label, fontsize=12, labelpad=8)
            if center_at_one: cbar.ax.axhline(1.0, color='black', linewidth=1.0)
        fig.savefig(output.plot); plt.close(fig)

rule plot_R_1_contours:
    output:
        plot="{outdir}/compartmental/R_1_contours.png",
    run:
        eps_ww = np.linspace(0.0, 1.0, 100)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for pathogen in PATHOGENS:
            lo, med, hi = np.clip(analytic_boundary(PRIORS[pathogen], eps_ww=eps_ww, n=200_000, seed=0), 0.0, 1.0)
            ax.fill_between(eps_ww, lo, hi, color=COLORS[pathogen], alpha=0.2, lw=0)
            ax.plot(eps_ww, med, color=COLORS[pathogen], label=pathogen)
        ax.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=12)
        ax.set_ylabel('Isolation efficacy $\\varepsilon_s$', fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_title(r"$\mathcal{R}_t=1$ boundary")
        ax.legend(handles=[Patch(facecolor=COLORS[p], alpha=0.5, label=p) for p in PATHOGENS], loc='upper right')
        fig.savefig(output.plot); plt.close(fig)

rule plot_combined_contour_grid_R1_Itot:
    output:
        plot="{outdir}/compartmental/combined_R1_and_Itot_reduction_contours.png"
    run:
        eps_ww = jnp.linspace(0.0, 0.999, 50)
        eps_ss = jnp.linspace(0.0, 0.999, 50)
        t1 = 10000.0
        R_crits = [0.9, 1.0, 1.1]
        linestyles = ['--', '-', ':']
        fig, (ax_R, ax_I) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        for pathogen in PATHOGENS:
            model = MODELS[pathogen]
            base_params = PARAMETERS[pathogen]
            theta = get_epi_characteristics_dict(base_params)["theta"]
            _, yy0 = model(params=base_params, t1=t1, E0=E0)
            baseline_Itot = yy0[0,0] - yy0[-1,0]
            for i, r_crit in enumerate(R_crits):
                eps_s_crit = eps_s_boundary(R_0=float(base_params.R_0), theta=theta, eps_w=eps_ww, k=float(base_params.k), R_crit=r_crit)
                ax_R.plot(eps_ww, eps_s_crit, color=COLORS[pathogen], linestyle=linestyles[i], linewidth=2, alpha=0.8)
                _, _, Itot_grid, _ = compute_metrics(model=model, base_params=base_params.update(R_crit=r_crit), eps_ww=eps_ww, eps_ss=eps_ss, t1=t1, E0=E0)
                ax_I.contour(eps_ww, eps_ss, np.array(Itot_grid) / float(baseline_Itot), levels=[0.2], colors=[COLORS[pathogen]], linestyles=[linestyles[i]], linewidths=2, alpha=0.8)
        ax_R.set_title('Controllability boundaries ($\\mathcal{R}_t = 1$)', fontsize=14, pad=10)
        ax_I.set_title('80% reduction in total infections', fontsize=14, pad=10)
        ax_R.set_ylabel('Isolation efficacy $\\varepsilon_s$', fontsize=12)
        ax_R.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=12)
        ax_I.set_xlabel('Warning response efficacy $\\varepsilon_w$', fontsize=12)
        ax_R.set_xlim(0, 1); ax_R.set_ylim(0, 1); ax_I.set_xlim(0, 1)
        ax_R.grid(True, alpha=0.3); ax_I.grid(True, alpha=0.3)
        ax_R.set_aspect('equal'); ax_I.set_aspect('equal')
        ax_R.legend(handles=[Line2D([0],[0],color=COLORS[p],lw=3,label=p) for p in PATHOGENS] + [Line2D([0],[0],color='gray',lw=2,linestyle=linestyles[i],label=f'$R_{{crit}}={r}$') for i, r in enumerate(R_crits)], loc='upper left', fontsize=11)
        fig.savefig(output.plot); plt.close(fig)
          
rule plot_controllability_boundaries:
    output:
        plot="{outdir}/compartmental/controllability_boundaries.png"
    run:
        ps = jnp.linspace(0.0, 0.999, 100)
        phis = jnp.linspace(0.0, 0.999, 100)
        eps_s_levels = [0.25, 0.5, 0.75, 1.0]
        eps_w_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        colormap = dict(zip(eps_s_levels, ['orange', 'yellow', 'lime', 'green']))
        linestylemap = dict(zip(eps_w_levels, [(0,(1,2)), (0,(1,5)), (0,(4,2)), (0,(8,3)), '-']))

        fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
        for ax, pathogen in zip(axs, ASYMPTOMATIC_PATHOGENS):
            base = PARAMETERS[pathogen]
            # nonsymptomatic fraction heatmap
            P, PHI = np.meshgrid(np.array(ps), np.array(phis), indexing='xy')
            Ra = P * PHI * base.mu_a_inv
            Rp = (1.0 - P) * base.phi_p * base.sigma_inv
            Rs = (1.0 - P) * base.mu_s_inv
            THETA = (Ra + Rp) / (Ra + Rp + Rs)
            mesh = ax.pcolormesh(np.array(ps), np.array(phis), THETA, cmap='Greys', vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
            if pathogen == "H1N1": fig.colorbar(mesh, ax=ax, label=r'proportion nonsymptomatic $\theta$')
            # Rt = 1 contours
            R0 = float(base.R_0)
            for eps_s in eps_s_levels:
                for eps_w in eps_w_levels:
                    theta_crit = 1.0 - (1.0 - 1.0 / (R0 * mean_warning_multiplier(eps_w))) / eps_s
                    if not (THETA.min() < theta_crit < min(THETA.max(), 1.0)):
                        continue
                    ax.contour(np.array(ps), np.array(phis), THETA, levels=[theta_crit], colors=[colormap[eps_s]], linestyles=[linestylemap[eps_w]], linewidths=2.0)
            # literature estimates
            p_lower, p_upper = P_CI[pathogen]
            phi_lower, phi_upper = PHI_A_CI[pathogen]
            xerr = np.array([[base.p - p_lower], [p_upper - base.p]]) if p_lower is not None else None
            yerr = np.array([[base.phi_a - phi_lower], [phi_upper - base.phi_a]]) if phi_lower is not None else None
            ax.errorbar(base.p, base.phi_a, xerr=xerr, yerr=yerr, fmt='o', color='white', markeredgecolor='black', ecolor='black', elinewidth=1.2, capsize=3, markersize=6, zorder=5)
            # axes
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xlabel(r'Proportion asymptomatic $p$', fontsize=12)
            ax.set_title(pathogen, fontsize=14, pad=8)
        axs[0].set_ylabel(r'Relative infectiousness $\varphi_a$', fontsize=12)
        # legend
        legend_handles = ([Patch(facecolor=c, label=rf'$\varepsilon_s = {e}$') for e, c in colormap.items()]
            + [Line2D([0], [0], color='gray', lw=2, ls=s, label=rf'$\varepsilon_w = {e}$') for e, s in linestylemap.items()]
            + [Line2D([0], [0], marker='o', color='white', markeredgecolor='black', linestyle='None', markersize=6, label=r'literature estimates')])
        fig.legend(handles=legend_handles, loc='outside lower center', ncol=5, frameon=False, fontsize=10)
        fig.savefig(output.plot); plt.close(fig)

rule derived_epi_characteristics:
    output:
        csv="{outdir}/compartmental/derived_epi_characteristics.csv",
    run:
        n_samples = 5000
        ci = (2.5, 97.5)
        seed = 0
        rows = []
        for pathogen in PATHOGENS:
            base = PARAMETERS[pathogen]
            fields = get_model_prior_list(PRIORS[pathogen])
            s = sample_derived(PRIORS[pathogen], n=n_samples, seed=seed)
            point = get_epi_characteristics_dict(base)
            draws = [get_epi_characteristics_dict(base.update(**{field: float(s[field][i]) for field in fields})) for i in range(n_samples)]
            for p in point:
                arr = np.array([d[p] for d in draws])
                lo_ci, hi_ci = np.percentile(arr, ci)
                rows.append((pathogen, p, point[p], float(np.median(arr)), float(lo_ci), float(hi_ci)))
            pu = probability_uncontrollable(PRIORS[pathogen], n=200_000, seed=0)
            rows.append((pathogen, "probability uncontrollable", "%.4f"%pu["P"], "%.3f"%pu["median"], "%.3f"%pu["lo"], "%.3f"%pu["hi"]))
        with open(output.csv, "w") as f:
            f.write("pathogen,quantity,point,median,ci_lo,ci_hi\n")
            for pathogen, quantity, pt, med, lo_ci, hi_ci in rows:
                f.write(f"{pathogen},{quantity},{pt},{med},{lo_ci},{hi_ci}\n")

rule baseline_intervention_table:
    output:
        tex="{outdir}/compartmental/baseline_intervention_table.tex",
    run:
        rows = {p: [] for p in PATHOGENS}
        for pathogen in PATHOGENS:
            ps = PARAMETERS_LCT[pathogen].update(R_crit=0.9)
            base = table_row_metrics(ps, MODELS_LCT[pathogen], 0.0, 0.0, T1, icrit=ICRIT)
            base["prevented"] = 0.0
            for name, (eps_s, eps_w) in INTERVENTION_SCENARIOS[pathogen].items():
                rows[pathogen].append((name, eps_s, eps_w, (base if (eps_s==0.0 and eps_w==0.0) else table_row_metrics(ps, MODELS_LCT[pathogen], eps_s, eps_w, T1, itot_baseline=base["itot"], icrit=ICRIT))))
        with open(output.tex, "w") as f:
            f.write("\\begin{table}[H]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{scenario} & $\\mathcal{R}_t$ & \\textbf{peak sympt.} & \\textbf{time to peak} & \\textbf{wave time} & \\textbf{attack rate} & \\textbf{inf. prevented} & \\textbf{isol. cost} & \\textbf{warn cost}\\\\\n\\midrule\n")
            for i, pathogen in enumerate(PATHOGENS):
                f.write(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{pathogen}}}}}\\\\\n")
                for name, eps_s, eps_w, m in rows[pathogen]:
                    tp, wt = f_days(m)
                    f.write(" & ".join([f"\\quad {table_scenario_label(name, eps_s, eps_w)}", f"{m['Rt']:.2f}", f_pct(m["peak_Is"], 1), tp, wt, f_pct(m["itot"], 0), f_pct(m["prevented"], 0), f"{m['isolation_cost']:.1f}", f"{m['warning_cost']:.1f}",]) + " \\\\\n")
                if i < len(PATHOGENS)-1: f.write("\\midrule\n")
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption[Baseline and intervention characteristics of epidemic scenarios]{Baseline and intervention characteristics of epidemic scenarios.}\\label{tab:baseline_intervention}\n\\end{table}\n")

rule plot_crossings:
    output:
        plot="{outdir}/compartmental/crossings.png"
    run:
        pathogen = "SARS-CoV-2"
        model = MODELS[pathogen]
        ps = PARAMETERS[pathogen]
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        
        t1 = T1/10
        R_crits = [0.9, 1.0, 1.1]
        linestyles = ['--', '-', ':']

        @partial(jax.jit, static_argnames=['model', 'n_ts'])
        def _run_grid(model, base_params, eps_ww, eps_ss, t1, E0, delta_dep=0.05, n_ts=5000):
            def wrap_metrics(w, s):
                params = base_params.update(epsilon_w=w, epsilon_s=s)
                tt, yy, *_ = model(params=params, t1=t1, E0=E0, n_ts=n_ts)
                crossings = outcome_metrics(tt, yy, params, t1, delta_dep)[-1]
                return crossings
            return jax.vmap(jax.vmap(wrap_metrics, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)

        crossings = _run_grid(model=model, base_params=ps.update(R_crit=1.0), eps_ww=eps_ww, eps_ss=eps_ss, t1=t1, E0=E0)
        fig, ax = plot_heatmap(
            X=eps_ww, Y=eps_ss, Z=crossings, cmap='magma', cbar_label='crossings',
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title='Number of warning threshold crossings'
        )
        fig.savefig(output.plot); plt.close(fig)


###############################################
# ASYMPTOMATIC
###############################################
rule plot_asymptomatic_grid_Rt_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Rt_final_{pathogen}.png"
    run:
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(model=MODELS[wildcards.pathogen], params=PARAMETERS[wildcards.pathogen], p_CI=P_CI[wildcards.pathogen], phi_a_CI=PHI_A_CI[wildcards.pathogen], total_infected=False, path=output.plot, t1=T1)

rule plot_asymptomatic_grid_Itot_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Itot_final_{pathogen}.png"
    run:
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(t1=600.0, model=MODELS[wildcards.pathogen], params=PARAMETERS[wildcards.pathogen], p_CI=P_CI[wildcards.pathogen], phi_a_CI=PHI_A_CI[wildcards.pathogen], total_infected=True, path=output.plot)

rule plot_asymptomatic_generation_time:
    output:
        plot="{outdir}/compartmental/asymptomatic_generation_time_{pathogen}.png"
    run:
        pathogen = wildcards.pathogen
        ps = PARAMETERS[pathogen]
        P = jnp.linspace(0.0, 0.999, 100)
        PHI_A = jnp.linspace(0.0, 0.999, 100)

        def get_generation_time(p, phi_a):
            nom = p * phi_a * ps.mu_a_inv**2 +(1-p)*(ps.phi_p * ps.sigma_inv**2 + ps.mu_s_inv**2 + ps.sigma_inv*ps.mu_s_inv)
            denom = p * phi_a * ps.mu_a_inv + (1-p)*(ps.phi_p * ps.sigma_inv + ps.mu_s_inv)
            return ps.gamma_inv + nom / denom
        generation_times = jax.vmap(jax.vmap(get_generation_time, in_axes=(None, 0)), in_axes=(0, None))(P, PHI_A)

        fig, ax = plot_heatmap(
            X=P, Y=PHI_A, Z=generation_times, cmap='magma', cbar_label='generation time', contour_levels=[11,12,13,14,15],
            title='Asymptomatic transmission and generation time',
            xlabel=r'Proportion asymptomatic, $p$', ylabel=r'Relative infectiousness, $\varphi_a$',
        )
        xerr = np.array([[ps.p - P_CI[pathogen][0]], [P_CI[pathogen][1] - ps.p]])
        yerr = np.array([[ps.phi_a - PHI_A_CI[pathogen][0]], [PHI_A_CI[pathogen][1] - ps.phi_a]])
        ax.errorbar(ps.p, ps.phi_a, xerr=xerr, yerr=yerr, fmt='o', color='white', markeredgecolor='black', ecolor='white', elinewidth=1.5, capsize=3, markersize=5)
        ax.set_ylim([0.0,1.0])
        fig.savefig(output.plot); plt.close(fig)

rule plot_asymptomatic_landscape:
    output:
        plot="{outdir}/compartmental/asymptomatic_landscape.png"
    run:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # asymptomatic landscape
        for pathogen in PATHOGENS_FULL_LANDSCAPE:
            s = sample_derived(PRIORS[pathogen], n=10000, seed=0)
            R0_s, theta_s = s["R_0"], epi_quantities(s)["theta"]
            ax.scatter(theta_s, R0_s, s=4, alpha=0.1, color=COLORS[pathogen], edgecolors='none')

        # controllability boundaries
        thetas = np.linspace(0, 1, 500)
        scenarios = [
            {"label": "Warning ($\epsilon_s=0, \epsilon_w=0.8$)", "eps_s": 0.0, "eps_w": 0.8, "color": "black"},
            {"label": "Isolation ($\epsilon_s=0.8, \epsilon_w=0$)", "eps_s": 0.8, "eps_w": 0.0, "color": "grey"},
            {"label": "Combined ($\epsilon_s=0.8, \epsilon_w=0.8$)", "eps_s": 0.8, "eps_w": 0.8, "color": "green"}]
        for s in scenarios:
            ax.plot(thetas, R_boundary(thetas, s["eps_s"], s["eps_w"]), color=s["color"], linewidth=2, linestyle='-', label=s["label"])

        # plotting
        ax.set_xlim([-0.01, 1])
        ax.set_ylim([0, 20])
        ax.set_xlabel('Proportion presymptomatic and asymptomatic $\\theta$', fontsize=12)
        ax.set_ylabel('Basic reproductive number $\\mathcal{R}_0$', fontsize=12)
        ax.add_artist(ax.legend(handles=[Patch(facecolor=COLORS[p], label=p) for p in PATHOGENS_FULL_LANDSCAPE], loc='upper left', title="Pathogens"))
        ax.legend(loc='upper right', title="Controllability ($\\mathcal{R}_t<1$)")
        ax.grid(True, alpha=0.3)
        plt.savefig(output.plot); plt.close()


###############################################
# SENSITIVITY ANALYSIS
###############################################
rule compute_prcc:
    output:
        npz="{outdir}/prcc/data_{pathogen}_{scenario}_{outcome}_{bounds}.npz"
    run:
        t1 = T1/10
        around_mean = wildcards.bounds=="symmetric"
        results = run_sensitivity_analysis(
            model=MODELS[wildcards.pathogen], scenario=wildcards.scenario, outcome=wildcards.outcome,
            base_params=PARAMETERS[wildcards.pathogen].update(epsilon_s=EPSILON_S, epsilon_w=EPSILON_W),
            priors=None if around_mean else SENSITIVITY_RANGES[wildcards.pathogen],
            t1=t1, E0=E0, n_lhs=5000, n_sobol_base=1024, around_mean=around_mean,
        )
        np.savez_compressed(
            output.npz, param_names=np.array(results.param_names), lower_bounds=np.array([results.bounds[k][0] for k in results.param_names]), 
            upper_bounds=np.array([results.bounds[k][1] for k in results.param_names]), samples=results.samples, outputs=results.outputs, 
            prcc_mean=results.prcc_mean, prcc_lower=results.prcc_lower, prcc_upper=results.prcc_upper, sobol_S1=results.sobol_S1, 
            sobol_S1_conf=results.sobol_S1_conf, sobol_ST=results.sobol_ST, sobol_ST_conf=results.sobol_ST_conf, sobol_S1_sum=results.sobol_S1_sum,
        )

rule plot_prcc_monotonicity:
    input:
        npz=rules.compute_prcc.output.npz
    output:
        plot="{outdir}/prcc/monotonicity_{pathogen}_{scenario}_{outcome}_{bounds}.png"
    run:
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
            ax.scatter(ex, ey, s=4, alpha=0.25, edgecolors='none', color=COLORS[wildcards.pathogen])
            ax.set_title(f'{param_symbol(name)} ({results.prcc_mean[i]:+.2f})', fontsize=18)
            x_trend, y_trend = calculate_binned_means(ex, ey)
            ax.plot(x_trend, y_trend, color='black', lw=1.4, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color='gray', lw=0.4, alpha=0.5); ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
        for j in range(d, n_rows*n_cols): axs[j//n_cols, j%n_cols].axis('off') # remove unused subplots

        # labels and titles
        fig.supxlabel('partial rank of parameter', fontsize=24)
        fig.supylabel('partial rank of output', fontsize=24)
        fig.suptitle(f'{wildcards.pathogen}, {PRCC_SCENARIO_TITLES[wildcards.scenario]}, {PRCC_OUTCOME_TITLES[wildcards.outcome]}', fontsize=32)
        fig.savefig(output.plot); plt.close(fig)

rule plot_prcc_grid:
    input:
        npz=lambda wc: expand("{outdir}/prcc/data_{pathogen}_{scenario}_{outcome}_{bounds}.npz", outdir=wc.outdir, pathogen=PATHOGENS, scenario=['start', 'threshold'], outcome=PRCC_OUTCOMES, bounds=wc.bounds)
    output:
        plot="{outdir}/prcc/combined_prcc_grid_{bounds}.png"
    run:
        # all params for global x axis
        all_params = ordered_params([p for path in input.npz for p in load_sensitivity_results(path).param_names])
        all_labels = [param_symbol(n) for n in all_params]
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
        n_rows = len(PATHOGENS)
        n_cols = len(PRCC_OUTCOMES)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), sharey=True)
        w = 0.32
        for r, pathogen in enumerate(PATHOGENS):
            color = COLORS[pathogen]
            for c, outcome in enumerate(PRCC_OUTCOMES):
                ax = axs[r, c]
                results_start = load_sensitivity_results(f"{wildcards.outdir}/prcc/data_{pathogen}_start_{outcome}_{wildcards.bounds}.npz")
                results_threshold = load_sensitivity_results(f"{wildcards.outdir}/prcc/data_{pathogen}_threshold_{outcome}_{wildcards.bounds}.npz")
                params_threshold_mean = params_aligned(results_threshold, 'prcc_mean', is_abs=False)
                labels = [label if np.isfinite(params_threshold_mean[i]) else "" for i, label in enumerate(all_labels)]
                err_kw = dict(ecolor='k', linewidth=0.6, capsize=1.5)
                edge_kw = dict(edgecolor='k', linewidth=0.3)
                ax.bar(x - 0.5*w, params_aligned(results_start, 'prcc_mean', is_abs=False), yerr=params_aligned(results_start, 'prcc_err'), width=w, color=color, **edge_kw, error_kw=err_kw)
                ax.bar(x + 0.5*w, params_threshold_mean, yerr=params_aligned(results_threshold, 'prcc_err'), width=w, color=color, alpha=0.5, hatch='////', **edge_kw, error_kw=err_kw)                
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
                ax.set_ylim(-1.05, 1.05)
                ax.grid(axis='y', alpha=0.3)
                if c == 0: ax.set_ylabel(f'{pathogen}', fontsize=20)
                if r == 0: ax.set_title(PRCC_OUTCOME_TITLES.get(outcome, outcome), fontsize=20)
        fig.legend(
            handles=[Patch(facecolor='gray', label=r'PRCC ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', hatch='////', label=r'PRCC ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3, alpha=0.5)],
            loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=14, frameon=True)
        fig.savefig(output.plot); plt.close(fig)

rule plot_combined_sensitivity_grid:
    input:
        npz=lambda wc: expand("{outdir}/prcc/data_{pathogen}_{scenario}_{outcome}_{bounds}.npz", outdir=wc.outdir, pathogen=PATHOGENS, scenario=['start', 'threshold'], outcome=PRCC_OUTCOMES, bounds=["empirical"])
    output:
        plot="{outdir}/prcc/combined_sensitivity_grid.png"
    run:
        # all params for global x axis
        all_params = ordered_params([p for path in input.npz for p in load_sensitivity_results(path).param_names])
        labels = [param_symbol(n) for n in all_params]
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
        n_rows = len(PATHOGENS)
        n_cols = len(PRCC_OUTCOMES)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 5*n_rows), sharey=True)
        w = 0.14
        for r, pathogen in enumerate(PATHOGENS):
            color = COLORS[pathogen]
            for c, outcome in enumerate(PRCC_OUTCOMES):
                ax = axs[r, c]
                results_start = load_sensitivity_results(f"{wildcards.outdir}/prcc/data_{pathogen}_start_{outcome}_empirical.npz")
                results_threshold = load_sensitivity_results(f"{wildcards.outdir}/prcc/data_{pathogen}_threshold_{outcome}_empirical.npz")
                err_kw = dict(ecolor='k', linewidth=0.6, capsize=1.5)
                edge_kw = dict(edgecolor='k', linewidth=0.3)
                ax.bar(x - 2.5*w, params_aligned(results_start, 'prcc_mean', is_abs=True), yerr=params_aligned(results_start, 'prcc_err'), width=w, color=color, **edge_kw, error_kw=err_kw)
                ax.bar(x - 1.5*w, params_aligned(results_threshold, 'prcc_mean', is_abs=True), yerr=params_aligned(results_threshold, 'prcc_err'), width=w, color=color, hatch='////', **edge_kw, error_kw=err_kw)
                ax.bar(x - 0.5*w, params_aligned(results_start, 'sobol_S1'), yerr=params_aligned(results_start, 'sobol_S1_conf', is_err=True), width=w, color=color, alpha=0.7, **edge_kw, error_kw=err_kw)
                ax.bar(x + 0.5*w, params_aligned(results_threshold, 'sobol_S1'), yerr=params_aligned(results_threshold, 'sobol_S1_conf', is_err=True), width=w, color=color, alpha=0.7, hatch='////', **edge_kw, error_kw=err_kw)
                ax.bar(x + 1.5*w, params_aligned(results_start, 'sobol_ST'), yerr=params_aligned(results_start, 'sobol_ST_conf', is_err=True), width=w, color=color, alpha=0.35, **edge_kw, error_kw=err_kw)
                ax.bar(x + 2.5*w, params_aligned(results_threshold, 'sobol_ST'), yerr=params_aligned(results_threshold, 'sobol_ST_conf', is_err=True), width=w, color=color, alpha=0.35, hatch='////', **edge_kw, error_kw=err_kw)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                if c == 0: ax.set_ylabel(f'{pathogen}', fontsize=20)
                if r == 0: ax.set_title(PRCC_OUTCOME_TITLES.get(outcome, outcome), fontsize=20)
        fig.legend(
            handles=[
                Patch(facecolor='gray', label=r'|PRCC| ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', hatch='////', label=r'|PRCC| ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.7, label=r'$S_1$ ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.7, hatch='////', label=r'$S_1$ ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.35, label=r'$S_T$ ($I_\text{crit}=0$)', edgecolor='k', linewidth=0.3),
                Patch(facecolor='gray', alpha=0.35, hatch='////', label=r'$S_T$ ($I_\text{crit}=10^{-4}$)', edgecolor='k', linewidth=0.3)], 
            loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=16, frameon=True)
        fig.savefig(output.plot); plt.close(fig)

rule export_param_bounds:
    input:
        npzs=lambda wc: expand(rules.compute_prcc.output.npz, pathogen=PATHOGENS, scenario="threshold", outcome="Itot", outdir=OUTDIR, bounds=wc.bounds)
    output:
        tex="{outdir}/prcc/sensitivity_bounds_table_{bounds}.tex"
    run:
        export_sensitivity_bounds(combinations=list(itertools.product(PATHOGENS, ["threshold"])), path=output.tex, npzs=input.npzs)

rule export_elasticities:
    output:
        tex="{outdir}/prcc/elasticities.tex",
    run:
        points = {pathogen: [(scenario, WARNING_SCENARIOS[pathogen][scenario]) for scenario in ["uncontrolled", "critical"]] for pathogen in PATHOGENS}
        all_results = {}
        for pathogen in PATHOGENS:
            base = PARAMETERS[pathogen]
            RR_a = (float(base.phi_a) * float(base.mu_a_inv) / float(base.mu_s_inv) if float(base.mu_a_inv) > 0 else 0.0)
            RR_p = (float(base.phi_p) * float(base.sigma_inv) / float(base.mu_s_inv) if float(base.sigma_inv) > 0 else 0.0)
            all_results[pathogen] = {}
            for label, (eps_s, eps_w) in points[pathogen]:
                all_results[pathogen][label] = Rt_elasticities(base.R_0, base.p, RR_a, RR_p, eps_s, eps_w, base.k, base.R_crit)
        order = ["R_0", "p", "RR_a", "RR_p", "epsilon_s", "epsilon_w", "k", "R_crit", "gamma_inv", "sigma_inv", "mu_s_inv", "mu_a_inv", "tau_W", "tau_B"]
        rows = []
        for name in order:
            row_values = []
            for pathogen in PATHOGENS:
                for l, _ in points[pathogen]:
                    val = "%+.3f" % all_results[pathogen][l]["elasticities"][name]
                    row_values.append("--\\," + val[1:] if val.startswith("-") else val)
            rows.append("%s & %s & %s \\\\" % (param_description(name), param_symbol(name), " & ".join(row_values)))
        header = " & & " + " & ".join(["\\multicolumn{%d}{c}{%s}" % (len(points[p]), p) for p in PATHOGENS]) + " \\\\"
        subheader = "Parameter & Symbol & " + " & ".join(["$\\mathcal{R}_t=%.3f$" % all_results[p][l]["Rt"] for p in PATHOGENS for l, e in points[p]]) + " \\\\"
        with open(output.tex, "w") as f:
            f.write("\\begin{table}[H]\n\\centering\n\\small\n\\begin{tabular}{ll%s}\n\\toprule\n" % ("c" * len(PATHOGENS) * len(points["H1N1"])))
            f.write(header + "\n")
            f.write(subheader + "\n\\midrule\n")
            f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
            f.write("\\caption[Elasticities of the reproductive number]{Elasticities of the reproductive number across all pathogens}\n\\label{tab:elasticities}\n\\end{table}\n")


###############################################
# STABILITY ANALYSIS
###############################################
rule plot_true_vs_reported_Rt_scenarios:
    output:
        plot="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_scenarios.png",
    run:
        taus_W = [7.0, 14.0, 21.0]
        taus_B = [3.0, 7.0, 14.0]
        ks = [5.0, 10.0, 50.0]
        k = 10.0
        t1 = 300.0

        epsilon_s = EPSILON_S if wildcards.pathogen == "SARS-CoV-2" else 0.0
        base_params = PARAMETERS[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, k=k)
        model = MODELS[wildcards.pathogen]

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

        base_params = PARAMETERS[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, tau_W=14.0, tau_B=7.0)

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
        plt.savefig(output.plot); plt.close()

rule plot_true_vs_reported_Rt_scenarios_piecewise:
    output:
        plot="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_k{k}_scenarios_{scenario}.png",
    run:
        k = float(wildcards.k)
        t1 = 300.0
        tau_W = 14.0
        tau_B = 7.0
        eps_w_values = [0.0, 0.4, 0.8, 1.0]

        asymmetric = wildcards.scenario=="asymmetric"
        discrete_eval = wildcards.scenario=="interval"
        lead = wildcards.scenario=="lead"
        T_lead = 7.0 if lead else 0.0

        model = MODELS_PIECEWISE_LCT[wildcards.pathogen]

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=1, ncols=len(eps_w_values), figsize=(16,4), sharex=True, sharey=True)

        for j, eps_w in enumerate(eps_w_values):
            ps = PARAMETERS_LCT[wildcards.pathogen].update(epsilon_s=EPSILON_S, epsilon_w=eps_w, k=k, R_off=0.8, eval_interval=28.0, T_lead=T_lead)
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
        fig.suptitle(f'{wildcards.pathogen}: $k={k:g}$, {wildcards.scenario} ($\\tau_W={ps.tau_W:g}$, $\\tau_B={ps.tau_B:g}$, $\\varepsilon_s={EPSILON_S:g}$)', fontsize=15, y=1.03,)
        fig.legend(
            [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='red', lw=2)],
            ['True $R_t$', 'Reported $R_t$'],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12), fontsize=14,
        )
        plt.savefig(output.plot); plt.close()

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
        pathogen = wildcards.pathogen
        taus_W = jnp.linspace(1.0, 31.0, 100)
        taus_B = jnp.linspace(1.0, 31.0, 100)
        t1 = T1/10
        k = float(wildcards.k)
        base_params = PARAMETERS_LCT[wildcards.pathogen].update(epsilon_s=EPSILON_S if wildcards.pathogen=="SARS-CoV-2" else 0.0, epsilon_w=0.8, k=k)

        Rt_final, time_to_below, Itot, peak_Is, amplitudes, time_above, crossings, delay = compute_delay_metrics_grid(model=MODELS_LCT[wildcards.pathogen], base_params=base_params, taus_W=taus_W, taus_B=taus_B)

        plt.figure(figsize=FIGSIZE)
        plt.scatter(amplitudes, Itot, c=delay, alpha=0.2, s=2)
        plt.title('Effect of oscillations on the number of infections')
        plt.xlabel('Oscillation amplitudes')
        plt.ylabel('Total fraction infected')
        plt.savefig(f"{wildcards.outdir}/compartmental/true_vs_reported_Rt_{wildcards.pathogen}_k{wildcards.k}_scatter_amplitudes_vs_fractions.png"); plt.close()

        kwargs = dict(x_logscale=False, xlabel='Behavioural delay ($\\tau_B$)', ylabel='Reporting delay ($\\tau_W$)')
        scenario = f'({wildcards.pathogen}, $k={k:g}$)'
        fig, _ = plot_heatmap(taus_B, taus_W, amplitudes, cmap='magma', cbar_label='Amplitude of oscillations', title=f'Stability of delayed response', **kwargs)
        fig.savefig(output.amplitudes); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, time_above, cmap='cividis', cbar_label='Days above warning threshold', title=f'Time above warning threshold', **kwargs)
        fig.savefig(output.time_above); plt.close(fig)
        fig, _ = plot_heatmap(taus_B, taus_W, crossings, cmap='plasma', cbar_label='Total times warned', title=f'Number of warning-threshold crossings', **kwargs)
        fig.savefig(output.crossings); plt.close(fig)

        # per pathogen
        Rt_g, tRt_g, Itot_g, peakIs_g = {}, {}, {}, {}
        ps = PARAMETERS_LCT[pathogen]
        Rt, tRt, It, pk = Rt_final, time_to_below, Itot, peak_Is
        Rt_g[pathogen] = np.array(Rt)
        tRt_g[pathogen] = np.array(tRt)
        _, yy0 = MODELS_LCT[pathogen](params=ps.update(epsilon_s=0.0, epsilon_w=0.0), t1=t1)
        Itot_g[pathogen] = np.array(It) / float(yy0[0,0] - yy0[-1,0])
        Is0 = yy0[:, -(ps.n_W + ps.n_B + 2)]
        peakIs_g[pathogen] = np.array(pk) / float(np.max(Is0))

        cols = [ # title, label, data, cmap, center_at_one, log
            ('Rt_final', '$\\mathcal{R}_t$', Rt_g, 'RdBu_r', True, False),
            ('time_to_below', 'Time to $\\mathcal{R}_t<1$', tRt_g, 'magma', False, True),
            ('Itot', '$I_\\text{tot}$ (relative to baseline)', Itot_g, 'viridis', False, False),
            ('peak_Is', 'Peak $I_s$', peakIs_g, 'viridis', False, False),
        ]
        for col, (title, label, data, cmap, center_at_one, logscale) in enumerate(cols):
            vals = np.concatenate([d.ravel() for d in data.values()])
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if center_at_one:
                d = max(abs(vmin - 1.0), abs(vmax - 1.0))
                norm = Normalize(vmin=1.0 - d, vmax=1.0 + d)
            elif logscale:
                norm = LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
            fig, ax = plot_heatmap(taus_B, taus_W, data[pathogen], cmap=cmap, cbar_label=label, title=f'{label} ({pathogen})', norm=norm, **kwargs)
            if center_at_one: ax.contour(np.array(taus_B), np.array(taus_W), data[pathogen], levels=[1.0], colors='black', linewidths=1.0, linestyles='--')
            if title=='Rt_final': fig.savefig(output.Rt_final)
            elif title=='time_to_below': fig.savefig(output.time_to_below)
            elif title=='Itot': fig.savefig(output.Itot)
            elif title=='peak_Is': fig.savefig(output.peak_Is)
            plt.close(fig)

rule plot_gain_margins:
    output:
        plot="{outdir}/compartmental/gain_margins_{pathogen}_vary_{scenario}.png"
    run:
        ps = PARAMETERS[wildcards.pathogen].update(epsilon_w=EPSILON_W)
        if wildcards.scenario == "delays":
            xs = np.linspace(1.0, 31.0, 100)
            ys = np.linspace(1.0, 31.0, 100)
            L0 = loop_gain(ps.R_0 * ps.rho, ps.epsilon_w, ps.k, ps.R_crit)
            MG = np.array([[gain_margin(L0, tw, tb) for tw in xs] for tb in ys])
            xlabel = r'Behavioural delay $\tau_B$'
            ylabel = r'Reporting delay $\tau_W$'
            ticks = [1,2,3,4,5,10,20]
            vmin, vmax = float(np.nanmin(MG)), float(np.min([np.nanmax(MG),int(1e6)]))
            norm = LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
            fig, ax = plot_heatmap(xs, ys, MG, norm=norm, cmap='magma_r', cbar_ticks=ticks, contour_levels=[1.0], xlabel=xlabel, ylabel=ylabel, title='Gain margin')
        else: # vary efficacies
            eps_ww = np.linspace(0.0, 0.999, 100)
            MG = np.array([gain_margin(loop_gain(ps.R_0 * ps.rho, ew, ps.k, ps.R_crit), ps.tau_W, ps.tau_B) for ew in eps_ww])
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(eps_ww, MG, c='k')
            ax.set_xlabel('Warning response efficacy $\\varepsilon_w$')
            ax.set_ylabel('Gain margin')
            ax.set_title('Gain margin for varying warning efficacies')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.2)
        fig.savefig(output.plot); plt.close(fig)

rule plot_delay_margins:
    output:
        plot="{outdir}/compartmental/delay_margins_{pathogen}_vary_{scenario}.png"
    run:
        ps = PARAMETERS[wildcards.pathogen].update(epsilon_w=EPSILON_W)
        if wildcards.scenario == "delays":
            taus_W = np.linspace(1.0, 31.0, 100)
            taus_B = np.linspace(1.0, 31.0, 100)
            MD = np.array([[delay_margin(ps.update(tau_W=tw, tau_B=tb)) for tb in taus_B] for tw in taus_W])
            xlabel = r'Behavioural delay $\tau_B$'
            ylabel = r'Reporting delay $\tau_W$'
            fig, ax = plot_heatmap(taus_B, taus_W, MD, cmap='magma_r', contour_levels=[0.0], xlabel=xlabel, ylabel=ylabel, title='Delay margin')
        else: # vary efficacies
            eps_ww = np.linspace(0.0, 0.999, 100)
            MD = np.array([delay_margin(ps.update(epsilon_w=ew)) for ew in eps_ww])
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(eps_ww, MD, c='k')
            ax.set_xlabel = 'Warning response efficacy $\\varepsilon_w$'
            ax.set_ylabel = 'Isolation efficacy $\\varepsilon_s$'
            ax.set_title = 'Delay margin'
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.2)
        fig.savefig(output.plot); plt.close(fig)

rule plot_period_and_damping_scatter:
    output:
        period ="{outdir}/compartmental/period_scatter_{pathogen}_k{k}.png",
        damping="{outdir}/compartmental/damping_scatter_{pathogen}_k{k}.png",
    run:
        N = 50
        eps_w = 0.8
        k = float(wildcards.k)
        t1 = 300.0
        epsilon_s = EPSILON_S if wildcards.pathogen == "SARS-CoV-2" else 0.0
        base_params = PARAMETERS_LCT[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=eps_w, k=k)
        model = MODELS_LCT[wildcards.pathogen]
        n_W = base_params.n_W
        n_B = base_params.n_B
        taus_W = jnp.linspace(3.0, 31.0, N)
        taus_B = jnp.linspace(1.0, 31.0, N)

        # analytical
        analytical_period = np.full((N, N), np.nan)
        analytical_damping = np.full((N, N), np.nan)
        for i, tw in enumerate(np.array(taus_W)):
            for j, tb in enumerate(np.array(taus_B)):
                pole = dominant_pole(base_params.update(tau_W=tw, tau_B=tb))
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
        fig, ax = plt.subplots(figsize=FIGSIZE)
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
        plt.savefig(output.period); plt.close(fig)

        # damping scatterplot
        valid = np.isfinite(simulation_damping) & np.isfinite(analytical_damping)
        fig, ax = plt.subplots(figsize=FIGSIZE)
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
        plt.savefig(output.damping); plt.close(fig)

rule plot_hopf_boundary:
    output:
        plot="{outdir}/compartmental/hopf_boundary.png",
    run:
        ps = PARAMETERS["SARS-CoV-2"].update(epsilon_s=EPSILON_S, epsilon_w=EPSILON_W)
        eps_w = np.linspace(0.01, 1.0, 400)
        kc = k_crit(critical_loop_gain(ps.tau_W, ps.tau_B, ps.n_W, ps.n_B), eps_w, ps.R_crit)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.fill_between(eps_w, kc, 1e3, color="r", alpha=0.1, lw=0)
        ax.fill_between(eps_w, K_RANGE[0], K_RANGE[1], color="k", alpha=0.1, lw=0)
        ax.plot(eps_w, kc, "k-")
        ax.text(0.9, 60, "unstable", ha="right", va="top", color="r", fontsize=10)
        ax.text(0.3, 20, "stable", ha="left", va="bottom", color="0.3", fontsize=10)
        ax.text(0.5, 5, r"plausible range $k\in[%g,%g]$" % K_RANGE, ha="center", va="center", color="k", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(1, 100)
        ax.set_xlim(0.05, 1.0)
        ax.set_xlabel(r"Warning response efficacy $\varepsilon_w$", fontsize=12)
        ax.set_ylabel(r"Response sharpness $k$", fontsize=12)
        ax.set_title(r"Hopf boundary $M_G=1$", fontsize=14)
        fig.savefig(output.plot); plt.close(fig)


###############################################
# STOCHASTIC
###############################################

rule plot_stochastic_baseline_trajectories:
    output:
        plot ="{outdir}/gillespie/stochastic_baseline_trajectories_{pathogen}_N{N}.png",
    run:
        N = int(wildcards.N)
        t1 = T1/10
        num_simulations = 100
        ps = PARAMETERS[wildcards.pathogen].concrete()
        Iest = establishment_threshold(q=calculate_mt_branching_q(ps,0,0), alpha=ALPHA)

        fig, (ax_traj, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=FIGSIZE, sharex=True, height_ratios=[2,1])

        initial_fadeout_times = []
        established_extinction_times = []
        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
            initial_fadeout = np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest
            if initial_fadeout: initial_fadeout_times.append((tt[-1]))
            else: established_extinction_times.append((tt[-1]))
            ax_traj.plot(tt, yy.T[0], alpha=0.05, color='grey' if initial_fadeout else COLORS[wildcards.pathogen])

        ymax = max(max(initial_fadeout_times), max(established_extinction_times))
        tt_det, yy_det = MODELS[wildcards.pathogen](params=ps, t1=t1, E0=1/N)
        S_det = yy_det.T[0] * N
        ax_traj.plot(tt_det, S_det, color=COLORS[wildcards.pathogen])
        ax_traj.set_xlim([0, ymax])
        ax_traj.set_title('Number of susceptibles')
        ax_hist.hist([initial_fadeout_times, established_extinction_times], density=True, stacked=True, color=['grey', COLORS[wildcards.pathogen]], bins=int(ymax//10))
        ax_hist.set_title('Extinction times')
        ax_hist.set_xlabel('days')
        fig.suptitle(f'Stochastic susceptible trajectories (N={N})')
        fig.savefig(output.plot); plt.close(fig)

rule plot_linearised_branching_process_extinction_probabilities:
    output:
        plot="{outdir}/gillespie/linearised_branching_process_extinction_probabilities_{pathogen}.png",
        I_establishment="{outdir}/gillespie/I_establishment_{pathogen}.png",
    run:
        ps = PARAMETERS[wildcards.pathogen]
        eps_ww = np.linspace(0.0, 0.999, 100)
        eps_ss = np.linspace(0.0, 0.999, 100)
        Iest = np.zeros((len(eps_ww), len(eps_ss)))
        qs = np.zeros((len(eps_ww), len(eps_ss)))

        for i, ew in enumerate(eps_ww):
            for j, es in enumerate(eps_ss):
                q = calculate_mt_branching_q(ps, ew, es)
                qs[j,i] = q
                Iest[j,i] = establishment_threshold(q=q, alpha=ALPHA)

        fig, ax = plot_heatmap(eps_ww, eps_ss, qs, cmap='magma_r', 
            contour_metric=compute_R_grid(MODELS[wildcards.pathogen], PARAMETERS[wildcards.pathogen].update(k=1), eps_ww, eps_ss, T1), 
            contour_levels=[1.0], contour_colors='white',
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title='Linearised branching process extinction probabilities')
        plt.savefig(output.plot); plt.close()

        fig, ax = plot_heatmap(eps_ww, eps_ss, Iest, cmap='plasma', 
            norm=LogNorm(vmin=np.max([float(np.nanmin(Iest)),1]), vmax=np.min([np.max([float(np.nanmax(Iest)),1]), 1e9])),
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title='$I_\\text{establishment}$')
        plt.savefig(output.I_establishment); plt.close()

rule simulate_stochastic_outcomes:
    output:
        npz="{outdir}/gillespie/{scenario}_stochastic_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    run:
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

                ps = PARAMETERS[wildcards.pathogen].update(epsilon_s=float(es), epsilon_w=float(ew)).concrete()
                Iest = establishment_threshold(q=calculate_mt_branching_q(ps, ew, es), alpha=ALPHA)

                for k in range(num_simulations):
                    tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
                    tt, yy = to_uniform_grid(tt, yy, t1)
                    if (wildcards.scenario == 'establishment') & (np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest):
                        continue

                    Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = outcome_metrics(tt, yy, PARAMETERS[wildcards.pathogen].update(epsilon_s=es, epsilon_w=ew), t1, population_size=N)
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
            contour_metric=compute_R_grid(MODELS[wildcards.pathogen], PARAMETERS[wildcards.pathogen]._replace(k=1), eps_ww, eps_ss, T1, E0=1/N), 
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title={"Rt": "Average Final $R_t$", "Rt_var": "Variance of Final $R_t$", "time_to_below": "Average Time to $R_t < 1$", "time_to_below_var": "Variance of Time to $R_t < 1$", "Itot": "Average Proportion Infected", "Itot_var": "Variance of Proportion Infected", "peak_Is": "Average Peak Symptomatic Proportion", "peak_Is_var": "Variance of Peak Symptomatic Proportion", "extinction_time": "95th Percentile Extinction Time", "extinction_time_var": "Variance of Extinction Time"}.get(metric, metric)
        )
        fig.savefig(output.plot); plt.close(fig)

rule plot_stochastic_cumulative_extinction_probability:
    output:
        plot ="{outdir}/gillespie/cumulative_extinction_probability_{pathogen}_N{N}_epsS_{eps_s}_epsW{eps_w}_combined.png",
    run:
        N = float(wildcards.N)
        eps_s = float(wildcards.eps_s)
        eps_w = float(wildcards.eps_w)
        t1 = T1/2
        num_simulations = 1000
        initial_fadeout_times = []
        established_extinction_times = []
        ps = PARAMETERS[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w).concrete()
        Iest = establishment_threshold(q=calculate_mt_branching_q(ps, eps_w, eps_s), alpha=ALPHA)

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
        model = MODELS[wildcards.pathogen]
        ps_det = PARAMETERS[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w)
        tt_det, yy_det = model(params=ps_det, t1=t1, E0=1/N)
        S_det = yy_det.T[0]

        # subplots with shared x axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10), sharex=True)
        plot_extinction_probability_scenario(ax1, extinction_times_all, "All introductions", tt_det, S_det)
        plot_extinction_probability_scenario(ax2, extinction_times_est, "Established outbreaks", tt_det, S_det)
        ax2.set_xlabel('days', fontsize=12)
        if len(extinction_times_all) > 0: ax2.set_xlim(-50, max(extinction_times_all))
        plt.suptitle(f'Cumulative Extinction Probability ({wildcards.pathogen}, $\\varepsilon_s={eps_s}$, $\\varepsilon_w={eps_w}$)', fontsize=14, y=0.98)
        plt.savefig(output.plot); plt.close()

### SUPERSPREADING ###
rule plot_superspreading_baseline_trajectories:
    output:
        plot ="{outdir}/gillespie/superspreading_baseline_trajectories_{pathogen}_N{N}.png",
    run:
        N = int(wildcards.N)
        t1 = T1/10
        num_simulations = 100
        ps = PARAMETERS[wildcards.pathogen].concrete()
        q = calculate_mt_branching_q_with_superspreading(DISPERSION_SC2,ps,0,0)
        Iest = establishment_threshold(q=q, alpha=ALPHA)

        fig, (ax_traj, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=FIGSIZE, sharex=True, height_ratios=[2,1])
        initial_fadeout_times = []
        established_extinction_times = []
        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=DISPERSION_SC2, a_ss=True, p_ss=True, s_ss=True)
            initial_fadeout = np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest
            if initial_fadeout: initial_fadeout_times.append((tt[-1]))
            else: established_extinction_times.append((tt[-1]))
            ax_traj.plot(tt, yy.T[0], alpha=0.05, color='grey' if initial_fadeout else COLORS[wildcards.pathogen])
        try: 
            ymax = max(max(initial_fadeout_times, default=0), max(established_extinction_times, default=0))
            if ymax > 0:
                ax_traj.set_xlim([0, ymax])
                num_bins = max(1, int(ymax // 10))
                ax_hist.hist([initial_fadeout_times, established_extinction_times], density=True, stacked=True, color=['grey', COLORS[wildcards.pathogen]], bins=num_bins)
        except Exception as e: print(e)
        tt_det, yy_det = MODELS[wildcards.pathogen](params=ps, t1=t1, E0=1/N)
        S_det = yy_det.T[0] * N
        ax_traj.plot(tt_det, S_det, color=COLORS[wildcards.pathogen])
        ax_traj.set_title('Number of susceptibles')
        ax_hist.set_title('Extinction times')
        ax_hist.set_xlabel('days')
        fig.suptitle(f'Stochastic susceptible trajectories ({wildcards.pathogen}, N={N})')
        fig.savefig(output.plot); plt.close(fig)

rule plot_stochastic_cumulative_extinction_probability_superspreading:
    output:
        plot ="{outdir}/gillespie/superspreading_cumulative_extinction_probability_{pathogen}_N{N}_epsS_{eps_s}_epsW{eps_w}_scenario_{scenario}.png",
    run:
        N = float(wildcards.N)
        eps_s = float(wildcards.eps_s)
        eps_w = float(wildcards.eps_w)
        t1 = T1/2
        num_simulations = 1000
        ps = PARAMETERS[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w).concrete()
        # simulations
        initial_fadeout_times = []
        established_extinction_times = []
        Iest = establishment_threshold(q=calculate_mt_branching_q_with_superspreading(DISPERSION_SC2, ps, eps_w, eps_s), alpha=ALPHA)
        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=DISPERSION_SC2, a_ss=True, p_ss=True, s_ss=True)
            I = yy[:,2] + yy[:,3] + yy[:,4]
            if np.max(I) < Iest: initial_fadeout_times.append((tt[-1]))
            else: established_extinction_times.append((tt[-1]))
        if wildcards.scenario == 'establishment': extinction_times = established_extinction_times
        else: extinction_times = np.concatenate([established_extinction_times, initial_fadeout_times])
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
        model = MODELS[wildcards.pathogen]
        ps = PARAMETERS[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w)
        tt, yy = model(params=ps, t1=t1, E0=1/N)
        ax1.plot(tt, yy.T[0], color='green', label='Deterministic susceptible trajectory')
        # histogram
        ax2 = ax1.twinx()
        ax2.hist(extinction_times, bins=100, density=True, color='gray', alpha=0.3, label='Extinction times histogram')
        ax2.set_ylabel('Density', color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')
        # styling
        plt.title(f'Cumulative extinction probability ({wildcards.pathogen}, $k={DISPERSION_SC2}$, $\\varepsilon_s={eps_s}$, $\\varepsilon_w={eps_w}$, {wildcards.scenario})', fontsize=14)
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(-50, max(extinction_times))
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1+lines_2, labels_1+labels_2, loc='best')
        ax1.grid(True, alpha=0.5)
        plt.savefig(output.plot); plt.close()

rule simulate_superspreading_outcomes:
    output:
        npz="{outdir}/gillespie/{scenario}_superspreading_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    run:
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
        fig.savefig(output.plot); plt.close(fig)

rule simulate_superspreading_outcomes_all_superspreading:
    output:
        npz="{outdir}/gillespie/{scenario}_superspreading_all_outcomes_{pathogen}_N{N}_sims{num_simulations}_res{resolution}.npz",
    run:
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

                ps = PARAMETERS[wildcards.pathogen].update(epsilon_s=float(es), epsilon_w=float(ew)).concrete()
                Iest = establishment_threshold(q=calculate_mt_branching_q_with_superspreading(k_ss, ps, ew, es), alpha=ALPHA)

                for _ in range(num_simulations):
                    tt, yy = gillespie_SEIPAR_W_superspreading(params=ps, N=N, t1=t1, k_ss=k_ss, a_ss=True, p_ss=True, s_ss=False)
                    tt, yy = to_uniform_grid(tt, yy, t1)
                    if (wildcards.scenario == 'establishment') & (np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest):
                        continue

                    Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = outcome_metrics(tt, yy, PARAMETERS[wildcards.pathogen].update(epsilon_s=es, epsilon_w=ew), t1, population_size=N)
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
        fig.savefig(output.plot); plt.close(fig)


###############################################
# ALTERNATIVE WARNING SYSTEMS
###############################################
rule compute_alternative_warning_strategies_grid:
    output:
        data="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_k{k}_epsW{eps_w}_epsS{eps_s}.npz",
    run:
        pathogen = wildcards.pathogen
        data, taus_W, taus_B = strategy_grid(
            model=MODELS_PIECEWISE_LCT[pathogen], base_params=PARAMETERS_LCT[pathogen], k=float(wildcards.k), 
            eps_w=float(wildcards.eps_w), eps_s=float(wildcards.eps_s), strategies=STRATEGIES,
            t1=300.0, taus_W=np.linspace(1.0, 30.0, 100), taus_B=np.linspace(1.0, 30.0, 100), R_off=0.8, eval_interval=EVAL_INTERVAL,
        )
        strategies = list(STRATEGIES)
        np.savez_compressed(output.data, grid=np.stack([data[s] for s in strategies]), taus_W=np.asarray(taus_W), taus_B=np.asarray(taus_B), strategies=np.asarray(strategies), metrics=np.asarray(METRIC_NAMES))

rule alternative_warning_strategies_table:
    output:
        tex="{outdir}/compartmental/alternative_warning_strategies_table_{pathogen}.tex",
    run:
        ps = PARAMETERS_LCT[wildcards.pathogen].update(R_off=R_OFF, eval_interval=EVAL_INTERVAL)
        model = MODELS_PIECEWISE_LCT[wildcards.pathogen]
        scenarios = {key: val for key, val in WARNING_SCENARIOS[wildcards.pathogen].items() if key != "critical"}.items()
        rows = {s: [] for s, _ in scenarios}
        base = table_row_metrics(ps, model, 0.0, 0.0, T1, icrit=ICRIT)
        base["prevented"] = 0.0
        for scenario, (eps_s, eps_w) in scenarios:
            for strategy in list(STRATEGIES):
                strategy_params = ps.update(**{"lead": dict(T_lead=T_LEAD), "lockdown": dict(k=50.0, eval_interval=2*EVAL_INTERVAL)}.get(strategy, {}))
                if eps_s == 0.0 and eps_w == 0.0 and strategy == "baseline": m = base
                else: m = table_row_metrics(strategy_params, model, eps_s, eps_w, T1 * 5, itot_baseline=base["itot"], strategy=strategy, icrit=ICRIT)
                rows[scenario].append((strategy, eps_s, eps_w, m))
        with open(output.tex, "w") as f:
            f.write("\\begin{table}[H]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lcccccccc}\n\\toprule\n\\textbf{scenario} & $\\mathcal{R}_t$ & \\textbf{peak sympt.} & \\textbf{time to peak} & \\textbf{wave time} & \\textbf{attack rate} & \\textbf{inf. prevented} & \\textbf{isol. cost} & \\textbf{warn cost}\\\\\n\\midrule\n")
            for i, (scenario, (eps_s, eps_w)) in enumerate(scenarios):
                f.write(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{table_scenario_label(scenario, eps_s, eps_w, bold=True)}}}}}\\\\\n")
                for strategy, _, _, m in rows[scenario]:
                    tp, wt = f_days(m)
                    f.write(" & ".join([f"\\quad {strategy}", f"{m['Rt']:.2f}", f_pct(m["peak_Is"], 1), tp, wt, f_pct(m["itot"], 0), f_pct(m["prevented"], 0), f"{m['isolation_cost']:.1f}", f"{m['warning_cost']:.1f}",]) + " \\\\\n")
                if i < len(scenarios)-1: f.write("\\midrule\n")
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption[Characteristics of epidemic scenarios under different warning strategies]{Characteristics of epidemic scenarios under different warning strategies}\\label{tab:alternative_strategies}\n\\end{table}\n")

rule plot_alternative_warning_strategies:
    input:
        data="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_k{k}_epsW{eps_w}_epsS{eps_s}.npz",
    output:
        plot="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_k{k}_epsW{eps_w}_epsS{eps_s}.png",
    run:
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

        model = MODELS_PIECEWISE_LCT[wildcards.pathogen]
        base = PARAMETERS_LCT[wildcards.pathogen].update(epsilon_s=eps_s, epsilon_w=eps_w, k=k, R_off=0.8, eval_interval=EVAL_INTERVAL)

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
                im = ax.imshow(grid[c, :, :, r], origin="lower", aspect="auto", cmap="magma", extent=[taus_B[0], taus_B[-1], taus_W[0], taus_W[-1]]) #vmin=vmin, vmax=vmax, 
                if r == 0:
                    ax.set_title(strategies[c], fontsize=10)
                if c == 0:
                    ims[r] = im
                    ax.set_ylabel(f"{metric_names[r]}\n$\\tau_W$", fontsize=12)
                else: ax.tick_params(labelleft=False)
                if r == nM - 1: ax.set_xlabel("$\\tau_B$")
                else: ax.tick_params(labelbottom=False)

        # oscillations
        for c, s in enumerate(strategies):
            asym, disc, tl, ci = STRATEGIES[s]
            params = base.update(tau_W=14.0, tau_B=7.0, T_lead=tl, eval_interval=2*EVAL_INTERVAL if s=="lockdown" else EVAL_INTERVAL)
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
        plt.savefig(output.plot); plt.close()

rule plot_alternative_warning_strategies_eps_w:
    output:
        plot="{outdir}/compartmental/alternative_warning_strategies_{pathogen}_epsW.png",
    run:
        pathogen = wildcards.pathogen
        model = MODELS_PIECEWISE_LCT[pathogen]
        eps_ww = np.linspace(0.0, 1.0, 60)
        eps_s = EPSILON_S if pathogen=="SARS-CoV-2" else 0.0
        nM, nR = len(METRIC_NAMES), len(R_CRITS)

        fig, axs = plt.subplots(nrows=nR, ncols=nM, sharex=True, squeeze=False, figsize=(10, 10*nR/6))
        for row, rc in enumerate(R_CRITS):
            base_params = PARAMETERS_LCT[pathogen].update(epsilon_s=eps_s, R_crit=rc, R_off=rc-(1.0-R_OFF), eval_interval=EVAL_INTERVAL)
            bm = jnp.unstack(strategy_metrics(tau_W=base_params.tau_W, tau_B=base_params.tau_B, model=model, base_params=base_params, t1=T1, asymmetric=False, discrete_eval=False, check_interval=1.0, T_lead_on=False))
            baseline = np.array([1.0, bm[1], bm[2], 1.0, T1, T1])
            for i, (s, (asym, disc, tl, ci)) in enumerate(STRATEGIES.items()):
                y = np.zeros((nM, len(eps_ww)))
                for j, ew in enumerate(eps_ww):
                    p = base_params.update(epsilon_w=float(ew), T_lead=tl, eval_interval=2*EVAL_INTERVAL if s=="lockdown" else EVAL_INTERVAL)
                    y[:, j] = jnp.unstack(strategy_metrics(tau_W=p.tau_W, tau_B=p.tau_B, model=model, base_params=p, t1=T1, asymmetric=asym, discrete_eval=disc, check_interval=ci, T_lead_on=tl > 1e-3))
                for r in range(nM):
                    axs[row, r].plot(eps_ww, y[r] / baseline[r], ls=LINESTYLES[i], color=PALETTE[i], lw=1.3, alpha=0.9, label=s)
            for r in range(nM):
                axs[row, r].set_ylim(bottom=0.0); axs[row, r].set_xlim(0, 1)
                if row == 0: axs[row, r].set_title(METRIC_NAMES[r], fontsize=8)
                if row == nR - 1: axs[row, r].set_xlabel(r"$\varepsilon_w$")
            axs[row, 0].set_ylabel(r"$\mathcal{R}_\mathrm{crit}=%.1f$" % rc)
        axs[0, 0].legend(loc="lower left", fontsize=6)
        fig.legend(loc='outside lower center', ncol=len(list(STRATEGIES)), frameon=False, fontsize=10)
        fig.suptitle(rf"Warning strategies for varying response strengths ($\varepsilon_s={eps_s}$)")
        fig.savefig(output.plot); plt.close(fig)


###############################################
# SPATIAL
###############################################

rule plot_spatial_heatmaps:
    output:
        plot="{outdir}/spatial/spatial_heatmap_split_vs_migration.png",
    run:
        pathogen = "SARS-CoV-2"
        response_in_B_to_A=True
        eps_s = 0.0
        eps_w = 1.0
        t1 = T1/10
        epi_params = PARAMETERS[pathogen].update(epsilon_w=eps_w, epsilon_s=eps_s)

        N_A_grid = jnp.linspace(0.01, 0.99, 100)
        m_grid = jnp.concatenate([jnp.array([0.0]), jnp.logspace(-6, 0, 100)])
        run_grid = jax.vmap(jax.vmap(partial(run_spatial, epi_params=epi_params, response_in_B_to_A=response_in_B_to_A, model=MODEL_NAMES[pathogen]), in_axes=(None, 0)), in_axes=(0, None))
        Itot_A, Itot_B, peak_Is_A, peak_Is_B, total_infections = run_grid(N_A_grid, m_grid)

        # baseline without ww
        ts_bl, ys_bl = SPATIAL_MODELS[pathogen](SpatialParams(epi_params=epi_params.update(epsilon_w=0.0), N_A=1.0, m=0.0), t1=t1, E0=E0)
        d_ref = unpack_spatial(ys_bl, epi_params, ww_in_B=False, model=MODEL_NAMES[pathogen])
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
        fig.suptitle(f"Spatial model ({pathogen}, $\\varepsilon_s={epi_params.epsilon_s}$, $\\varepsilon_w={epi_params.epsilon_w}$, $\\tau_W={epi_params.tau_W}$, $\\tau_B={epi_params.tau_B:g}$)", y=1.0, fontsize=16)
        plt.savefig(output.plot)
        plt.close()

rule plot_spatial_trajectories:
    output:
        migration="{outdir}/spatial/spatial_trajectories_migration.png",
        split="{outdir}/spatial/spatial_trajectories_split.png",
    run:
        pathogen = "SARS-CoV-2"
        t1 = T1/10
        epi_params = PARAMETERS[pathogen].update(epsilon_w=0.4)

        # effect of migration rate at fixed 50/50 split
        m_values = [0.0, 0.01, 0.01, 0.05]
        N_A = 0.5
        fig, axs = plt.subplots(1, len(m_values), figsize=(4*len(m_values), 3.5), sharey=True)
        for ax, m in zip(axs, m_values):
            sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
            ts, ys = SPATIAL_MODELS[pathogen](sp, t1=t1, E0=E0, primary_in_A=False)
            d = unpack_spatial(ys, epi_params, model=MODEL_NAMES[pathogen])
            ax.plot(ts, d["Is_A"] / N_A, label="surveilled deme A", color="blue", alpha=0.7)
            ax.plot(ts, d["Is_B"] / (1-N_A), label="unsurveilled deme B", color="red", alpha=0.7)
            ax.set_title(f"m = {m}/day")
            ax.set_xlabel("time (days)")
        axs[0].set_ylabel("symptomatic prevalence")
        axs[0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Spatial model: effect of migration rate ($N_A=N_B=0.5$)", y=1.04, fontsize=16)
        plt.savefig(f"{output.migration}")
        plt.close()

        # effect of population split at fixed migration rate
        N_A_values = [0.0, 0.1, 0.5, 0.9]
        m = 0.01
        fig, axs = plt.subplots(1, len(m_values), figsize=(4*len(m_values), 3.5), sharey=True)
        for ax, N_A in zip(axs, N_A_values):
            sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
            ts, ys = SPATIAL_MODELS[pathogen](sp, t1=t1, E0=E0, primary_in_A=False)
            d = unpack_spatial(ys, epi_params, model=MODEL_NAMES[pathogen])
            ax.plot(ts, d["Is_A"] / N_A, label="surveilled deme A", color="blue", alpha=0.7)
            ax.plot(ts, d["Is_B"] / (1-N_A), label="unsurveilled deme B", color="red", alpha=0.7)
            ax.set_title(f"$N_A$ = {N_A}")
            ax.set_xlabel("time (days)")
        axs[0].set_ylabel("symptomatic prevalence")
        axs[0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Spatial model: effect of population split (m = {m}/day)", y=1.04, fontsize=16)
        plt.savefig(f"{output.split}")
        plt.close()

rule plot_Rt_spatial:
    input:
        phones=f'{DATADIR}/phones_CH/swiss_travellers_phones.feather',
        geo=f'{DATADIR}/map_CH/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp'
    output:
        plot="{outdir}/spatial/Rt_spatial_{canton_pairs}.png",
    run:
        pathogen = "SARS-CoV-2"
        t1 = 700.0
        response_in_B_to_A = True
        epi_params = PARAMETERS[pathogen].update(epsilon_w=EPSILON_W)

        pair_list = wildcards.canton_pairs.split("_")
        df, _ = load_and_preprocess_phone_data(data_path=input.phones, geo_path=input.geo)
        
        # columns are canton pairs, rows are holiday_types
        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=len(HOLIDAY_TYPES), ncols=len(pair_list), figsize=(12, 8), sharex=True, sharey=True)
        for i, ht in enumerate(HOLIDAY_TYPES):
            m_list, N_A_list = get_canton_pair_stats(df, wildcards.canton_pairs, 'weekday', ht)
            for j, (m, N_A) in enumerate(zip(m_list, N_A_list)):
                spatial_params = SpatialParams(epi_params=epi_params, m=m, N_A=N_A)
                tt, yy = SPATIAL_MODELS[pathogen](spatial_params=spatial_params, t1=t1, primary_in_A=False, response_in_B_to_A=response_in_B_to_A)
                c = unpack_spatial(yy, epi_params, model=MODEL_NAMES[pathogen], response_in_B_to_A=response_in_B_to_A)
                N_B = 1.0 - N_A
                rt_A = jnp.where(N_A > 0.0, epi_params.R_0 * epi_params.rho * c["B_A"][:, -1] * c["S_A"] / N_A, jnp.nan)
                rt_B = jnp.where(N_B > 0.0, epi_params.R_0 * epi_params.rho * c["B_B"][:, -1] * c["S_B"] / N_B, jnp.nan)
                rt_reported = c["W_A"][:, -1]
                rt_diff = jnp.mean(jnp.abs(rt_A - rt_B)) * 100

                ax = axs[i,j]
                if j == 0: ax.set_ylabel(ht, fontsize=16)
                if i == 0: ax.set_title(f"{pair_list[j][:2]} to {pair_list[j][2:]}", fontsize=16)
                ax.set_xticks([])
                ax.plot(tt, rt_A, color='blue', alpha=0.8)
                ax.plot(tt, rt_B, color='red', alpha=0.8)
                ax.plot(tt, rt_reported, color='black', alpha=0.8)
                ax.axhline(epi_params.R_crit, color='grey', linestyle='--')
                ax.text(250,1.7,rf"$m=${m:.4f}"+"\n"+rf"$N_A=${N_A:.4f}"+"\n"+"$\\Delta\\mathcal{R}_t=$"+rf"{rt_diff:.4f}%")

        fig.legend(
            [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='black', lw=2)],
            ['True $\\mathcal{R}_t$ in A', 'True $\\mathcal{R}_t$ in B', 'Reported $\\mathcal{R}_t$'],
            loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=16,
        )
        plt.savefig(output.plot); plt.close()


rule plot_Rt_divergence_heatmap:
    input:
        phones=f'{DATADIR}/phones_CH/swiss_travellers_phones.feather',
        geo=f'{DATADIR}/map_CH/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp'
    output:
        plot="{outdir}/spatial/spatial_heatmap_Rt_divergence_{pathogen}.png",
        csv="{outdir}/spatial/Rt_divergence_canton_pairs_{pathogen}.csv",
    run:
        pathogen = wildcards.pathogen
        epi_params = PARAMETERS[pathogen].update(epsilon_w=0.4)
        t1 = T1/10
        response_in_B_to_A = True

        N_A_grid = jnp.linspace(0.01, 0.99, 100)
        m_grid = jnp.logspace(-6, 0, 100)

        def rt_diff(N_A, m):
            sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
            _, yy = SPATIAL_MODELS[pathogen](sp, t1=t1, E0=E0, primary_in_A=False, response_in_B_to_A=response_in_B_to_A)
            c = unpack_spatial(yy, epi_params, model=MODEL_NAMES[pathogen], response_in_B_to_A=response_in_B_to_A)
            N_B = 1.0 - N_A
            rt_A = epi_params.R_0 * epi_params.rho * c["B_A"][:, -1] * c["S_A"] / N_A
            rt_B = epi_params.R_0 * epi_params.rho * c["B_B"][:, -1] * c["S_B"] / N_B
            return jnp.mean(jnp.abs(rt_A - rt_B))

        # get canton pairs
        df, _ = load_and_preprocess_phone_data(data_path=input.phones, geo_path=input.geo)
        all_zh_pairs = sorted(set(ALL_ZH_PAIRS.split("_")))
        all_ge_pairs = sorted(set(ALL_GE_PAIRS.split("_")))
        all_pairs = all_ge_pairs + all_zh_pairs
        pair_string = "_".join(all_pairs)
        m_list, N_A_list = get_canton_pair_stats(df, pair_string, "weekday", "workday")

        # create df
        rows = []
        for pair, m_raw, N_A in zip(all_pairs, m_list, N_A_list):
            m_eff = rescale_migration_rate(m_raw, MEAN_TRIP_DURATION)
            A = pair[:2]
            B = pair[2:]
            rows.append((pair, A, B, m_raw, m_eff, N_A, rt_diff(N_A, m_raw), rt_diff(N_A, m_eff)))
        columns = ["pair", "A", "B", "m_raw", "m_effective", "N_A", "divergence_raw", "divergence_effective"]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output.csv)

        # plot
        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, ax = plt.subplots(figsize=(6.5, 5))
        im = ax.pcolormesh(
            np.array(m_grid), np.array(N_A_grid), 
            np.array(jax.vmap(jax.vmap(rt_diff, in_axes=(None, 0)), in_axes=(0, None))(N_A_grid, m_grid)), 
            cmap="magma", shading="auto", vmin=0.0
        )
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("migration rate")
        ax.set_ylabel("population fraction in A")
        ax.set_title("Mean difference in $\\mathcal{R}_t$ between demes, "+pathogen, fontsize=12)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("average $\\mathcal{R}_t$ difference")

        df_plot = df.loc[df.groupby("B")["m_effective"].idxmax()]
        selected_pairs = CANTON_PAIRS.split("_")
        colors = []
        j = 0
        for i in range(len(all_pairs)):
            if all_pairs[i] in selected_pairs:
                colors.append(PALETTE[j])
                j += 1
            else: colors.append("white")
        # for j, ht in enumerate(HOLIDAY_TYPES):
        #     m_list, N_A_list = get_canton_pair_stats(df, pair_string, 'weekday', ht)
        #     for i, (m, N_A) in enumerate(zip(m_list, N_A_list)):
        #         plt.scatter(m, N_A, color=colors[i], marker=MARKERS[j], alpha=0.5, sizes=[40 if all_pairs[i] in selected_pairs else 10])
        color_handles = []
        for i, pair in enumerate(all_pairs):
            if colors[i]!="white": color_handles.append(Patch(facecolor=colors[i], label=f"{pair[:2]} to {pair[2:]}"))
        # marker_handles = [Line2D([0], [0], marker=MARKERS[j], color="grey", label=ht) for j, ht in enumerate(HOLIDAY_TYPES)]
        
        sns.scatterplot(data=df_plot, x="m_effective", y="N_A", hue="pair", palette=colors)
        ax.set_title(pathogen + " ($\\varepsilon_w=0.4$)")
        plt.legend(handles=color_handles, loc="lower right")
        fig.savefig(output.plot); plt.close(fig)


rule plot_migration:
    input:
        phones=f'{DATADIR}/phones_CH/swiss_travellers_phones.feather',
        geo=f'{DATADIR}/map_CH/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp'
    output:
        plot="{outdir}/spatial/migration.png",
    run:
        df, gdf = load_and_preprocess_phone_data(data_path=input.phones, geo_path=input.geo)
        fig, ax = plot_flows(df, gdf)
        plt.savefig(output.plot); plt.close()


###############################################
# INFECTIOUSNESS DISTRIBUTIONS / GENERATION TIME
###############################################

rule plot_infectiousness_distributions:
    output:
        plot="{outdir}/compartmental/infectiousness_distributions.png",
    run:
        ps = PARAMETERS["SARS-CoV-2"]
        nE, nP, nS, nA = 10,10,10,10
        shape = 8
        mean = 5.5
        scale = mean/shape

        def generation_time(nE, nP, nS, phi_p, w_p=None, w_s=None):
            def _infected_subsystem(t, y):
                E, Ip, Is = y[0:0+nE], y[nE:nE+nP], y[nE+nP:nE+nP+nS]
                dE, E_out = linear_chain(X=E, inflow=0.0, rate=nE/ps.gamma_inv)
                dIp, Ip_out = linear_chain(X=Ip, inflow=(1.0-ps.p)*E_out, rate=nP/ps.sigma_inv)
                dIs, Is_out = linear_chain(X=Is, inflow=Ip_out, rate=nS/ps.mu_s_inv)
                return np.concatenate([dE, dIp, dIs])

            sol = solve_ivp(_infected_subsystem, [0, tt[-1]], y0=np.concatenate([[1.0], np.zeros(nE+nP+nS-1)]), t_eval=tt, rtol=1e-10, atol=1e-13)
            Ip, Is = sol.y[nE:nE+nP], sol.y[nE+nP:nE+nP+nS]

            b = phi_p * (w_p[:, None] * Ip).sum(0) + (w_s[:, None] * Is).sum(0)
            gt = b / np.trapezoid(b, tt)
            mean = np.trapezoid(tt*gt, tt)
            return gt, mean

        tt = np.linspace(0, 1000, 1_000_000)
        colors = sns.color_palette("colorblind", 4)
        plt.figure(figsize=(8, 4))

        # Model generation times
        g_flat, m_flat = generation_time(1, 1, 1, ps.phi_p, np.full(nP, ps.mu_s_inv), np.full(nS, ps.sigma_inv))
        plt.plot(tt, g_flat, lw=2.2, color=colors[1], label=f'SEIPAR (mean {m_flat:.1f})')
        plt.fill_between(tt, g_flat, color=colors[1], alpha=0.2)

        g_chain_flat, m_chain_flat = generation_time(nE, nP, nS, ps.phi_p, np.full(nP, ps.mu_s_inv), np.full(nS, ps.sigma_inv))
        plt.plot(tt, g_chain_flat, lw=2.2, color=colors[2], label=f'SEIPAR-LCT (mean {m_chain_flat:.1f})')
        plt.fill_between(tt, g_chain_flat, color=colors[2], alpha=0.15)

        w_p, w_s = compute_weights(ps.gamma_inv, ps.sigma_inv, ps.mu_s_inv, shape, scale, nP, nS, ps.phi_p)
        g_weighted, m_weighted = generation_time(nE, nP, nS, ps.phi_p, w_p, w_s)
        plt.plot(tt, g_weighted, lw=2.2, color=colors[0], label=f'SEIPAR-LCT weighted (mean {m_weighted:.1f})')
        plt.fill_between(tt, g_weighted, color=colors[0], alpha=0.2)

        # Empirical Erlang-8 approximation
        pdf = erlang.pdf(tt, shape, scale=scale)
        plt.plot(tt, pdf, label=f'Erlang-{shape} (mean {mean})', color='black')
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
        plt.savefig(output.plot)
        plt.close()

rule lct_growth_rate:
    output:
        csv="{outdir}/lct/growth_comparison.csv",
    run:
        with open(output.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["pathogen", "model", "alpha", "doubling_time"])
            for p in ASYMPTOMATIC_PATHOGENS:
                for label, ps, gr in [("exponential", PARAMETERS[p], growth_rate), ("LCT", PARAMETERS_LCT[p], growth_rate_erlang)]:
                    alpha = float(gr(ps))
                    w.writerow([p, label, alpha, np.log(2)/alpha])


###############################################
# IDENTIFIABILITY
###############################################
rule run_remaster:
    input:
        xml="workflows/remaster.xml"
    output:
        traj="{outdir}/beast/remaster.traj",
        stats_trees="{outdir}/beast/remaster.stats.trees",
        typed_trees="{outdir}/beast/remaster.typed.trees",
    shell:
        "beast -overwrite -prefix {wildcards.outdir}/beast/ {input.xml}"



###############################################
# ALL
###############################################
rule all:
    input:
        # model
        expand(rules.plot_nonlinear_response_analysis.output.plot, outdir=OUTDIR),
        # generation time and infectiousness
        expand(rules.plot_infectiousness_distributions.output, outdir=OUTDIR),
        expand(rules.lct_growth_rate.output, outdir=OUTDIR),
        # basic results
        expand(rules.derived_epi_characteristics.output, outdir=OUTDIR),
        expand(rules.baseline_intervention_table.output, outdir=OUTDIR),
        expand(rules.plot_baseline_trajectories.output.plot, pathogen=PATHOGENS, outdir=OUTDIR),
        expand(rules.plot_trajectory.output.plot, outdir=OUTDIR, pathogen=PATHOGENS, epsilon_s=[0.0, 0.4, 0.8], epsilon_w=[0.0, 0.4, 0.8]),
        expand(rules.plot_main_intervention_grid.output.plot, outdir=OUTDIR),
        expand(rules.plot_R_1_contours.output.plot, outdir=OUTDIR),
        expand(rules.plot_combined_contour_grid_R1_Itot.output.plot, outdir=OUTDIR),
        expand(rules.delayed_ww_intervention.output.plot, outdir=OUTDIR, pathogen=["SARS-CoV-2"]),
        expand(rules.plot_controllability_boundaries.output.plot, outdir=OUTDIR),
        # asymptomaticity
        expand(rules.plot_asymptomatic_landscape.output.plot, outdir=OUTDIR),
        expand(rules.plot_asymptomatic_grid_Rt_final.output.plot, outdir=OUTDIR, pathogen=ASYMPTOMATIC_PATHOGENS),
        expand(rules.plot_asymptomatic_grid_Itot_final.output.plot, outdir=OUTDIR, pathogen=ASYMPTOMATIC_PATHOGENS),
        expand(rules.plot_asymptomatic_generation_time.output.plot, outdir=OUTDIR, pathogen=ASYMPTOMATIC_PATHOGENS),
        # sensitivity
        expand(rules.export_param_bounds.output.tex, outdir=OUTDIR, bounds=["empirical", "symmetric"]),
        expand(rules.plot_prcc_grid.output.plot, outdir=OUTDIR, bounds=["empirical", "symmetric"]),
        expand(rules.plot_prcc_monotonicity.output.plot, outdir=OUTDIR, pathogen=PATHOGENS, scenario=PRCC_SCENARIOS, outcome=PRCC_OUTCOMES, bounds=["empirical", "symmetric"]),
        expand(rules.plot_combined_sensitivity_grid.output.plot, outdir=OUTDIR),
        expand(rules.export_elasticities.output, outdir=OUTDIR),
        # stability
        expand(rules.plot_true_vs_reported_Rt_scenarios.output, pathogen=PATHOGENS, outdir=OUTDIR),
        expand(rules.plot_true_vs_reported_Rt_scenarios_piecewise.output, pathogen=["SARS-CoV-2"], outdir=OUTDIR, k=[10], scenario=list(STRATEGIES)),
        expand(rules.plot_true_vs_reported_Rt_heatmaps.output, pathogen=["SARS-CoV-2"], outdir=OUTDIR, k=[10]), #k=[1, 3, 10, 30],),
        expand(rules.plot_hopf_boundary.output, outdir=OUTDIR),
        expand(rules.plot_crossings.output, outdir=OUTDIR),
        expand(rules.plot_gain_margins.output.plot, outdir=OUTDIR, pathogen=PATHOGENS, scenario=["delays", "efficacies"]),
        expand(rules.plot_delay_margins.output.plot, outdir=OUTDIR, pathogen=PATHOGENS, scenario=["delays", "efficacies"]),
        expand(rules.plot_period_and_damping_scatter.output, outdir=OUTDIR, pathogen=PATHOGENS, k=[10, 30]),
        # stochasticity
        expand(rules.plot_linearised_branching_process_extinction_probabilities.output, outdir=OUTDIR, pathogen=["SARS-CoV-2"]),
        expand(rules.plot_stochastic_baseline_trajectories.output, outdir=OUTDIR, pathogen=["SARS-CoV-2"], N=[100, 50_000, 500_000]),
        expand(rules.plot_stochastic_cumulative_extinction_probability.output.plot, outdir=OUTDIR, pathogen=["SARS-CoV-2"], N=[10000], eps_s=[0.4], eps_w=[0.4, 0.8]),
        expand(rules.plot_superspreading_baseline_trajectories.output, outdir=OUTDIR, pathogen=["SARS-CoV-2"], N=[100, 50_000, 500_000]),
        expand(rules.plot_stochastic_cumulative_extinction_probability_superspreading.output.plot, outdir=OUTDIR, pathogen=["SARS-CoV-2"], N=[10000], eps_s=[0.4], eps_w=[0.4, 0.8], scenario=['establishment', 'all']),
        expand(rules.plot_stochastic_intervention_grid.output.plot, outdir=OUTDIR,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[10], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"]),
        expand(rules.plot_superspreading_intervention_grid.output.plot, outdir=OUTDIR,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[10], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"]),
        expand(rules.plot_superspreading_intervention_grid_all_superspreading.output.plot, outdir=OUTDIR,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[10], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"]),
        # alternative warning strategies
        expand(rules.alternative_warning_strategies_table.output, outdir=OUTDIR, pathogen=PATHOGENS),
        expand(rules.plot_alternative_warning_strategies_eps_w.output, outdir=OUTDIR, pathogen=PATHOGENS),
        expand(rules.plot_alternative_warning_strategies.output.plot, outdir=OUTDIR, pathogen=["SARS-CoV-2"], k=[10], eps_s=[0.5], eps_w=[0.0, 0.4, 0.8, 1.0]),
        # spatial model
        expand(rules.plot_migration.output, outdir=OUTDIR),
        expand(rules.plot_Rt_divergence_heatmap.output, outdir=OUTDIR, pathogen=PATHOGENS),
        expand(rules.plot_Rt_spatial.output, outdir=OUTDIR, canton_pairs=CANTON_PAIRS),
        expand(rules.plot_spatial_heatmaps.output, outdir=OUTDIR),
        expand(rules.plot_spatial_trajectories.output, outdir=OUTDIR),
        # identifiability
        expand(rules.run_remaster.output, outdir=OUTDIR),
