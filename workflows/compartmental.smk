import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import gamma
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from math import comb

from functools import partial
import itertools
import os
import concurrent.futures

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

from models.parameters import Params, logistic_response_function
from models.compartmental import simulate_SEIPAR_W, simulate_SEIAR_W, simulate_SEIR_W
from models.gillespie import gillespie_SEIPAR_W
from models.scenarios import compute_R_grid, compute_asymptomatic_grid_Rt, outcome_metrics, compute_metrics, compute_delay_metrics_grid
from models.prcc import SensitivityResults, run_sensitivity_analysis, partial_rank_residuals
from models.plotting import (
    plot_heatmap, plot_trajectory,
    plot_final_R, plot_I_tot, plot_I_tot_delayed_ww, plot_asymptomatic_effect_for_range_of_intervention_efficacies,
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
asymptomatic_pathogens = ["SARS-CoV-2", "Influenza A"]
E0 = 1e-6

best_params_kwargs = { # low R0, 1/sigma, 1/mu_a, p, phi; high 1/gamma, 1/mu_s
    "SARS-CoV-2": dict(R_0=2.40, gamma_inv=5.86, sigma_inv=0.52, mu_s_inv=10.0, mu_a_inv=4.63, p=0.230, phi=0.07),
    "Influenza A": dict(R_0=1.30, gamma_inv=3.12, mu_s_inv=4.69, mu_a_inv=2.06, p=0.33, phi=0.50),
    "Ebola": dict(R_0=1.74, gamma_inv=10.38, mu_s_inv=6.30)
}
worst_params_kwargs = { # low high 1/gamma, 1/mu_s; high R0, 1/sigma, 1/mu_a, p, phi
    "SARS-CoV-2": dict(R_0=2.98, gamma_inv=5.06, sigma_inv=3.00, mu_s_inv=7.80, mu_a_inv=5.50, p=0.399, phi=0.28),
    "Influenza A": dict(R_0=1.70, gamma_inv=2.28, mu_s_inv=2.06, mu_a_inv=4.69, p=0.33, phi=0.50),
    "Ebola": dict(R_0=2.15, gamma_inv=8.80, mu_s_inv=3.70)
}
colors = {"SARS-CoV-2": "tab:blue", "Influenza A": "tab:orange", "Ebola": "tab:green"}
p_CI = {"SARS-CoV-2": (0.23, 0.399), "Influenza A": (None, None)}
phi_CI = {"SARS-CoV-2": (0.07, 0.28), "Influenza A": (None, None)}

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

gillespie_popsizes = [10000] #, 1_000_000]
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
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], p_CI=p_CI[wildcards.pathogen], phi_CI=phi_CI[wildcards.pathogen], total_infected=False, path=output.plot, image_resolution=image_resolution, t1=Rt_times[wildcards.pathogen])

rule plot_asymptomatic_grid_Itot_final:
    output:
        plot="{outdir}/compartmental/asymptomatic_grid_Itot_final_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_asymptomatic_effect_for_range_of_intervention_efficacies(t1=600.0, model=models[wildcards.pathogen], params=parameters[wildcards.pathogen], p_CI=p_CI[wildcards.pathogen], phi_CI=phi_CI[wildcards.pathogen], total_infected=True, path=output.plot,image_resolution=image_resolution)

rule compute_prcc:
    output:
        npz="{outdir}/compartmental/prcc/data_{pathogen}_{scenario}_{outcome}.npz"
    run:
        os.makedirs(os.path.dirname(output.npz), exist_ok=True)
        p = wildcards.pathogen
        t1 = {'Rt': Rt_times, 'Itot': {patho: 600.0 for patho in pathogens}}[wildcards.outcome][p]
        best = best_params_kwargs[p]; worst = worst_params_kwargs[p]
        specific_bounds = {}
        for k in best.keys():
            l_val = min(best[k], worst[k]); u_val = max(best[k], worst[k])
            if l_val == u_val: u_val += 1e-5
            specific_bounds[k] = (l_val, u_val)
        results = run_sensitivity_analysis(model=models[wildcards.pathogen], base_params=parameters[wildcards.pathogen], manual_bounds=specific_bounds, scenario=wildcards.scenario, outcome=wildcards.outcome, t1=t1, E0=E0, n_lhs=5000, n_bootstrap=100, n_sobol_base=1024, avg_frac=0.1)
        np.savez_compressed(output.npz, param_names=np.array(results.param_names), lower_bounds=np.array([results.bounds[k][0] for k in results.param_names]), upper_bounds=np.array([results.bounds[k][1] for k in results.param_names]), samples=results.samples, outputs=results.outputs, prcc_mean=results.prcc_mean, prcc_lower=results.prcc_lower, prcc_upper=results.prcc_upper, prcc_samples=results.prcc_samples, sobol_S1=results.sobol_S1, sobol_S1_conf=results.sobol_S1_conf, sobol_ST=results.sobol_ST, sobol_ST_conf=results.sobol_ST_conf)

def load_sensitivity_results(path: str) -> SensitivityResults:
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["param_names"]]
    return SensitivityResults(param_names=names, bounds={n:(float(l),float(u)) for n,l,u in zip(names,d["lower_bounds"],d["upper_bounds"])}, samples=d["samples"], outputs=d["outputs"], prcc_mean=d["prcc_mean"], prcc_lower=d["prcc_lower"], prcc_upper=d["prcc_upper"], prcc_samples=d["prcc_samples"], sobol_S1=d["sobol_S1"], sobol_S1_conf=d["sobol_S1_conf"], sobol_ST=d["sobol_ST"], sobol_ST_conf=d["sobol_ST_conf"])

rule plot_prcc_monotonicity:
    input:
        npz=rules.compute_prcc.output.npz
    output:
        plot="{outdir}/compartmental/prcc/monotonicity_{pathogen}_{scenario}_{outcome}.png"
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
            # scatterplot partial rank residuals
            ex, ey = partial_rank_residuals(results.samples, results.outputs, i)
            ax.scatter(ex, ey, s=4, alpha=0.25, edgecolors='none', color=colors[wildcards.pathogen])
            ax.set_title(f'{PARAM_LABELS.get(name, name)} (PRCC $= {results.prcc_mean[i]:+.2f}$)', fontsize=18)
            # trendlines
            x_trend, y_trend = calculate_binned_means(ex, ey)
            ax.plot(x_trend, y_trend, color='black', lw=1.4, alpha=0.8)
            # styling
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(0, color='gray', lw=0.4, alpha=0.5); ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
        # remove unused subplots
        for j in range(d, n_rows*n_cols): axs[j//n_cols, j%n_cols].axis('off')

        # labels and titles
        fig.supxlabel('partial rank of parameter', fontsize=24)
        fig.supylabel('partial rank of output', fontsize=24)
        scenario_title = prcc_scenario_titles[wildcards.scenario]
        outcome_title = prcc_outcome_titles[wildcards.outcome]
        fig.suptitle(f'{wildcards.pathogen}, {scenario_title}, {outcome_title}', fontsize=32)
        
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)

rule plot_prcc_grid:
    input:
        npz=lambda wc: expand("{outdir}/compartmental/prcc/data_{pathogen}_{scenario}_{outcome}.npz", outdir=wc.outdir, pathogen=pathogens, scenario=['start', 'threshold'], outcome=prcc_outcomes)
    output:
        plot="{outdir}/compartmental/combined_prcc_grid.png"
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
                results_start = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_start_{outcome}.npz")
                results_threshold = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_threshold_{outcome}.npz")
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
        npz=lambda wc: expand("{outdir}/compartmental/prcc/data_{pathogen}_{scenario}_{outcome}.npz", outdir=wc.outdir, pathogen=pathogens, scenario=['start', 'threshold'], outcome=prcc_outcomes)
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
                results_start = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_start_{outcome}.npz")
                results_threshold = load_sensitivity_results(f"{wildcards.outdir}/compartmental/prcc/data_{pathogen}_threshold_{outcome}.npz")
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
        npzs=expand(rules.compute_prcc.output.npz, pathogen=pathogens, scenario=prcc_scenarios, outcome="Itot", outdir=outdir)
    output:
        tex="{outdir}/compartmental/prcc/sensitivity_bounds_table.tex"
    run:
        os.makedirs(os.path.dirname(output.tex), exist_ok=True)
        combinations = list(itertools.product(pathogens, prcc_scenarios))
        col_mapping = {("SARS-CoV-2", "start"): "SARS-CoV-2", ("SARS-CoV-2", "threshold"): "SARS-CoV-2 $I_\\text{crit}$", ("Influenza A", "start"): "H1N1", ("Influenza A", "threshold"): "H1N1 $I_\\text{crit}$", ("Ebola", "start"): "Ebola", ("Ebola", "threshold"): "Ebola $I_\\text{crit}$"}
        param_defs = {
            "R_0": ("$\\mathcal{R}_0$", "basic reproductive number"), "gamma_inv": ("$1/\\gamma$", "latent period"), "mu_s_inv": ("$1/\\mu_s$", "symptomatic period"), 
            "sigma_inv": ("$1/\\sigma$", "presymptomatic period"), 
            "mu_a_inv": ("$1/\\mu_a$", "asymptomatic period"), "p": ("$p$", "proportion asymptomatic"), "phi": ("$\\varphi$", "relative asympt. infectiousness"), 
            "epsilon_s": ("$\\varepsilon_s$", "isolation efficacy"), "epsilon_w": ("$\\varepsilon_w$", "warning response efficacy"), "tau_W": ("$\\tau_W$", "reporting delay"), "tau_B": ("$\\tau_B$", "behavioural delay"), 
            "log_k": ("$\\log_{10} k$", "warning gate sharpness"), "R_crit": ("$\\mathcal{R}_{\\text{crit}}$", "warning threshold"), 
            "log_k_I": ("$\\log_{10} k_I$", "prevalence gate sharpness"), "log_I_crit": ("$\\log_{10} I_{\\text{crit}}$", "prevalence threshold"), 
        }
        bounds_data = {}
        for combination, fpath in zip(combinations, input.npzs): bounds_data[combination] = load_sensitivity_results(fpath).bounds
        with open(output.tex, 'w') as f:
            f.write("\\begin{table}\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{llcccccc}\n\\toprule\n")
            header = ["Parameter", ""] + [col_mapping[c] for c in combinations]
            f.write(" & ".join(header) + " \\\\\n\\midrule\n")
            for p_key, (symbol, desc) in param_defs.items():
                row = [desc, symbol]
                for combination in combinations: 
                    if p_key in bounds_data[combination]: row.append(f"$[{bounds_data[combination][p_key][0]:g}, {bounds_data[combination][p_key][1]:g}]$")
                    else: row.append("---")
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\caption{Parameter ranges used for Latin hypercube sampling in the global sensitivity analysis.}\n\\label{tab:prcc-bounds}\n\\end{table}\n")

rule calculate_generation_times:
    output:
        txt="{outdir}/compartmental/generation_times.txt"
    run:
        os.makedirs(os.path.dirname(output.txt), exist_ok=True)
        def _get_generation_time(ps: Params):
            nom = ps.p * ps.phi * ps.mu_a_inv**2 +(1-ps.p)*(ps.sigma_inv**2 + ps.mu_s_inv**2 + ps.sigma_inv*ps.mu_s_inv)
            denom = ps.p * ps.phi * ps.mu_a_inv + (1-ps.p)*(ps.sigma_inv + ps.mu_s_inv)
            return ps.gamma_inv + nom / denom

        with open(output.txt, 'w') as f:
            for pathogen in pathogens:
                f.write(f"{pathogen}: {_get_generation_time(parameters[pathogen])}\n")


trajectory_end_times = {"SARS-CoV-2": 530, "Influenza A": 874, "Ebola": 1820} # 5x total wave time, rounded to nearest 10

rule plot_trajectory:
    output:
        plot="{outdir}/compartmental/trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        params = parameters[wildcards.pathogen].update(epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w))
        plot_trajectory(t1=trajectory_end_times[wildcards.pathogen], model=models[wildcards.pathogen], params=params, path=output.plot, title=f"{wildcards.pathogen} ($\\varepsilon_s={wildcards.epsilon_s}, \\varepsilon_w={wildcards.epsilon_w}$)", image_resolution=image_resolution, plot_total_I=True)

rule plot_trajectory_delayed_ww_intervention:
    output:
        plot="{outdir}/compartmental/delayed_ww_intervention_trajectory_{pathogen}_epss{epsilon_s}_epsw{epsilon_w}_Icrit{I_crit}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        params = parameters[wildcards.pathogen].update(I_crit=float(wildcards.I_crit), epsilon_s=float(wildcards.epsilon_s), epsilon_w=float(wildcards.epsilon_w))
        plot_trajectory(model=models[wildcards.pathogen], params=params, path=output.plot, title=f"Trajectory: {wildcards.pathogen} (eps_s={wildcards.epsilon_s}, eps_w={wildcards.epsilon_w})", t1=600.0, image_resolution=image_resolution, plot_total_I=True, semilogy=True)

# TODO: currently vary tau_W delay, tau_B is default
rule delayed_ww_intervention:
    output:
        plot="{outdir}/compartmental/delay_grid_ww_intervention_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        fig = plot_I_tot_delayed_ww(model=simulate_SEIPAR_W, parameters=Params.for_SEIPAR(epsilon_s=0.0, epsilon_w=0.8))
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule baseline_trajectories:
    output:
        plot="{outdir}/compartmental/baseline_trajectories_{pathogen}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        peak_Is, time_to_peak, total_time = plot_trajectory(model = models[wildcards.pathogen], params = parameters[wildcards.pathogen].update(I_crit=1e-4), path = output.plot, title = f"{wildcards.pathogen}", image_resolution = image_resolution, plot_total_I = True, t1 = 500.0)
        print(wildcards.pathogen); print(peak_Is, time_to_peak, total_time)

rule baseline_trajectories_no_asymptomatic:
    output:
        plot="{outdir}/compartmental/trajectories_no_asymptomatic_{pathogen}.png"
    run:    
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plot_trajectory(model = models[wildcards.pathogen], params = {"SARS-CoV-2": Params.for_SEIPAR(p=0.0, phi=0.0), "Influenza A": Params.for_SEIAR(p=0.0, phi=0.0), "Ebola": Params.for_SEIR(),}[wildcards.pathogen], path = output.plot, title = f"No asymptomatic transmission for {wildcards.pathogen}", image_resolution = image_resolution)

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

        epsilon_s = 0.5 if wildcards.pathogen == "SARS-CoV-2" else 0.0
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

rule plot_true_vs_reported_Rt_scenarios_vary_k:
    output:
        plot="{outdir}/compartmental/true_vs_reported_Rt_{pathogen}_tauW{tau_W}_tauB{tau_B}_scenarios.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)

        ks = [1.0, 5.0, 10.0, 50.0, 100.0]
        tau_W = float(wildcards.tau_W)
        tau_B = float(wildcards.tau_B)
        t1 = 300.0

        epsilon_s = 0.5 if wildcards.pathogen == "SARS-CoV-2" else 0.0
        base_params = parameters[wildcards.pathogen].update(epsilon_s=epsilon_s, epsilon_w=0.8, tau_W=tau_W, tau_B=tau_B)
        model = models[wildcards.pathogen]

        sns.set_theme(style="white", rc={"axes.grid": False})
        fig, axs = plt.subplots(nrows=1, ncols=len(ks), figsize=(3*len(ks), 4), sharex=True, sharey=True)

        for i, k in enumerate(ks):
            params = base_params.update(k=k)
            tt, yy = model(params=params, t1=t1)
            rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
            rt_reported = yy[:, -(params.n_B + 1)]
            above = (rt_reported >= params.R_crit).astype(jnp.float32)
            total_time_above = float(above.mean() * t1)
            num_crossings = int(jnp.sum(jnp.diff(above) > 0))

            ax = axs[i]
            ax.set_title(f'$k={k}$', fontsize=16)
            ax.plot(tt, rt_true, color='black')
            ax.plot(tt, rt_reported, color='red')
            ax.axhline(params.R_crit, color='grey', linestyle='--')
            ax.text(0.97, 0.1, f'{total_time_above:.0f} days above $R_{{crit}}$\n{num_crossings} warnings', transform=ax.transAxes, ha='right', va='top', fontsize=8)

        fig.suptitle(f'{wildcards.pathogen}: $\\tau_W={params.tau_W:g}$, $\\tau_B={params.tau_B:g}$', fontsize=18, y=0.995)
        fig.legend(
            [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='red', lw=2)],
            ['True $R_t$', 'Reported $R_t$'],
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=16,
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
                norm = mpl.colors.Normalize(vmin=1.0 - d, vmax=1.0 + d)
            elif logscale:
                norm = mpl.colors.LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

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
            if title=='Rt_final':
                fig.savefig(output.Rt_final, dpi=image_resolution, bbox_inches='tight')
            elif title=='time_to_below':
                fig.savefig(output.time_to_below, dpi=image_resolution, bbox_inches='tight')
            elif title=='Itot':
                fig.savefig(output.Itot, dpi=image_resolution, bbox_inches='tight')
            elif title=='peak_Is':
                fig.savefig(output.peak_Is, dpi=image_resolution, bbox_inches='tight')
            plt.close(fig)

rule plot_main_intervention_grid:
    output:
        plot="{outdir}/compartmental/main_intervention_grid.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_ww = jnp.linspace(0.0, 0.999, 100)
        eps_ss = jnp.linspace(0.0, 0.999, 100)
        t1 = 10000.0

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
            ('Peak $I_s$', peakIs_g, 'viridis', False, False),
        ]
        fig, axs = plt.subplots(nrows=len(rows), ncols=len(pathogens), figsize=(13, 16), sharex=True, sharey=True)
        for row_idx, (label, data, cmap, center_at_one, logscale) in enumerate(rows):

            # normalisation
            vals = np.concatenate([d.ravel() for d in data.values()])
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if center_at_one:
                d = max(abs(vmin - 1.0), abs(vmax - 1.0))
                norm = mpl.colors.Normalize(vmin=1.0 - d, vmax=1.0 + d)
            elif logscale:
                norm = mpl.colors.LogNorm(vmin=np.max([vmin,1]), vmax=np.max([vmax,1]))
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
        linestyles = ['--', '-', '-.', ':']

        fig, (ax_R, ax_I) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        for pathogen in pathogens:
            model = models[pathogen]
            base_params = parameters[pathogen]
            tRt = Rt_times[pathogen]
            color = colors[pathogen]
            _, yy0 = model(params=base_params.update(epsilon_s=0.0, epsilon_w=0.0), t1=t1, E0=E0)
            baseline_Itot = yy0[0,0] - yy0[-1,0]
            for i, r_crit in enumerate(R_crits):
                Rt_grid, _, Itot_grid, _ = compute_metrics(model, base_params.update(R_crit=r_crit), eps_ww, eps_ss, t1, E0)
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
            t1 = Rt_times[pathogen]
            shade_map = {0.0: 'red', 0.2: 'orange', 0.4: 'yellow', 0.6:'lime', 0.8: 'green'}

            # basic nonsymptomatic fraction heatmap
            P, PHI = np.meshgrid(np.array(ps), np.array(phis), indexing='xy')
            Ra = P * PHI * base.mu_a_inv
            Rp = (1.0 - P) * base.sigma_inv
            Rs = (1.0 - P) * base.mu_s_inv
            mesh = ax.pcolormesh(np.array(ps), np.array(phis), (Ra + Rp) / (Ra + Rp + Rs), cmap='Greys', vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
            cbar = fig.colorbar(mesh)

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
            ax.set_xlabel(r'Proportion asymptomatic, $p$', fontsize=12)
            ax.set_title(pathogen, fontsize=14, pad=8)
        axs[0].set_ylabel(r'Relative infectiousness, $\varphi$', fontsize=12)

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

        fig.suptitle(r'Controllability boundary ($\mathcal{R}_t=1$) for varying asymptomaticity', fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


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
            title='Influence of asymptomatic transmission on generation time',
            xlabel=r'Proportion asymptomatic, $p$', ylabel=r'Relative infectiousness, $\varphi$',
        )
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_nonlinear_response_analysis:
    output:
        plot="{outdir}/compartmental/nonlinear_response_analysis.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        dt = 0.1
        t = np.arange(0, 50, dt)
        n_W, tau_W = 3.0, 14.0
        eps_w, k, threshold = 0.5, 10.0, 1.0
        n_B, tau_B = 1.0, 7.0

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), width_ratios=(1,2))

        # Reporting delay
        reporting_delay = gamma.pdf(t, a=n_W, scale=tau_W/n_W)
        axes[0, 0].plot(t, reporting_delay, color='purple', linewidth=2)
        axes[0, 0].fill_between(t, reporting_delay, alpha=0.1, color='purple')
        axes[0, 0].axvline(tau_W, color='purple', linestyle=':', linewidth=2, label=f'Mean: $\\tau_W={tau_W:.0f}$')
        axes[0, 0].set_title(f'Reporting Delay ($n_W={n_W}$)')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 0.06)
        axes[0, 0].grid(True, alpha=0.3)

        # Logistic response
        x_pure = np.linspace(0, 3.5, 400)
        y_pure = 1 - (eps_w / (1 + np.exp(-k * (x_pure - threshold))))
        axes[1, 0].plot(x_pure, y_pure, color='black', linewidth=2)
        axes[1, 0].axvline(threshold, color='grey', linestyle='--', linewidth=2, label=r'$\mathcal{R}_\text{crit}=1.0$')
        # 95% interval
        p_low, p_high = 0.025, 0.975 
        x_low, x_high = threshold + (1 / k) * np.log(p_low / (1 - p_low)), threshold + (1 / k) * np.log(p_high / (1 - p_high))
        axes[1, 0].axvspan(x_low, x_high, color='gray', alpha=0.1, label=f'95%: [{x_low:.2f} - {x_high:.2f}]')
        axes[1, 0].set_title(f'Logistic Response ($k={k}$)')
        axes[1, 0].set_xlabel('Reproductive number')
        axes[1, 0].set_ylim(0, 1.2)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Behavioural delay
        beh_delay = gamma.pdf(t, a=n_B, scale=tau_B/n_B)
        axes[2, 0].plot(t, beh_delay, color='brown', linewidth=2)
        axes[2, 0].fill_between(t, beh_delay, alpha=0.1, color='brown')
        axes[2, 0].axvline(tau_B, color='brown', linestyle=':', linewidth=2, label=f'Mean: $\\tau_B={tau_B:.0f}$')
        axes[2, 0].set_title(f'Behavioural Delay ($n_B={n_B}$)')
        axes[2, 0].set_xlabel('Days')
        axes[2, 0].legend()
        axes[2, 0].set_ylim(0, 0.15)
        axes[2, 0].grid(True, alpha=0.3)

        # Combined response
        def total_response(amp):
            x = amp * gamma.cdf(t, a=n_W, scale=tau_W/n_W)
            y = 1 - (eps_w / (1 + np.exp(-k * (x - threshold))))
            z_padded = np.convolve(np.concatenate([np.ones(len(t)), y]), gamma.pdf(t, a=n_B, scale=tau_B/n_B), mode='full') * dt
            z = z_padded[len(t) : 2 * len(t)]
            return x, y, z
        
        for p in pathogens:
            x_m, y_m, z_m = total_response(parameters[p].R_0)
            x_l, y_l, z_l = total_response(best_params_kwargs[p]["R_0"])
            x_h, y_h, z_h = total_response(worst_params_kwargs[p]["R_0"])
            color = colors[p]
            axes[0, 1].plot(t, x_m, color=color, linewidth=2, label=p)
            axes[0, 1].fill_between(t, x_l, x_h, color=color, alpha=0.2)
            axes[1, 1].plot(t, y_m, color=color, linewidth=2)
            axes[1, 1].fill_between(t, np.minimum(y_l, y_h), np.maximum(y_l, y_h), color=color, alpha=0.2)
            axes[2, 1].plot(t, z_m, color=color, linewidth=2)
            axes[2, 1].fill_between(t, np.minimum(z_l, z_h), np.maximum(z_l, z_h), color=color, alpha=0.2)
        axes[0, 1].axhline(threshold, color='grey', linestyle='--', linewidth=2, label=r'$\mathcal{R}_\text{crit}=1.0$')
        axes[1, 1].axhline(eps_w, color='grey', linestyle='--', linewidth=2, label=r'$\varepsilon_w=0.5$')
        axes[2, 1].axhline(eps_w, color='grey', linestyle='--', linewidth=2, label=r'$\varepsilon_w=0.5$')

        # Formatting
        axes[0, 1].set_title('Reported Reproductive Number')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title(f'Instantaneous Warning Response ($\epsilon_w={eps_w}$)')
        axes[1, 1].set_ylim(0, 1.2)
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('Effective Transmission Modification')
        axes[2, 1].set_xlabel('Days')
        axes[2, 1].set_ylim(0, 1.2)
        axes[2, 1].legend(loc='upper right')
        axes[2, 1].grid(True, alpha=0.3)
        fig.suptitle("Wastewater Warning Response Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output.plot, dpi=image_resolution); plt.close(fig)

rule plot_asymptomatic_landscape:
    output:
        plot="{outdir}/compartmental/asymptomatic_landscape.png"
    run:
        def sample(mean_params, best_params, worst_params, n_samples, seed=0):
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

        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        plt.figure(figsize=(6,6))
        for pathogen in pathogens:
            mean_params = parameters[pathogen]
            best_params = mean_params.update(**best_params_kwargs[pathogen])
            worst_params = mean_params.update(**worst_params_kwargs[pathogen])
            R0_s, theta_s = sample(mean_params=mean_params, best_params=best_params, worst_params=worst_params, n_samples=10000)
            plt.scatter(theta_s, R0_s, s=4, alpha=0.1, color=colors[pathogen], edgecolors='none')

        plt.xlim([-0.01, 1])
        plt.ylim([0, 5])
        plt.xlabel('Proportion presymptomatic and asymptomatic')
        plt.ylabel('Basic reproductive number')
        plt.legend(handles=[Patch(facecolor=colors[pathogen], label=pathogen) for pathogen in pathogens], loc='upper right')
        plt.savefig(output.plot, dpi=image_resolution); plt.close()


def arg_L(omega, tau_W, tau_B, n_W=3, n_B=1):
    return -n_W*np.arctan(omega*tau_W/n_W) - n_B*np.arctan(omega*tau_B/n_B)

rule plot_gain_margins:
    output:
        plot="{outdir}/compartmental/gain_margins.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_w = 0.8
        k = 10.0
        n_W = 3
        n_B = 1

        def gain_margin(tau_W, tau_B):
            def _omega_PC(tau_W, tau_B):
                return brentq(lambda w: np.pi + arg_L(omega=w,tau_W=tau_W,tau_B=tau_B,n_W=n_W,n_B=n_B), 1e-10, 1000.0)
            omega_PC = _omega_PC(tau_W, tau_B)
            if omega_PC is None: return np.inf
            return (2*(2-eps_w))/(eps_w*k) * (1+(omega_PC*tau_W/n_W)**2)**(n_W/2) * (1+(omega_PC*tau_B/n_B)**2)**(n_B/2)

        taus_W = np.linspace(1.0, 31.0, 100)
        taus_B = np.linspace(1.0, 31.0, 100)
        MG = np.array([[gain_margin(tw, tb) for tb in taus_B] for tw in taus_W])
        fig, ax = plot_heatmap(taus_B, taus_W, MG, cmap='magma_r', contour_levels=[0.0], xlabel=r'Behavioural delay ($\tau_B$)', ylabel=r'Reporting delay ($\tau_W$)', title='Gain margin')
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)


rule plot_delay_margins:
    output:
        plot="{outdir}/compartmental/delay_margins.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        eps_w = 0.8
        k = 10.0
        n_W = 3
        n_B = 1
        L0 = (eps_w*k)/(2*(2-eps_w))

        def delay_margin(tau_W, tau_B):
            def _omega_c(tau_W, tau_B):
                def g(omega): return (L0**2 * (1 + (omega*tau_W/n_W)**2)**(-n_W) * (1 + (omega*tau_B/n_B)**2)**(-n_B) - 1)
                if g(0) <= 0: return None
                return brentq(g, 1e-10, 1000.0)
            omega_c = _omega_c(tau_W, tau_B)
            print(_omega_c(14,7))
            if omega_c is None: return np.inf
            return (np.pi - arg_L(omega=omega_c, tau_W=tau_W, tau_B=tau_B, n_W=n_W, n_B=n_B)) / omega_c

        taus_W = np.linspace(1.0, 31.0, 100)
        taus_B = np.linspace(1.0, 31.0, 100)
        MD = np.array([[delay_margin(tw, tb) for tb in taus_B] for tw in taus_W])
        fig, ax = plot_heatmap(taus_B, taus_W, MD, cmap='magma_r', contour_levels=[0.0], xlabel=r'Behavioural delay ($\tau_B$)', ylabel=r'Reporting delay ($\tau_W$)', title='Delay margin')
        fig.savefig(output.plot, dpi=image_resolution); plt.close(fig)

# TODO: move functions to new stability script
def _characteristic_polynomial(tau_W, tau_B, eps_w, k, n_W=3, n_B=1):
    """pW * pB + L0 = 0."""
    P = np.convolve(
        np.array([comb(n_W, j) * (tau_W/n_W)**j for j in range(n_W+1)]),
        np.array([comb(n_B, j) * (tau_B/n_B)**j for j in range(n_B+1)]))
    P[0] += (eps_w * k) / (2 * (2-eps_w))
    return P

def _dominant_pole(tau_W, tau_B, eps_w, k, n_W=3, n_B=1):
    """Dominant complex root of characteristic polynomial."""
    roots = np.roots(_characteristic_polynomial(tau_W, tau_B, eps_w, k, n_W, n_B)[::-1])
    complex_roots = roots[np.abs(roots.imag) > 1e-9]
    if complex_roots.size == 0: return np.nan
    return complex_roots[np.argmax(complex_roots.real)]

@partial(jax.jit, static_argnames=['model', 't1'])
def _compute_rt_grid(model, base_params, taus_W, taus_B, t1=300.0):
    """True Rt in (tau_W, tau_B)."""
    def _rt(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        _, yy = model(params=params, t1=t1)
        return params.R_0 * params.rho * yy[:,-1] * yy[:,0]
    return jax.vmap(jax.vmap(_rt, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

def _period_and_damping(t, x, t0=50.0, t1=250.0, smoothing_days=20.0, peak_threshold=0.2, T_min=4.0, T_max=200.0):
    """Period and damping rate from trajectory."""
    t_m = t[(t>t0) & (t<t1)]
    x_m = x[(t>t0) & (t<t1)]
    dt = float(t_m[1] - t_m[0])

    x_m = x_m - gaussian_filter1d(x_m, sigma=smoothing_days/dt) # Gaussian smoothing
    x_m = x_m - x_m.mean() # normalise around 0
    if x_m.std() < 1e-9: return np.nan, np.nan # no oscillations

    # period from autocorrelation
    ac = np.correlate(x_m, x_m, mode='full')[len(x_m)-1:]
    ac = ac/ac[0]
    period, i_peak1, offset1, denom1 = np.nan, -1, 0.0, 0.0
    for i in range(max(2, int(T_min/dt)), min(len(ac)-1, int(T_max/dt))):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > peak_threshold: # first peak above threshold
            # parabolic interpolation with peak at 0.5*(ac[i-1] - ac[i+1]) / (ac[i-1] - 2*ac[i] + ac[i+1])
            denom1 = ac[i-1] - 2*ac[i] + ac[i+1]
            offset1 = 0.5*(ac[i-1] - ac[i+1])/denom1 if denom1 != 0 else 0.0
            period = (i + offset1) * dt
            i_peak1 = i
            break
    
    # damping alpha from 2nd ac peak
    alpha = np.nan
    if not np.isnan(period):
        # find 2nd peak in window around 1st peak + period
        i_peak2 = int(2 * period / dt)
        window_radius = int(0.5 * period / dt) # half a period before and after
        window_start = max(2, i_peak2 - window_radius)
        window_end = min(len(ac)-1, i_peak2 + window_radius)
        if window_end - window_start > 3:
            # get largest value and ensure it is a local maximum
            i_peak2 = window_start + np.argmax(ac[window_start:window_end])
            if ac[i_peak2] > ac[i_peak2-1] and ac[i_peak2] > ac[i_peak2+1]:
                # parabolic interpolation for 2nd peak location
                denom2 = ac[i_peak2-1] - 2*ac[i_peak2] + ac[i_peak2+1]
                offset2 = 0.5*(ac[i_peak2-1] - ac[i_peak2+1])/denom2 if denom2 != 0 else 0.0
                # parabolic interpolation for peak heights: ac[i] - 0.25*(ac[i-1] - ac[i+1]) * offset
                h1 = ac[i_peak1] - 0.25 * (ac[i_peak1-1] - ac[i_peak1+1]) * offset1 if denom1 != 0 else ac[i_peak1]
                h2 = ac[i_peak2] - 0.25 * (ac[i_peak2-1] - ac[i_peak2+1]) * offset2 if denom2 != 0 else ac[i_peak2]
                if h1 > 0 and h2 > 0:
                    # h1 = exp(-at), h2 = exp(-a(t+T)) => a = -log(h2/h1)/T
                    alpha = -np.log(h2/h1) / period
                    return period, alpha

    return period, alpha


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
                pole = _dominant_pole(float(tw), float(tb), eps_w, k, n_W, n_B)
                if not np.isnan(pole):
                    analytical_period[i, j] = 2*np.pi / abs(pole.imag)
                    analytical_damping[i, j] = -pole.real

        # simulation grid
        rt_grid = np.array(_compute_rt_grid(model, base_params, taus_W, taus_B, t1=t1))
        tt = np.linspace(0.0, t1, rt_grid.shape[-1])
        simulation_period = np.full((N, N), np.nan)
        simulation_damping = np.full((N, N), np.nan)
        for i in range(N):
            for j in range(N):
                simulation_period[i, j], simulation_damping[i, j] = _period_and_damping(tt, rt_grid[i, j], smoothing_days = max(20, 2*analytical_period[i,j]))

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
        plt.colorbar(sc, ax=ax, label=r'$\tau_W + \tau_B$ (days)')
        plt.tight_layout()
        plt.savefig(output.period, dpi=image_resolution); plt.close(fig)

        # damping scatterplot
        valid = np.isfinite(simulation_damping) & np.isfinite(analytical_damping)
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(analytical_damping[valid], simulation_damping[valid], c=total_delay[valid], cmap='viridis', s=10, alpha=0.8)
        if valid.any():
            lim = [min(analytical_damping[valid].min(), simulation_damping[valid].min()), 1.05*max(analytical_damping[valid].max(), simulation_damping[valid].max())]
            ax.plot(lim, lim, 'k--', lw=1)
            ax.axhline(0, color='k', lw=0.5, ls='--'); ax.axvline(0, color='k', lw=0.5, ls='--')
            ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_xlabel('analytical')
        ax.set_ylabel('simulated')
        ax.set_title(f'Decay rates ({wildcards.pathogen}, $k={k:g}$)')
        plt.colorbar(sc, ax=ax, label=r'$\tau_W + \tau_B$ (days)')
        plt.tight_layout()
        plt.savefig(output.damping, dpi=image_resolution); plt.close(fig)


rule plot_stochastic_baseline_trajectories:
    output:
        plot ="{outdir}/gillespie/stochastic_baseline_trajectories_{pathogen}_N{N}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        t1 = 1000.0
        num_simulations = 100
        extinction_times = []
        ps = Params.for_SEIPAR()
        E0 = 1/N

        fig, (ax_hist, ax_traj) = plt.subplots(nrows=2, ncols=1, figsize=(6,6), sharex=True, height_ratios=[1,2])

        extinction_times = []
        for _ in range(num_simulations):
            tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
            S_traj = yy.T[0]
            ax_traj.plot(tt, S_traj, alpha=0.05, color=colors[wildcards.pathogen])
            extinction_times.append(tt[-1])

        ymax = max(extinction_times)
        tt_det, yy_det = models[wildcards.pathogen](params=ps, t1=t1, E0=E0)
        S_det = yy_det.T[0] * N
        ax_traj.plot(tt_det, S_det, color=colors[wildcards.pathogen])
        ax_traj.set_xlim([0, ymax])

        ax_hist.hist(extinction_times, density=True, color=colors[wildcards.pathogen], bins=int(ymax//10))
        
        ax_hist.set_title('Extinction times')
        ax_traj.set_title('Number of susceptibles')
        ax_traj.set_xlabel('days')
        fig.suptitle(f'Stochastic susceptible trajectories ({wildcards.pathogen}, N={N})')
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)

def calculate_mt_branching_q(ps, ew, es):
    def extinction_prob(q):
        asyx = ps.phi * ps.beta * ps.mu_a_inv * (1-ew/2)
        presyx = ps.beta * ps.sigma_inv * (1-ew/2)
        syx = ps.beta * ps.mu_s_inv * (1-es) * (1-ew/2)
        return ps.p / (1 + asyx * (1-q)) + (1-ps.p) / ((1 + presyx * (1-q)) * (1 + syx * (1-q))) - q
    ext_prob = 1.0
    try: ext_prob = brentq(extinction_prob, 0.0, 1.0-1e-9)
    except: pass
    return ext_prob

rule plot_linearised_branching_process_extinction_probabilities:
    output:
        plot="{outdir}/gillespie/linearised_branching_process_extinction_probabilities_{pathogen}.png",
        I_establishment="{outdir}/gillespie/I_establishment_{pathogen}.png",

    run:
        for path in output: os.makedirs(os.path.dirname(path), exist_ok=True)

        ps = parameters[wildcards.pathogen]
        eps_ww = np.linspace(0.0, 0.999, 100)
        eps_ss = np.linspace(0.0, 0.999, 100)
        qs = np.zeros((len(eps_ww), len(eps_ss)))
        alpha = 0.01
        Iest = np.zeros((len(eps_ww), len(eps_ss)))

        for i, ew in enumerate(eps_ww):
            for j, es in enumerate(eps_ss):
                q = calculate_mt_branching_q(ps, ew, es)
                qs[j,i] = q
                Iest[j,i] = np.ceil(np.log(alpha)/np.log(q))

        deterministic_Rt_grid = compute_R_grid(models[wildcards.pathogen], parameters[wildcards.pathogen]._replace(k=1), eps_ww, eps_ss, Rt_times[wildcards.pathogen])
        fig, ax = plot_heatmap(eps_ww, eps_ss, qs, cmap='magma_r', 
            contour_metric=deterministic_Rt_grid, contour_levels=[1.0], contour_colors='white',
            xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$',
            title='Linearised branching process extinction probabilities')
        plt.savefig(output.plot, dpi=image_resolution); plt.close()

        fig, ax = plot_heatmap(eps_ww, eps_ss, Iest, cmap='plasma', 
            norm=mpl.colors.LogNorm(vmin=np.max([float(np.nanmin(Iest)),1]), vmax=np.max([float(np.nanmax(Iest)),1])),
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
                Iest = np.ceil(np.log(alpha)/np.log(calculate_mt_branching_q(ps, eps_w, eps_s)))
                
                for k in range(num_simulations):
                    tt, yy = gillespie_SEIPAR_W(params=ps, N=N, t1=t1)
                    if (wildcards.scenario == 'establishment') & (np.max(yy[:,2] + yy[:,3] + yy[:,4]) < Iest):
                        continue

                    Rt, time_to_below, Itot, peak_Is, extinction_time, _, _, _ = outcome_metrics(tt, yy, Params.for_SEIPAR(epsilon_s=es, epsilon_w=ew), t1)
                    Rt_list.append(Rt)
                    time_to_below_list.append(time_to_below)
                    Itot_list.append(Itot / N)
                    peak_Is_list.append(peak_Is / N)
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
        plot="{outdir}/gillespie/{scenario}_pathogen{pathogen}_N{N}_sims{num_simulations}_res{resolution}_outcome{metric}.png"
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = int(wildcards.N)
        res = int(wildcards.resolution)
        metric = wildcards.metric
        eps_ww = np.linspace(0.0, 0.999, res)
        eps_ss = np.linspace(0.0, 0.999, res)

        data = np.load(input.npz)[f"{metric}_grid"]
        if metric[-1] == 'r': data /= np.load(input.npz)[f"{metric[:-4]}_grid"]
        if metric.startswith("Rt"): data /= N
        cmap = 'magma' if metric[0] == 'e' else 'RdBu_r' if metric == 'Rt' else 'viridis'
        norm = mpl.colors.CenteredNorm(vcenter=1.0) if metric == 'Rt' else None
        contour_colors = 'black' if metric == 'Rt' else 'white'
        
        fig, ax = plot_heatmap(
            eps_ww, eps_ss, data.T, 
            cmap=cmap, norm=norm, contour_levels=[1.0], contour_colors=contour_colors,
            contour_metric=compute_R_grid(models[wildcards.pathogen], parameters[wildcards.pathogen]._replace(k=1), eps_ww, eps_ss, Rt_times[wildcards.pathogen], E0=1/N), 
            xlabel='Warning response efficacy $\\varepsilon_w$', 
            ylabel='Isolation efficacy $\\varepsilon_s$',
            title={
                "Rt": "Average Final $R_t$", "Rt_var": "CV of Final $R_t$",
                "time_to_below": "Average Time to $R_t < 1$", "time_to_below_var": "CV of Time to $R_t < 1$",
                "Itot": "Average Proportion Infected", "Itot_var": "CV of Proportion Infected",
                "peak_Is": "Average Peak Symptomatic Proportion", "peak_Is_var": "CV of Peak Symptomatic Proportion",
                "extinction_time": "95th Percentile Extinction Time", "extinction_time_var": "CV of Extinction Time"
                }.get(metric, metric)
        )
        fig.savefig(output.plot, dpi=image_resolution, bbox_inches='tight'); plt.close(fig)


rule plot_stochastic_cumulative_extinction_probability:
    output:
        plot ="{outdir}/gillespie/cumulative_extinction_probability_{pathogen}_N{N}_epsS_{eps_s}_epsW{eps_w}_scenario_{scenario}.png",
    run:
        os.makedirs(os.path.dirname(output.plot), exist_ok=True)
        N = float(wildcards.N)
        eps_s = float(wildcards.eps_s)
        eps_w = float(wildcards.eps_w)
        t1 = 2000.0
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
        plt.title(f'Cumulative extinction probability ({wildcards.pathogen}, $\\varepsilon_s={eps_s}$, $\\varepsilon_w={eps_w}$, {wildcards.scenario})', fontsize=14)
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(-50, max(extinction_times))
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1+lines_2, labels_1+labels_2, loc='best')
        ax1.grid(True, alpha=0.5)
        fig.tight_layout()
        plt.savefig(output.plot, dpi=image_resolution); plt.close()


rule all:
    input:
        expand(
            rules.plot_stochastic_cumulative_extinction_probability.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], eps_s=[0.4], eps_w=[0.4, 0.8], scenario=['establishment', 'all'],
        ),
        expand(
            rules.plot_stochastic_intervention_grid.output.plot, outdir=outdir,
            pathogen=["SARS-CoV-2"], N=[10000], num_simulations=[1000], resolution=[4], scenario=['establishment', 'all'],
            metric=["Rt", "Rt_var", "time_to_below", "time_to_below_var", "Itot", "Itot_var", "peak_Is", "peak_Is_var", "extinction_time", "extinction_time_var"],
        ),
        expand(rules.plot_linearised_branching_process_extinction_probabilities.output, outdir=outdir, pathogen=["SARS-CoV-2"]),
        expand(rules.plot_efficacy_grid_Rt_final.output.plot, pathogen=pathogens, outdir=outdir),
        expand(rules.plot_efficacy_grid_Itot_final.output.plot, pathogen=pathogens, outdir=outdir),
        expand(rules.plot_asymptomatic_grid_Rt_final.output.plot, outdir=outdir, pathogen=asymptomatic_pathogens),
        expand(rules.plot_asymptomatic_grid_Itot_final.output.plot, outdir=outdir, pathogen=asymptomatic_pathogens),
        expand(rules.plot_prcc_monotonicity.output.plot, outdir=outdir, pathogen=pathogens, scenario=prcc_scenarios, outcome=prcc_outcomes),
        expand(rules.plot_prcc_grid.output.plot, outdir=outdir),
        expand(rules.plot_combined_sensitivity_grid.output.plot, outdir=outdir),
        expand(rules.plot_stochastic_baseline_trajectories.output, outdir=outdir, pathogen=["SARS-CoV-2"], #pathogens,
            N=[100, 50_000, 500_000] #,100000], # N=gillespie_popsizes,
        ),
        expand(rules.plot_trajectory.output.plot, outdir=outdir, pathogen=pathogens,
            epsilon_s=[0.0, 0.4, 0.8], epsilon_w=[0.0, 0.4, 0.8],
        ),
        expand(rules.plot_trajectory_delayed_ww_intervention.output.plot, outdir=outdir, pathogen=asymptomatic_pathogens,
            epsilon_s=[0.8], epsilon_w=[0.8], I_crit=[1e-5, 1e-4, 1e-3, 1e-2],
        ),
        expand(rules.delayed_ww_intervention.output.plot, 
            outdir=outdir, pathogen=["SARS-CoV-2"], # TODO: change main rule for generalisation
        ),
        expand(rules.baseline_trajectories.output.plot, pathogen=pathogens, outdir=outdir),
        # expand(rules.baseline_trajectories_no_asymptomatic.output.plot, pathogen=pathogens, outdir=outdir),
        expand(rules.plot_true_vs_reported_Rt_scenarios.output, pathogen=pathogens, outdir=outdir, k=[1, 3, 10, 30],),
        expand(rules.plot_true_vs_reported_Rt_scenarios_vary_k.output, pathogen=pathogens, outdir=outdir, tau_W=[14], tau_B=[7]),
        expand(rules.plot_true_vs_reported_Rt_heatmaps.output, pathogen=["SARS-CoV-2"], outdir=outdir, k=[10]), #k=[1, 3, 10, 30],),
        expand(rules.plot_response_function.output.plot, outdir=outdir),
        expand(rules.plot_main_intervention_grid.output.plot, outdir=outdir),
        expand(rules.plot_R_1_contours.output.plot, outdir=outdir),
        expand(rules.plot_combined_contour_grid_R1_Itot.output.plot, outdir=outdir),
        expand(rules.export_param_bounds.output.tex, outdir=outdir),
        expand(rules.plot_controllability_boundaries.output.plot, outdir=outdir),
        expand(rules.plot_asymptomatic_generation_time.output.plot, outdir=outdir),
        expand(rules.plot_nonlinear_response_analysis.output.plot, outdir=outdir),
        expand(rules.plot_asymptomatic_landscape.output.plot, outdir=outdir),
        expand(rules.calculate_generation_times.output.txt, outdir=outdir),
        expand(rules.plot_gain_margins.output.plot, outdir=outdir),
        expand(rules.plot_delay_margins.output.plot, outdir=outdir),
        expand(rules.plot_period_and_damping_scatter.output, outdir=outdir, pathogen=pathogens, k=[10, 30, 60, 80]),
