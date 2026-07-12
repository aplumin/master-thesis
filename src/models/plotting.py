"""
Plotting functions.
"""

import jax.numpy as jnp
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import gamma
from scipy.integrate import trapezoid

from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.parameters import Params
from models.compartmental import simulate_SEIPAR_W
from models.metrics import (
    outcome_metrics, infectious_fractions, transmission_fractions,
    compute_I_tot_grid, compute_R_grid, 
    compute_asymptomatic_grid_Rt, compute_asymptomatic_grid_Itot, compute_I_tot_grid_delayed_ww,
)


def plot_heatmap(
    X, Y, Z, 
    cmap='viridis', shading='auto', norm=None,
    contour_metric = None, contour_levels=[], contour_colors='black', contour_linestyles=['-'], contour_alpha=1.0,
    title=None, title_fontsize=18, title_pad=10,
    xlabel=None, ylabel=None,
    xlabelsize=14, ylabelsize=14,
    x_logscale=False, y_logscale=False, 
    cbar_shrink=0.8, cbar_aspect=30, cbar_label=None, cbar_labelsize=14, cbar_labelpad=10,
    cbar_axhlines=[], cbar_axhlines_colors=[], cbar_axhlines_linestyles=[], cbar_ticks=[],
):
    """General heatmap plotting function."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_box_aspect(1)

    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, shading=shading, norm=norm)
    
    if contour_metric is None: contour_metric = Z
    if contour_levels:
        ax.contour(X, Y, contour_metric, levels=contour_levels, colors=contour_colors, linestyles=contour_linestyles, alpha=contour_alpha)
    
    if len(cbar_ticks) > 0:
        cbar = fig.colorbar(mesh, ax=ax, shrink=cbar_shrink, aspect=cbar_aspect, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks)
    else:
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
        title=title, xlabel='Behavioural delay $\\tau_B$', ylabel='Infection threshold',
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
    Assume compartment order: S, E, [I compartments], R, [W delay compartments], [B delay compartments].
    """
    # run the model
    tt, yy = model(params=params, t1=t1)
    compartments = yy.T
    R_idx = -(params.n_W + params.n_B + 1)
    I_compartments = compartments[slice(2, R_idx) if R_idx != -1 else slice(2, None)]
    total_I = np.sum(I_compartments, axis=0)
    Is = I_compartments[-1]
    
    # plot
    fig, (ax_main, ax_rt, ax_trans) = plt.subplots(nrows=3, ncols=1, figsize=(6, 14), gridspec_kw={'height_ratios': [6,2,6]})

    # trajectories
    if plot_S: ax_main.plot(tt, compartments[0], label='$S$', color='green')
    if plot_E: ax_main.plot(tt, compartments[1], label='$E$', color='orange')
    if plot_Is and len(I_compartments) > 0: ax_main.plot(tt, Is, label='$I_s$', color='blue')
    if plot_Ia and len(I_compartments) > 1: ax_main.plot(tt, I_compartments[0], label='$I_a$', color='purple')
    if plot_Ip and len(I_compartments) > 2: ax_main.plot(tt, I_compartments[1], label='$I_p$', color='skyblue')
    if plot_total_I: ax_main.plot(tt, total_I, label='$I_{total}$', color='red', linestyle='--')
    if plot_R: ax_main.plot(tt, compartments[R_idx], label='$R$', color='black')

    plt.suptitle(title, fontsize=20)
    ax_rt.set_xlabel("Time (days)", fontsize=16)
    ax_main.set_ylabel("Population", fontsize=16)
    if semilogy: ax_main.set_yscale('log')

    # Is peak size, time to peak, and total wave time
    idx_peak = np.argmax(Is)
    peak_Is = Is[idx_peak]
    t_peak = tt[idx_peak]
    # first time Is crossed threshold
    indices_above = np.where(Is > params.I_crit)[0]
    t1_crit = tt[indices_above[0]] if indices_above.size > 0 else tt[-1]
    # first time after Is is below threshold
    t2_crit = tt[indices_above[-1] + 1] if indices_above.size > 0 else tt[-1]
    time_to_peak = t_peak - t1_crit
    total_time = t2_crit - t1_crit

    # final size
    def calculate_final_size(R0):
        def final_size_equation(Z): 
            return 1-Z-np.exp(-R0*Z)
        return fsolve(final_size_equation, x0=0.5)[0]
    final_size = calculate_final_size(params.R_0)
    ax_main.axhline(final_size, label=r'$I_\text{tot}=$'+f'{final_size:.2f}', color='grey', linestyle='--')
    ax_main.legend(loc='upper right')

    # Rt
    rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
    ax_rt.plot(tt, rt_true, color='black', label='$\mathcal{R}_t$')
    ax_rt.axhline(params.R_crit, color='grey', linestyle='--')
    s_contribution = (1-params.epsilon_s) * (1-params.p) * params.beta * params.mu_s_inv
    a_contribution = params.p * params.phi * params.beta * params.mu_a_inv
    p_contribution = (1-params.p) * params.beta * params.sigma_inv
    total_contributions = s_contribution + a_contribution + p_contribution
    rt_s = rt_true * s_contribution / total_contributions
    rt_a = rt_true * a_contribution / total_contributions
    rt_p = rt_true * p_contribution / total_contributions
    params_baseline = params.update(epsilon_s=0.0, epsilon_w=0.0)
    _, yy0 = model(params=params_baseline, t1=t1)
    ax_rt.fill_between(tt, 0, params_baseline.R_0 * params_baseline.rho * yy0[:,-1] * yy0[:,0], color='grey', alpha=0.2)
    if rt_a.any() > 0: ax_rt.fill_between(tt, 0, rt_a, color='purple', alpha=0.5, label=r'$\mathcal{R}_a$')
    if rt_p.any() > 0: ax_rt.fill_between(tt, rt_a, rt_a + rt_p, color='skyblue', alpha=0.5, label=r'$\mathcal{R}_p$')
    ax_rt.fill_between(tt, rt_a + rt_p, rt_a + rt_p + rt_s, color='blue', alpha=0.5, label=r'$\mathcal{R}_s$')
    ax_rt.legend()

    # infections vs transmissions
    type_order  = ["a", "p", "s"]
    type_color  = {"a": "purple", "p": "skyblue", "s": "blue"}
    type_I_tex  = {"a": r"$I_a$", "p": r"$I_p$", "s": r"$I_s$"}
    type_R_tex  = {"a": r"$\mathcal{R}_a$", "p": r"$\mathcal{R}_p$", "s": r"$\mathcal{R}_s$"}

    cols = [infectious_fractions(params)[j] for j in type_order]
    rows = [transmission_fractions(params)[i] for i in type_order]
    xb = np.concatenate([[0.0], np.cumsum(cols)])
    yb = np.concatenate([[0.0], np.cumsum(rows)])
    for row, i in enumerate(type_order):
        for col, j in enumerate(type_order):
            w, h = cols[col], rows[row]
            if w <= 0 or h <= 0: continue
            x0, y0 = xb[col], 1.0 - yb[row+1]
            ax_trans.add_patch(mpl.patches.Rectangle((x0, y0), w, h, facecolor=type_color[type_order[row]], alpha=1-0.25*(2-col), edgecolor="white"))
            ax_trans.text(x0+w/2, y0+h/2, f"{100*h*w:.0f}%", ha="center", va="center", fontsize=10, zorder=3, color="white")
            ax_trans.text(xb[col]+cols[col]/2, 1.04, type_R_tex[j]+f"\n{100*cols[col]:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=type_color[j])
            ax_trans.text(-0.04, 1.0-(yb[row]+yb[row+1])/2, type_I_tex[i]+f"\n{100*rows[row]:.0f}%", ha="right", va="center", fontsize=11, fontweight="bold", color=type_color[i])
        ax_trans.set_xlim(-0.12, 1.0)
        ax_trans.set_ylim(0.0, 1.12)
        ax_trans.set_aspect("equal")
        ax_trans.axis("off")

    # save and close
    plt.tight_layout()
    fig.savefig(path, dpi=image_resolution)
    plt.close(fig)
    return peak_Is, time_to_peak, total_time



def plot_asymptomatic_effect_for_range_of_intervention_efficacies(
    model: Callable = simulate_SEIPAR_W, 
    params: Params = Params.for_SEIPAR(),
    total_infected: bool = False,
    ps = jnp.linspace(0.0, 0.999, 100),
    phis = jnp.linspace(0.0, 0.999, 100),
    p_CI = (None, None),
    phi_CI = (None, None),
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
            base_params = params.update(epsilon_s=float(eps_s), epsilon_w=float(eps_w))
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
        # mean and CI cross
        has_p_ci = p_CI[0] is not None and p_CI[1] is not None
        has_phi_ci = phi_CI[0] is not None and phi_CI[1] is not None
        if has_p_ci or has_phi_ci:
            xerr = np.array([[params.p - p_CI[0]], [p_CI[1] - params.p]]) if has_p_ci else None
            yerr = np.array([[params.phi - phi_CI[0]], [phi_CI[1] - params.phi]]) if has_phi_ci else None
            ax.errorbar(params.p, params.phi, xerr=xerr, yerr=yerr, fmt='o', color='white', markeredgecolor='black', ecolor='white', elinewidth=1.5, capsize=3, markersize=5)
        else:
            ax.plot(params.p, params.phi, marker='o', color='white', markeredgecolor='black', markersize=5)
        ax.set_ylim([0.0,1.0])
        return mesh
    g.map_dataframe(_meshmap, **plot_kwargs)

    # labels and title
    g.set_titles(row_template=r"$\varepsilon_s = {row_name}$", col_template=r"$\varepsilon_w = {col_name}$")
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

def plot_extinction_probability_scenario(ax, times, title_label, tt_det, S_det):
    if len(times) == 0: return
    
    sorted_times = np.sort(times)
    n_events = sorted_times.shape[0]
    cumulative_prob = np.arange(1, n_events + 1) / n_events
    
    z_score = 1.96
    std_error = np.sqrt(cumulative_prob * (1-cumulative_prob) / n_events)
    ci_lower = np.maximum(0, cumulative_prob - z_score*std_error)
    ci_upper = np.minimum(1, cumulative_prob + z_score*std_error)
    ax.step(sorted_times, cumulative_prob, where='post', label='Cumulative extinction probability', color='blue', linewidth=2)
    ax.fill_between(sorted_times, ci_lower, ci_upper, step='post', color='blue', alpha=0.25)
    
    # median time
    median_time = np.median(times)
    idx_med_upper = np.argmax(ci_upper >= 0.5)
    idx_med_lower = np.argmax(ci_lower >= 0.5)
    if idx_med_upper < n_events and ci_upper[-1] >= 0.5:
        median_time_ci_lower = sorted_times[idx_med_upper]
        median_time_ci_upper = sorted_times[idx_med_lower] if ci_lower[-1] >= 0.5 else sorted_times[-1]
        ax.axvline(median_time, color='red', label=f'Median: {median_time:.2f} [{median_time_ci_lower:.2f}, {median_time_ci_upper:.2f}]')
        ax.axvspan(median_time_ci_lower, median_time_ci_upper, color='red', alpha=0.2)
    else:
        ax.axvline(median_time, color='red', label=f'Median: {median_time:.2f}')
    
    # 95% time
    time_95 = np.percentile(times, 95)
    idx_95_upper = np.argmax(ci_upper >= 0.95)
    idx_95_lower = np.argmax(ci_lower >= 0.95)
    if idx_95_upper < n_events and ci_upper[-1] >= 0.95:
        time_95_ci_lower = sorted_times[idx_95_upper]
        time_95_ci_upper = sorted_times[idx_95_lower] if ci_lower[-1] >= 0.95 else sorted_times[-1]
        ax.axvline(time_95, color='orange', label=f'95%: {time_95:.2f} [{time_95_ci_lower:.2f}, {time_95_ci_upper:.2f}]')
        ax.axvspan(time_95_ci_lower, time_95_ci_upper, color='orange', alpha=0.2)
    else:
        ax.axvline(time_95, color='orange', label=f'95%: {time_95:.2f}')
    
    # deterministic susceptible trajectory
    ax.plot(tt_det, S_det, color='green', label='Deterministic susceptible trajectory')
    
    # histogram
    ax_hist = ax.twinx()
    ax_hist.hist(times, bins=100, density=True, color='gray', alpha=0.3, label='Extinction times histogram')
    ax_hist.set_ylabel('Density', color='gray', fontsize=11)
    ax_hist.tick_params(axis='y', labelcolor='gray')
    
    # styling
    ax.set_title(title_label, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.5)
    
    # legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_hist.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=9)

def plot_nonlinear_response_analysis(dt, n_W, tau_W, n_B, tau_B, k, threshold, eps_w, path, res, pathogens, colors, parameters, best_params_kwargs, worst_params_kwargs):
    t = np.arange(0, 50, dt)
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
    axes[1, 0].set_ylim(-0.2, 1.2)
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
    axes[1, 1].set_ylim(-0.2, 1.2)
    # axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title('Effective Transmission Modification')
    axes[2, 1].set_xlabel('Days')
    axes[2, 1].set_ylim(-0.2, 1.2)
    # axes[2, 1].legend(loc='upper right')
    axes[2, 1].grid(True, alpha=0.3)
    # fig.suptitle("Wastewater Warning Response Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(path, dpi=res); plt.close(fig)

def table_scenario_label(name, eps_s, eps_w, bold=False):
    parts = []
    if eps_s > 0: 
        parts.append("\\mathbf{\\varepsilon_s="+f"{eps_s:g}"+"}") if bold else parts.append(f"\\varepsilon_s={eps_s:g}")
    if eps_w > 0: 
        parts.append("\\mathbf{\\varepsilon_w="+f"{eps_w:g}"+"}") if bold else parts.append(f"\\varepsilon_w={eps_w:g}")
    return name if not parts else f"{name} (${', '.join(parts)}$)"

def f_days(m): 
    return "---" if np.isnan(m["time_to_peak"]) else f"{m['time_to_peak']:.1f}", "---" if np.isnan(m["wave_time"]) else f"{m['wave_time']:.1f}"

def f_pct(x, dp=0):
    return "---" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{100*x:.{dp}f}\\%"

def table_row_metrics(ps, model, eps_s, eps_w, t1, E0=1e-6, itot_baseline=None, icrit=1e-4, strategy="baseline"):
    ps = ps.update(epsilon_s=eps_s, epsilon_w=eps_w)
    if strategy=="asymmetric":
        tt, yy, *_ = model(params=ps, t1=t1, E0=E0, asymmetric=True)
    elif strategy=="interval":
        tt, yy, *_ = model(params=ps, t1=t1, E0=E0, discrete_eval=True)
    else:
        tt, yy, *_ = model(params=ps, t1=t1, E0=E0)
    Rt_final = float(outcome_metrics(tt, yy, ps, t1)[0])
    S = yy[:, 0]
    B_out = yy[:, -1]
    Is = yy[:, -(ps.n_W + ps.n_B + 2)]
    itot = float(S[0] - S[-1])
    prevented = np.nan if itot_baseline in (None, 0.0) else 1.0 - itot / itot_baseline
    peak_idx = int(np.argmax(Is))
    peak_Is = float(Is[peak_idx])
    above_idx = np.where(Is > icrit)[0]
    if above_idx.size == 0: time_to_peak = wave_time = np.nan
    else:
        t_start = tt[above_idx[0]]
        last = int(above_idx[-1])
        if last + 1 < len(tt): t_end = tt[last + 1]
        else: t_end = tt[-1]
        time_to_peak = float(tt[peak_idx] - t_start)
        wave_time = float(t_end - t_start)
    warning_cost = float(trapezoid(1.0 - B_out, tt))
    isolation_cost = float(trapezoid(eps_s * Is, tt))
    return dict(
        Rt=Rt_final, peak_Is=peak_Is, time_to_peak=time_to_peak, wave_time=wave_time, itot=itot, 
        prevented=prevented, warning_cost=warning_cost, isolation_cost=isolation_cost, cost=warning_cost + isolation_cost
    )
