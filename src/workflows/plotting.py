"""
Plotting functions.
"""

from collections.abc import Callable

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import fsolve
from scipy.stats import gamma

from models.compartmental import simulate_SEIPAR_W
from models.metrics import (
    infectious_fractions,
    transmission_fractions,
)
from models.parameters import Params
from workflows.jax_grids import (
    compute_asymptomatic_grid_Itot,
    compute_asymptomatic_grid_Rt,
    compute_I_tot_grid,
    compute_I_tot_grid_delayed_ww,
    compute_R_grid,
)

plt.rcParams['figure.constrained_layout.use'] = True

def plot_heatmap(
    X, Y, Z, 
    cmap='viridis', shading='auto', norm=None,
    contour_metric=None, contour_levels=None, contour_colors='black', contour_linestyles=('-',), contour_alpha=1.0,
    title=None, title_fontsize=14, title_pad=None, figsize=(6, 6),
    xlabel=None, ylabel=None, xlabelsize=12, ylabelsize=12,
    x_logscale=False, y_logscale=False, 
    cbar_shrink=0.8, cbar_aspect=30, cbar_label=None, cbar_labelsize=12, cbar_labelpad=10,
    cbar_axhlines=(), cbar_axhlines_colors=(), cbar_axhlines_linestyles=(), cbar_ticks=(),
):
    """General heatmap plotting function."""
    contour_levels = contour_levels or []
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(1)
    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, shading=shading, norm=norm)
    if contour_metric is None: contour_metric = Z
    if contour_levels: ax.contour(X, Y, contour_metric, levels=contour_levels, colors=contour_colors, linestyles=list(contour_linestyles), alpha=contour_alpha)
    if len(cbar_ticks) > 0:
        cbar = fig.colorbar(mesh, ax=ax, shrink=cbar_shrink, aspect=cbar_aspect, ticks=list(cbar_ticks))
        cbar.ax.set_yticklabels(cbar_ticks)
    else:
        cbar = fig.colorbar(mesh, ax=ax, shrink=cbar_shrink, aspect=cbar_aspect)
    cbar.set_label(cbar_label, fontsize=cbar_labelsize, labelpad=cbar_labelpad)
    for hline, color, ls in zip(cbar_axhlines, cbar_axhlines_colors, cbar_axhlines_linestyles, strict=True):
        cbar.ax.axhline(hline, color=color, linestyle=ls)
    if x_logscale: ax.set_xscale('log')
    if y_logscale: ax.set_yscale('log')
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel(xlabel, fontsize=xlabelsize)
    ax.set_ylabel(ylabel, fontsize=ylabelsize)
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
        title=title, xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$')
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
        cmap='RdBu_r', norm=mpl.colors.CenteredNorm(vcenter=1.0), contour_levels=[1.0],
        title=title, xlabel='Warning response efficacy $\\varepsilon_w$', ylabel='Isolation efficacy $\\varepsilon_s$')
    return fig

def plot_I_tot_delayed_ww(model=simulate_SEIPAR_W, parameters=Params.for_SEIPAR(), title=None, t1=600.0, E0=1e-6):
    taus = jnp.linspace(1.0, 30.0, 100)
    I_crit_list = jnp.logspace(-6, 0, 100)
    fig, _ = plot_heatmap(
        taus, I_crit_list, compute_I_tot_grid_delayed_ww(model=model, base_params=parameters, taus=taus, I_crit_list=I_crit_list, t1=t1, E0=E0),
        title=title, xlabel='Behavioural delay $\\tau_B$', ylabel='Infection threshold', cbar_shrink=0.7,
        y_logscale=True, cbar_label='Total infections (relative to baseline)', figsize=(6,6))
    return fig


def plot_trajectory(
    model: Callable = simulate_SEIPAR_W, params: Params = Params.for_SEIPAR(), 
    path = "trajectory.png", title = "Trajectory", t1 = 600.0, icrit = None, image_resolution = 900,
    plot_S = True, plot_E = True, plot_Ia = True, plot_Ip = True, plot_Is = True, plot_total_I = False,
    plot_R = True, semilogy = False, no_decomp = False, model_type = "exponential"):
    """
    Simulate and plot trajectories.
    Assume compartment order: S, E, [I compartments], R, [W delay compartments], [B delay compartments].
    """
    # run the model
    tt, yy = model(params=params, t1=t1, n_ts=int(t1))
    compartments = yy.T
    R_idx = -(params.n_W + params.n_B + 1)
    iI = 1 + int(getattr(params, "nE", 1))
    I_compartments = compartments[slice(iI, R_idx) if R_idx != -1 else slice(iI, None)]
    total_I = np.sum(I_compartments, axis=0)
    E = np.sum(compartments[1:iI], axis=0)
    if model_type == "Erlang":
        if len(I_compartments) > params.nS:
            Ia = np.sum(I_compartments[:params.nA], axis=0)
            Ip = np.sum(I_compartments[params.nA:params.nA+params.nP], axis=0)
            Is = np.sum(I_compartments[params.nA+params.nP:], axis=0)
        else:
            Ia, Ip = None, None
            Is =  np.sum(I_compartments, axis=0)
    else: # exponential
        Is = I_compartments[-1] if len(I_compartments) > 0 else None
        Ia = I_compartments[0] if len(I_compartments) > 1 else None
        Ip = I_compartments[1] if len(I_compartments) > 2 else None
    # end time
    end_time = _wave_end_time(tt, total_I, icrit)
    if end_time <= 0 or not np.isfinite(end_time):
        end_time = tt[-1]
    
    # plot
    if no_decomp:
        fig, (ax_main, ax_rt) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), gridspec_kw={'height_ratios': [6,2]})
    else:
        fig, (ax_main, ax_rt, ax_trans) = plt.subplots(nrows=3, ncols=1, figsize=(6, 14), gridspec_kw={'height_ratios': [6,2,6]})

    # trajectories
    if plot_S: ax_main.plot(tt, compartments[0], label='$S$', color='green')
    if plot_Is and Is is not None: ax_main.plot(tt, Is, label='$I_s$', color='blue', linestyle='-')
    if plot_Ia and Ia is not None: ax_main.plot(tt, Ia, label='$I_a$', color='purple', linestyle='--')
    if plot_Ip and Ip is not None: ax_main.plot(tt, Ip, label='$I_p$', color='skyblue', linestyle='-.')
    if plot_E: ax_main.plot(tt, E, label='$E$', color='orange', linestyle=':')
    if plot_total_I: ax_main.plot(tt, total_I, label='$I_{total}$', color='red', linestyle='-')
    if plot_R: ax_main.plot(tt, compartments[R_idx], label='$R$', color='black')

    # final size
    if not no_decomp:
        def calculate_final_size(R0):
            """1 - Z = exp(-R0*Z)."""
            def _fs(Z): 
                return 1-Z-np.exp(-R0*Z)
            return fsolve(_fs, x0=0.5)[0]
        final_size = calculate_final_size(params.R_0)
        ax_main.axhline(final_size, label=r'$I_\text{tot}=$'+f'{final_size:.2f}', color='grey', linestyle='--')
    
    # trajectory styling
    plt.suptitle(title, fontsize=20)
    ax_rt.set_xlabel("Time (days)", fontsize=16)
    ax_main.set_ylabel("Population", fontsize=16)
    if semilogy: ax_main.set_yscale('log')
    ax_main.legend(loc='upper right', ncol=2, fontsize=12)
    ax_main.set_xlim(0, end_time)

    # Rt
    rt_true = params.R_0 * params.rho * yy[:,-1] * yy[:,0]
    ax_rt.plot(tt, rt_true, color='black', label=r'$\mathcal{R}_t$')
    ax_rt.axhline(params.R_crit, color='grey', linestyle='--')
    s_contribution = (1-params.epsilon_s) * (1-params.p) * params.beta * params.mu_s_inv
    a_contribution = params.p * params.phi_a * params.beta * params.mu_a_inv
    p_contribution = (1-params.p) * params.phi_p * params.beta * params.sigma_inv
    total_contributions = s_contribution + a_contribution + p_contribution
    rt_s = rt_true * s_contribution / total_contributions
    rt_a = rt_true * a_contribution / total_contributions
    rt_p = rt_true * p_contribution / total_contributions
    # baseline
    params_baseline = params.update(epsilon_s=0.0, epsilon_w=0.0)
    _, yy0 = model(params=params_baseline, t1=t1, n_ts=int(t1))
    rt_baseline = params_baseline.R_0 * params_baseline.rho * yy0[:,-1] * yy0[:,0]
    # styling
    ax_rt.fill_between(tt, 0, rt_baseline, color='grey', alpha=0.2)
    ax_rt.fill_between(tt, 0, rt_s, color='blue', alpha=0.5, label=r'$\mathcal{R}_s$')
    if (rt_p > 0).any(): ax_rt.fill_between(tt, rt_s, rt_s + rt_p, color='skyblue', alpha=0.5, label=r'$\mathcal{R}_p$')
    if (rt_a > 0).any(): ax_rt.fill_between(tt, rt_s + rt_p, rt_s + rt_p + rt_a, color='purple', alpha=0.5, label=r'$\mathcal{R}_a$')
    ax_rt.legend(loc='upper right', fontsize=12)
    ax_rt.set_xlim(0, end_time)

    # infections vs transmissions
    if not no_decomp:
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
                ax_trans.add_patch(mpl.patches.Rectangle((x0, y0), w, h, facecolor=type_color[i], alpha=1-0.25*(2-col), edgecolor="white"))
                ax_trans.text(x0+w/2, y0+h/2, f"{100*h*w:.0f}%", ha="center", va="center", fontsize=14, zorder=3, color="white")
                ax_trans.text(xb[col]+cols[col]/2, 1.04, type_I_tex[j]+f"\n{100*cols[col]:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold", color=type_color[j])
                ax_trans.text(-0.04, 1.0-(yb[row]+yb[row+1])/2, type_R_tex[i]+f"\n{100*rows[row]:.0f}%", ha="right", va="center", fontsize=14, fontweight="bold", color=type_color[i])
            ax_trans.set_xlim(-0.12, 1.0)
            ax_trans.set_ylim(0.0, 1.12)
            ax_trans.set_aspect("equal")
            ax_trans.axis("off")

    # save and close
    fig.savefig(path, dpi=image_resolution)
    plt.close(fig)

    ### metrics ##
    # Is peak size, time to peak, and total wave time
    if Is is None:
        return float("nan"), float("nan"), float("nan")
    idx_peak = np.argmax(Is)
    peak_Is = Is[idx_peak]
    t_peak = tt[idx_peak]
    # first time Is crossed threshold
    indices_above = np.where(Is > params.I_crit)[0]
    t1_crit = tt[indices_above[0]] if indices_above.size > 0 else tt[-1]
    # first time after Is is below threshold
    t2_crit = tt[indices_above[-1] + 1] if indices_above.size > 0 and indices_above[-1] + 1 < len(tt) else tt[-1]
    time_to_peak = t_peak - t1_crit
    total_time = t2_crit - t1_crit
    return peak_Is, time_to_peak, total_time


def plot_asymptomatic_effect_for_range_of_intervention_efficacies(
    model: Callable = simulate_SEIPAR_W, 
    params: Params = Params.for_SEIPAR(),
    total_infected: bool = False,
    ps = jnp.linspace(0.0, 0.999, 100),
    phi_as = jnp.linspace(0.0, 0.999, 100),
    p_CI = (None, None),
    phi_a_CI = (None, None),
    epsilon_s = (0.0, 0.4, 0.8),
    epsilon_w = (0.0, 0.4, 0.8),
    E0: float = 1e-6,
    t1: float | None = None, 
    n_ts: int | None = None,
    image_resolution: int = 900,
    path: str = "asymptomatic_effect.png",
) -> None:
    
    # end time
    if t1 is None: t1 = 600.0 # if total_infected else 50.0

    # build dataframe
    p_grid, phi_a_grid = jnp.meshgrid(ps, phi_as, indexing="xy")
    df_list = []
    for eps_s in epsilon_s:
        for eps_w in epsilon_w:
            base_params = params.update(epsilon_s=float(eps_s), epsilon_w=float(eps_w))
            if total_infected:
                Z = compute_asymptomatic_grid_Itot(model=model, base_params=base_params, p=ps, phi_a=phi_as, t1=t1, E0=E0, n_ts=n_ts)
            else:
                Z = compute_asymptomatic_grid_Rt(model=model, base_params=base_params, p=ps, phi_a=phi_as, t1=t1, E0=E0, n_ts=n_ts)
            df_list.append(pd.DataFrame({'p': np.array(p_grid.flatten()), 'phi_a': np.array(phi_a_grid.flatten()), 'Z': np.array(Z.flatten()), 'eps_s': eps_s, 'eps_w': eps_w}))
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
        Z_matrix = data.pivot(index='phi_a', columns='p', values='Z').values
        mesh = ax.pcolormesh(p_grid, phi_a_grid, Z_matrix, linewidth=0, edgecolors='none', rasterized=True, **kwargs)
        if not total_infected: # R=1 contour
            ax.contour(p_grid, phi_a_grid, Z_matrix, levels=[1.0], colors='black', linewidths=1.5, linestyles='dashed')
        # mean and CI cross
        has_p_ci = p_CI[0] is not None and p_CI[1] is not None
        has_phi_a_ci = phi_a_CI[0] is not None and phi_a_CI[1] is not None
        if has_p_ci or has_phi_a_ci:
            xerr = np.array([[params.p - p_CI[0]], [p_CI[1] - params.p]]) if has_p_ci else None
            yerr = np.array([[params.phi_a - phi_a_CI[0]], [phi_a_CI[1] - params.phi_a]]) if has_phi_a_ci else None
            ax.errorbar(params.p, params.phi_a, xerr=xerr, yerr=yerr, fmt='o', color='white', markeredgecolor='black', ecolor='white', elinewidth=1.5, capsize=3, markersize=5)
        else:
            ax.plot(params.p, params.phi_a, marker='o', color='white', markeredgecolor='black', markersize=5)
        ax.set_ylim([0.0,1.0])
        return mesh
    g.map_dataframe(_meshmap, **plot_kwargs)

    # labels and title
    g.set_titles(row_template=r"$\varepsilon_s = {row_name}$", col_template=r"$\varepsilon_w = {col_name}$")
    g.set(xlabel=None, ylabel=None, aspect='equal')
    g.figure.supxlabel("Proportion asymptomatic", fontsize=14)
    g.figure.supylabel("Relative infectiousness", fontsize=14)
    g.figure.subplots_adjust(left=0.1)

    # colorbar
    mesh = g.axes[-1, -1].collections[0] 
    cbar = g.figure.colorbar(mesh, ax=g.axes.ravel().tolist(), shrink=0.8, aspect=30)
    if not total_infected: cbar.ax.axhline(1.0, color='black', linewidth=1.5)

    # save and close
    plt.savefig(path, dpi=image_resolution)
    plt.close(g.figure)

def plot_extinction_probability_scenario(ax, times, title_label, tt_det, S_det):
    if len(times) == 0:
        return
    
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
        ax.axvline(median_time, color='red', label='Median')
        ax.axvspan(median_time_ci_lower, median_time_ci_upper, color='red', alpha=0.2)
    else:
        ax.axvline(median_time, color='red', label='Median')
    
    # 95% time
    time_95 = np.percentile(times, 95)
    idx_95_upper = np.argmax(ci_upper >= 0.95)
    idx_95_lower = np.argmax(ci_lower >= 0.95)
    if idx_95_upper < n_events and ci_upper[-1] >= 0.95:
        time_95_ci_lower = sorted_times[idx_95_upper]
        time_95_ci_upper = sorted_times[idx_95_lower] if ci_lower[-1] >= 0.95 else sorted_times[-1]
        ax.axvline(time_95, color='orange', label='95%')
        ax.axvspan(time_95_ci_lower, time_95_ci_upper, color='orange', alpha=0.2)
    else:
        ax.axvline(time_95, color='orange', label='95%')
    
    # deterministic susceptible trajectory
    ax.plot(tt_det, S_det, color='green', label='Deterministic susceptible trajectory')
    
    # histogram
    ax_hist = ax.twinx()
    ax_hist.hist(times, bins=100, density=True, color='gray', alpha=0.3, label='Extinction times histogram')
    ax_hist.set_ylabel('Density', color='gray', fontsize=12)
    ax_hist.tick_params(axis='y', labelcolor='gray')
    
    # styling
    ax.set_title(title_label, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.5)
    
    # legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_hist.get_legend_handles_labels()
    return (lines_1 + lines_2, labels_1 + labels_2)

def plot_nonlinear_response_analysis(dt, n_W, tau_W, n_B, tau_B, k, threshold, eps_w, path, pathogens, colors, parameters, R0_lo, R0_hi):
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
        x_l, y_l, z_l = total_response(R0_lo[p])
        x_h, y_h, z_h = total_response(R0_hi[p])
        color = colors[p]
        axes[0, 1].plot(t, x_m, color=color, linewidth=2, label=p)
        axes[0, 1].fill_between(t, x_l, x_h, color=color, alpha=0.2)
        axes[1, 1].plot(t, y_m, color=color, linewidth=2)
        axes[1, 1].fill_between(t, np.minimum(y_l, y_h), np.maximum(y_l, y_h), color=color, alpha=0.2)
        axes[2, 1].plot(t, z_m, color=color, linewidth=2)
        axes[2, 1].fill_between(t, np.minimum(z_l, z_h), np.maximum(z_l, z_h), color=color, alpha=0.2)
    axes[0, 1].axhline(threshold, color='grey', linestyle='--', linewidth=2, label=r'$\mathcal{R}_\text{crit}=1.0$')
    axes[1, 1].axhline(eps_w, color='grey', linestyle='--', linewidth=2, label=rf'$1-\varepsilon_w={1-eps_w:g}$')
    axes[2, 1].axhline(eps_w, color='grey', linestyle='--', linewidth=2, label=rf'$1-\varepsilon_w={1-eps_w:g}$')
    # Formatting
    axes[0, 1].set_title('Reported Reproductive Number')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title(rf'Instantaneous Warning Response ($\epsilon_w={eps_w}$)')
    axes[1, 1].set_ylim(-0.2, 1.2)
    axes[1, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title('Effective Transmission Modification')
    axes[2, 1].set_xlabel('Days')
    axes[2, 1].set_ylim(-0.2, 1.2)
    axes[2, 1].grid(True, alpha=0.3)
    fig.suptitle("Wastewater warning response", fontsize=16)
    plt.savefig(path); plt.close(fig)

def _wave_end_time(tt, infected, wave_floor):
    """First time after peak at which wave drops below wave_floor."""
    peak = int(np.argmax(infected))
    tail = infected[peak:]
    below = np.flatnonzero(tail < wave_floor)
    if below.size == 0:
        return float("inf")
    return float(tt[peak + int(below[0])])
