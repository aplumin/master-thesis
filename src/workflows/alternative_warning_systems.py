"""
Alternative warning systems.
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from models.compartmental import simulate_SEIPAR_W, simulate_SEIR_W
from models.compartmental_erlang import simulate_SEIPAR_W_Erlang
from models.compartmental_piecewise import (
    simulate_SEIPAR_W_piecewise,
    simulate_SEIR_W_piecewise,
)
from models.compartmental_piecewise_erlang import simulate_SEIPAR_W_piecewise_Erlang
from models.metrics import (
    calculate_averaged_Rt,
    column,
    cost_and_time_above,
    get_crossing,
    n_Is_compartments,
    rt_amplitude,
    trajectory_indices,
    trapz_to,
)
from workflows.plotting import _wave_end_time

CONTINUOUS = {
    simulate_SEIPAR_W_piecewise: simulate_SEIPAR_W, 
    simulate_SEIR_W_piecewise: simulate_SEIR_W,
    simulate_SEIPAR_W_piecewise_Erlang: simulate_SEIPAR_W_Erlang
}
T_LEAD = 7.0
EVAL_INTERVAL = 14.0
R_OFF = 0.8
LOCKDOWN_K = 50.0
CHECK_INTERVAL = 0.1
T_END = 6_000.0
E0 = 1e-6
GRID_T1 = 300.0
GRID_TAUS_W = np.linspace(1.0, 30.0, 100)
GRID_TAUS_B = np.linspace(1.0, 30.0, 100)
STRATEGIES = {
    "baseline": {"asymmetric": False, "discrete_eval": False, "kwargs": {}},
    "lead": {"asymmetric": False, "discrete_eval": False, "kwargs": {"T_lead": T_LEAD}},
    "interval": {"asymmetric": False, "discrete_eval": True, "kwargs": {}},
    "asymmetric": {"asymmetric": True, "discrete_eval": False, "kwargs": {}},
    "lockdown": {"asymmetric": True, "discrete_eval": True, "kwargs": {"k": LOCKDOWN_K, "eval_interval": 2 * EVAL_INTERVAL}},
}
METRIC_NAMES = ["$\\mathcal{R}_t$", "total number of infections", "symptomatic peak", "steady-state $\\mathcal{R}_t$ amplitude", "time above $\\mathcal{R}_{crit}$", "total cost"]
METRIC_BOUNDS = [(0.0, 2.5), (0.0, 1.0), (0.0, 0.02), (0.0, 1.0), (0.0, 300.0), (0.0, 175.0)]
R_CRITS = [0.9, 1.0, 1.1]
LINESTYLES = ["-", "--", "-.", ":", (0, (1, 2))]
PALETTE = sns.color_palette("colorblind", 1024)

def asymmetric_R_off(R_crit, R_off=R_OFF):
    return float(R_crit) - (1.0 - float(R_off))

def strategy_params(base_params, strategy, eps_s, eps_w, R_off=R_OFF, eval_interval=EVAL_INTERVAL):
    if strategy not in STRATEGIES:
        raise KeyError(strategy)
    strategy_dict = STRATEGIES[strategy]
    updates = {"eval_interval": eval_interval, **strategy_dict["kwargs"]}
    ps = base_params.update(epsilon_s=eps_s, epsilon_w=eps_w, **updates)
    if strategy_dict["asymmetric"]:
        if "R_off" not in updates:
            ps = ps.update(R_off=asymmetric_R_off(ps.R_crit, R_off))
    else:
        ps = ps.update(R_off=float(ps.R_crit))
    if "T_lead" not in updates:
        ps = ps.update(T_lead=0.0)
    return ps

def run_scenario(base_params, model, eps_s, eps_w, strategy="baseline", t1=T_END, E0=E0, check_interval=CHECK_INTERVAL, save_per_seg=1, **kwargs):
    spec = STRATEGIES[strategy]
    ps = strategy_params(base_params, strategy, eps_s, eps_w, **kwargs)
    if not (spec["asymmetric"] or spec["discrete_eval"] or float(ps.T_lead) > 0.0):
        tt, yy = CONTINUOUS[model](params=ps, t1=float(t1), E0=float(E0))
        return ps, tt, yy, None
    tt, yy, ms = model(
        params=ps, t1=float(t1), E0=float(E0),
        asymmetric=spec["asymmetric"], discrete_eval=spec["discrete_eval"],
        check_interval=float(check_interval), save_per_seg=int(save_per_seg),
    )
    return ps, tt, yy, ms

def scenario_metrics(ps, tt, yy, ms=None, itot_baseline=None, wave_floor=E0, delta_dep=0.05):
    idx = trajectory_indices(ps.n_W, ps.n_B, n_S=n_Is_compartments(ps))
    S = np.asarray(yy[:, idx["S"]])
    R = np.asarray(yy[:, idx["R"]])
    B_out = np.asarray(yy[:, idx["B_out"]])
    Is = np.asarray(column(yy, idx["Is"]))
    infected = 1.0 - S - R
    rt_true = float(ps.R_0) * float(ps.rho) * B_out * S
    peak_idx = int(np.argmax(Is))
    itot = float(S[0] - S[-1])
    wave_time = _wave_end_time(tt, infected, wave_floor)
    warn_state = _warning_state(ps, yy, ms)
    isol_cost = trapz_to(tt, float(ps.epsilon_s) * Is, wave_time)
    warn_cost, time_above = cost_and_time_above(tt, B_out, warn_state, wave_time)
    return {
        "Rt": float(calculate_averaged_Rt(ps, tt, S, Is, rt_true, delta_dep)),
        "peak_Is": float(Is[peak_idx]),
        "time_to_peak": float(tt[peak_idx]),
        "wave_time": wave_time,
        "itot": itot,
        "prevented": (float("nan") if itot_baseline in (None, 0.0) else 1.0 - itot / float(itot_baseline)),
        "isol_cost": isol_cost,
        "warn_cost": warn_cost,
        "isol_rate": float("nan") if not np.isfinite(wave_time) else isol_cost / wave_time,
        "warn_rate": float("nan") if not np.isfinite(wave_time) else warn_cost / wave_time,
        "time_above": time_above,
        "n_warnings": int(np.sum(np.diff(warn_state * (tt <= wave_time)) > 0.0)),
        "time_to_below_1": float(get_crossing(tt, -rt_true, -1.0, rising=True, fallback=float("nan"))),
    }

def scenario_row(base_params, model, eps_s, eps_w, strategy="baseline", itot_baseline=None, t1=T_END, E0=E0, check_interval=CHECK_INTERVAL, **kwargs):
    ps, tt, yy, ms = run_scenario(base_params, model, eps_s, eps_w, strategy=strategy, t1=t1, E0=E0, check_interval=check_interval, **kwargs)
    return scenario_metrics(ps, tt, yy, ms, itot_baseline=itot_baseline)

def _warning_state(ps, yy, ms=None):
    """Array of published warning states (1 = warning on)."""
    if ms is not None:
        return (np.asarray(ms) >= 0.5).astype(float)
    idx = trajectory_indices(ps.n_W, ps.n_B, n_S=n_Is_compartments(ps))
    W_out = np.asarray(column(yy, idx["W_out"]))
    return (W_out >= float(ps.R_crit)).astype(float)

def _true_and_reported_Rt(ps, yy):
    """True Rt and reported (delayed) Rt from a trajectory."""
    idx = trajectory_indices(ps.n_W, ps.n_B, n_S=n_Is_compartments(ps))
    S = np.asarray(yy[:, idx["S"]])
    B_out = np.asarray(column(yy, idx["B_out"]))
    rt_true = float(ps.R_0) * float(ps.rho) * B_out * S
    rt_reported = np.asarray(column(yy, idx["W_out"]))
    return rt_true, rt_reported

def strategy_grid(model, base_params, k, eps_w, eps_s, strategies=None, t1=GRID_T1, taus_W=None, taus_B=None, R_off=R_OFF, eval_interval=EVAL_INTERVAL, check_interval=CHECK_INTERVAL):
    """Delay-grid metrics for every warning strategy."""
    names = list(STRATEGIES if strategies is None else strategies)
    taus_W = GRID_TAUS_W if taus_W is None else np.asarray(taus_W, dtype=float)
    taus_B = GRID_TAUS_B if taus_B is None else np.asarray(taus_B, dtype=float)
    tW, tB = jnp.asarray(taus_W), jnp.asarray(taus_B)
    data = {}
    for s in names:
        spec = STRATEGIES[s]
        ps = strategy_params(base_params, s, eps_s, eps_w, R_off=R_off, eval_interval=eval_interval, k=k)
        data[s] = np.asarray(strategy_metric_grid(model, ps, tW, tB, float(t1), asymmetric=spec["asymmetric"], discrete_eval=spec["discrete_eval"], check_interval=float(check_interval), T_lead_on=float(ps.T_lead) > 0.0))
    return data, taus_W, taus_B

def save_strategy_grid(path, model, base_params, k, eps_w, eps_s):
    data, taus_W, taus_B = strategy_grid(model, base_params, k, eps_w, eps_s)
    names = list(data)
    np.savez_compressed(path, grid=np.stack([data[s] for s in names]), taus_W=np.asarray(taus_W), taus_B=np.asarray(taus_B), strategies=np.asarray(names), metrics=np.asarray(METRIC_NAMES))
    return path

def load_strategy_grid(path):
    npz = np.load(path)
    return (npz["grid"], npz["taus_W"], npz["taus_B"], [str(s) for s in npz["strategies"]], [str(m) for m in npz["metrics"]])

def plot_strategy_grid(path, data_path, pathogen, model, base_params, k, eps_s, eps_w, t1=GRID_T1, check_interval=CHECK_INTERVAL, metric_bounds=None, R_off=R_OFF, eval_interval=EVAL_INTERVAL):
    """Delay grid heatmaps per strategy."""
    grid, taus_W, taus_B, strategies, metric_names = load_strategy_grid(data_path)
    bounds = METRIC_BOUNDS if metric_bounds is None else metric_bounds
    nS = len(strategies)
    nM = len(metric_names)
    nrows = nM + 1
    base = base_params.update(k=k)

    sns.set_theme(style="white", rc={"axes.grid": False})
    fig = plt.figure(figsize=(3.05 * nS + 1.4, 3.05 * nrows))
    gs = fig.add_gridspec(nrows, nS + 2, width_ratios=[1.0] * nS + [0.16, 0.16], height_ratios=[1.0] * nrows, hspace=0.05, wspace=0.05)
    axs = np.empty((nrows, nS), dtype=object)
    for r in range(nrows):
        for c in range(nS):
            axs[r, c] = fig.add_subplot(gs[r, c])
            axs[r, c].set_box_aspect(1)

    ims = [None] * nM
    for c, s in enumerate(strategies):
        for r in range(nM):
            ax = axs[r, c]
            vmin, vmax = bounds[r]
            if r == 0:
                ax.set_title(s, fontsize=10)
            im = ax.imshow(grid[c, :, :, r], origin="lower", aspect="auto", cmap="magma", extent=[taus_B[0], taus_B[-1], taus_W[0], taus_W[-1]], vmin=vmin, vmax=vmax)
            if c == 0:
                ims[r] = im
                ax.set_ylabel(f"{metric_names[r]}\n$\\tau_W$", fontsize=12)
            else:
                ax.tick_params(labelleft=False)
            if r == nM - 1: 
                ax.set_xlabel("$\\tau_B$")
            else:
                ax.tick_params(labelbottom=False)

    for c, s in enumerate(strategies):
        ps, tt, yy, ms = run_scenario(base, model, eps_s, eps_w, s, t1=t1, check_interval=check_interval, R_off=R_off, eval_interval=eval_interval)
        rt_true, rt_reported = _true_and_reported_Rt(ps, yy)
        warn = _warning_state(ps, yy, ms)

        ax = axs[-1, c]
        ax.plot(tt, rt_true, color="black")
        ax.plot(tt, rt_reported, color="red")
        ax.axhline(float(ps.R_crit), color="grey", linestyle="--")
        ax.set_ylim(0, 1.75)
        ax.set_xlim(0, t1)
        if c == 0: ax.set_ylabel(f"true vs reported $R_t$\n($\\tau_W={float(ps.tau_W):g}$, $\\tau_B={float(ps.tau_B):g}$)", fontsize=11)
        ax.set_xlabel("Time (days)")
        ax.text(0.96, 0.94, f"{float(warn.mean() * t1):.0f} days above $R_{{crit}}$\n{int(np.sum(np.diff(warn) > 0))} warnings", transform=ax.transAxes, ha="right", va="top", fontsize=7)

    for r in range(nM): fig.colorbar(ims[r], cax=fig.add_subplot(gs[r, nS + 1]), orientation="vertical")
    lax = fig.add_subplot(gs[nrows - 1, nS + 1])
    lax.axis("off")
    lax.legend([Line2D([0], [0], color="black", lw=2), Line2D([0], [0], color="red", lw=2)], ["True $R_t$", "Reported $R_t$"], loc="center", ncol=1, fontsize=11, frameon=False)
    fig.suptitle(f"Warning strategy comparison ({pathogen}, $\\varepsilon_s={eps_s:g}$, $\\varepsilon_w={eps_w:g}$, $k={k:g}$)", fontsize=18)
    fig.savefig(path)
    plt.close(fig)

@partial(jax.jit, static_argnames=["model", "t1", "asymmetric", "discrete_eval", "check_interval", "T_lead_on"])
def strategy_metrics_over_epsw(model, params, eps_ww, t1, asymmetric, discrete_eval, check_interval, T_lead_on=False):
    """strategy_metrics vmapped over epsilon_w."""
    def _sm(ew):
        ps = params.update(epsilon_w=ew)
        return strategy_metrics(tau_W=ps.tau_W, tau_B=ps.tau_B, model=model, base_params=ps, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval, T_lead_on=T_lead_on)
    return jax.vmap(_sm)(eps_ww)

def plot_strategies_vs_eps_w(path, pathogen, model, base_params, eps_s, eps_ww=None, R_crits=None, strategies=None, t1=10_000.0, check_interval=1.0, R_off=R_OFF, eval_interval=EVAL_INTERVAL, palette=None, linestyles=None):
    names = list(STRATEGIES if strategies is None else strategies)
    eps_ww = np.linspace(0.0, 1.0, 50) if eps_ww is None else np.asarray(eps_ww, dtype=float)
    R_crits = R_CRITS if R_crits is None else list(R_crits)
    palette = PALETTE if palette is None else palette
    linestyles = LINESTYLES if linestyles is None else linestyles
    nM = len(METRIC_NAMES)
    nR = len(R_crits)

    fig, axs = plt.subplots(nrows=nR, ncols=nM, sharex=True, figsize=(10, 10 * nR / 6), squeeze=False)
    for row, rc in enumerate(R_crits):
        bp = base_params.update(epsilon_s=eps_s, R_crit=rc)
        no_warning = bp.update(epsilon_w=0.0, R_off=float(rc), T_lead=0.0)
        bm = np.asarray(strategy_metrics(tau_W=no_warning.tau_W, tau_B=no_warning.tau_B, model=model, base_params=no_warning, t1=t1, asymmetric=False, discrete_eval=False, check_interval=check_interval, T_lead_on=False))
        baseline = np.array([1.0, bm[1], bm[2], 1.0, t1, t1])

        for i, s in enumerate(names):
            spec = STRATEGIES[s]
            params = strategy_params(bp, s, eps_s, 0.0, R_off=R_off, eval_interval=eval_interval)
            y = np.asarray(strategy_metrics_over_epsw(model, params, jnp.asarray(eps_ww), t1, spec["asymmetric"], spec["discrete_eval"], check_interval, float(params.T_lead) > 0.0)).T
            for r in range(nM-2):
                axs[row, r].plot(eps_ww, y[r], ls=linestyles[i % len(linestyles)], color=palette[i], lw=1.3, alpha=0.9, label=s)
            for r in range(nM-2, nM):
                axs[row, r].plot(eps_ww, y[r] / baseline[r], ls=linestyles[i % len(linestyles)], color=palette[i], lw=1.3, alpha=0.9, label=s)
        for r in range(nM):
            axs[row, r].set_ylim(bottom=0.0)
            axs[row, r].set_xlim(0, 1)
            if row == 0:
                axs[row, r].set_title(METRIC_NAMES[r], fontsize=8)
            if row == nR - 1:
                axs[row, r].set_xlabel(r"$\varepsilon_w$")
        axs[row, 0].set_ylabel(rf"$\mathcal{{R}}_\mathrm{{crit}}={rc:.1f}$")

    fig.legend(handles=[Line2D([0], [0], color=palette[i], ls=linestyles[i % len(linestyles)], label=s) for i, s in enumerate(names)], loc="outside lower center", ncol=len(names), frameon=False, fontsize=10)
    pathogen_title = pathogen + ", LCT" if n_Is_compartments(base_params) > 1 else pathogen
    fig.suptitle(rf"Warning strategies for varying response strengths ({pathogen_title}, $\varepsilon_s={eps_s:g}$)")
    fig.savefig(path)
    plt.close(fig)


def plot_true_vs_reported_Rt(path, pathogen, model, base_params, strategy, k, eps_s, eps_w_values=(0.0, 0.4, 0.8, 1.0), t1=GRID_T1, check_interval=CHECK_INTERVAL, R_off=R_OFF, eval_interval=2 * EVAL_INTERVAL):
    base = base_params.update(k=k)
    sns.set_theme(style="white", rc={"axes.grid": False})
    fig, axs = plt.subplots(nrows=1, ncols=len(eps_w_values), figsize=(16, 4), sharex=True, sharey=True)

    ps = base
    for j, eps_w in enumerate(eps_w_values):
        ps, tt, yy, ms = run_scenario(base, model, eps_s, eps_w, strategy, t1=t1, check_interval=check_interval, R_off=R_off, eval_interval=eval_interval)
        rt_true, rt_reported = _true_and_reported_Rt(ps, yy)
        warn = _warning_state(ps, yy, ms)
        ax = axs[j]
        ax.set_title(f"$\\varepsilon_w={eps_w:g}$", fontsize=16)
        if j == 0:
            ax.set_ylabel("$R_t$", fontsize=16)
        ax.set_xlabel("time (days)", fontsize=12)
        ax.plot(tt, rt_true, color="black")
        ax.plot(tt, rt_reported, color="red")
        ax.axhline(float(ps.R_crit), color="grey", linestyle="--")
        ax.text(0.97, 0.95, f"{float(warn.mean() * t1):.0f} days above $R_{{crit}}$\n{int(np.sum(np.diff(warn) > 0))} warnings", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.suptitle(f"{pathogen}: $k={k:g}$, {strategy} ($\\tau_W={float(ps.tau_W):g}$, $\\tau_B={float(ps.tau_B):g}$, $\\varepsilon_s={eps_s:g}$)", fontsize=15)
    fig.legend([Line2D([0], [0], color="black", lw=2), Line2D([0], [0], color="red", lw=2)], ["True $R_t$", "Reported $R_t$"], loc="outside lower center", ncol=2, fontsize=14)
    fig.savefig(path)
    plt.close(fig)

def strategy_metrics(tau_W, tau_B, model, base_params, t1, asymmetric, discrete_eval, check_interval, T_lead_on=False):
    """
    Summary metrics for one piecewise warning strategy given delays (tau_W, tau_B):
        Rt_final, Itot, peak_Is, amplitude, time_above, cost.
    """
    params = base_params.update(tau_W=tau_W, tau_B=tau_B)
    n_W, n_B = params.n_W, params.n_B
    ts, ys, ms = model(params=params, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval)
    idx = trajectory_indices(n_W, n_B, n_S=n_Is_compartments(params))
    S, B_out = ys[:, idx["S"]], ys[:, idx["B_out"]]
    rt_true = params.R_0 * params.rho * B_out * S
    amplitude = rt_amplitude(ts, rt_true, window="final")
    if asymmetric or discrete_eval:
        time_above = jnp.trapezoid(ms, ts)
    else:
        W_out = ys[:, idx["W_out"]]
        if T_lead_on:
            R_est = W_out + params.T_lead * (n_W / params.tau_W) * (ys[:, idx["W_out"] - 1] - W_out)
        else:
            R_est = W_out
        time_above = jnp.trapezoid((R_est >= params.R_crit), ts)
    cost = jnp.trapezoid(1.0 - B_out, ts)
    Itot = S[0] - S[-1]
    Is = column(ys, idx["Is"])
    peak_Is = jnp.max(Is)
    Rt_final = calculate_averaged_Rt(params, ts, S, Is, rt_true, 0.05)
    return jnp.stack([Rt_final, Itot, peak_Is, amplitude, time_above, cost])

@partial(jax.jit, static_argnames=['model', 't1', 'asymmetric', 'discrete_eval', 'check_interval', 'T_lead_on'])
def strategy_metric_grid(model, base_params, taus_W, taus_B, t1, asymmetric, discrete_eval, check_interval, T_lead_on=False):
    """strategy_metrics over the full (tau_W, tau_B) grid."""
    strategy_metrics_vmap = partial(strategy_metrics, model=model, base_params=base_params, t1=t1, asymmetric=asymmetric, discrete_eval=discrete_eval, check_interval=check_interval, T_lead_on=T_lead_on)
    return jax.vmap(jax.vmap(strategy_metrics_vmap, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)
