"""
Export tables.
"""

import numpy as np

COLUMNS = [
    ("Rt",           r"$\mathcal{R}_t$",         "num",  2),
    ("peak_Is",      r"\textbf{peak sympt.}",    "pct",  1),
    ("time_to_peak", r"\textbf{time to peak}",   "days", 0),
    ("wave_time",    r"\textbf{wave time}",      "days", 0),
    ("itot",         r"\textbf{attack rate}",    "pct",  0),
    ("prevented",    r"\textbf{inf. prevented}", "pct",  0),
    ("isol_cost",    r"\textbf{isol. cost}",     "num",  1),
    ("warn_cost",    r"\textbf{warn cost}",      "num",  1),
]

def fmt(value, kind, dp):
    if value is None or (isinstance(value, float) and np.isnan(value)) or isinstance(value, float) and np.isinf(value):
        return "---"
    if kind == "pct":
        return f"{100 * value:.{dp}f}\\%"
    if kind == "days":
        return f"{value:.{dp}f} d"
    return f"{value:.{dp}f}"


def render_table(groups, caption, short_caption, label, group_label_fn=str):
    """groups: list of (group_heading, [(row_label, metrics_dict), ...])"""
    header = " & ".join([r"\textbf{scenario}"] + [h for _, h, _, _ in COLUMNS])
    out = [r"\begin{table}[H]", r"\centering", r"\small", r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule", header + r"\\", r"\midrule"]
    for i, (heading, rows) in enumerate(groups):
        out.append(r"\multicolumn{%d}{l}{%s}\\" % (len(COLUMNS) + 1, group_label_fn(heading)))
        for row_label, m in rows:
            cells = [rf"\quad {row_label}"] + [fmt(m.get(key), kind, dp) for key, _, kind, dp in COLUMNS]
            out.append(" & ".join(cells) + r" \\")
        if i < len(groups) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", "}", rf"\caption[{short_caption}]{{{caption}}}\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)

DERIVED_ROWS = [
    ("beta",  "transmission rate",                      r"$\beta$",         "num", 3),
    ("phi_a", "relative asymptomatic infectiousness",   r"$\varphi_a$",     "num", 2),
    ("phi_p", "relative presymptomatic infectiousness", r"$\varphi_p$",     "num", 2),
    ("R_a",   "asymptomatic transmission",              r"$\mathcal{R}_a$", "num", 2),
    ("R_p",   "presymptomatic transmission",            r"$\mathcal{R}_p$", "num", 2),
    ("R_s",   "symptomatic transmission",               r"$\mathcal{R}_s$", "num", 2),
    ("theta", "nonsymptomatic transmission fraction",  r"$\theta$",        "pct", 1),
    ("generation_time", "generation time",              r"$T_g$",           "num", 1),
]

def fmt_ci(stat, kind="num", dp=2):
    """Format (central, lo, hi) as 'central (lo -- hi)'; '---' if not in the model."""
    if stat is None:
        return "---"
    central, lo, hi = (float(v) for v in stat)
    if not all(np.isfinite(v) for v in (central, lo, hi)) or (central == 0.0 and hi == 0.0):
        return "---"
    scale, unit = (100.0, r"\%") if kind == "pct" else (1.0, "")
    return f"{scale * central:.{dp}f}{unit} ({scale * lo:.{dp}f} -- {scale * hi:.{dp}f})"

def render_derived_table(stats, columns, rows=DERIVED_ROWS, caption=None, short_caption="Derived parameter estimates", label="tab:derived_parameters"):
    """stats: {column_key: {quantity: (central, lo, hi)}}, columns: [(column_key, heading), ...]"""
    caption = caption or (
        "Derived parameter estimates. Central values are evaluated at the point estimates of "
        "Table~\\ref{tab:parameters}, with the 2.5\\% to 97.5\\% range from Monte Carlo sampling of the "
        "primitive parameters in brackets. Rates are per day, periods in days. Quantities that are not "
        "part of a pathogen's model are marked ---."
    )
    out = [r"\begin{table}[H]", r"\centering", r"\renewcommand{\arraystretch}{1.2}",
           rf"\caption[{short_caption}]{{{caption}}}", rf"\label{{{label}}}",
           r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{ll%s}" % ("c" * len(columns)), r"\toprule",
           " & ".join([r"\textbf{Derived parameter}", ""] + [r"\textbf{%s}" % h for _, h in columns]) + r"\\",
           r"\midrule"]
    for key, description, symbol, kind, dp in rows:
        out.append(" & ".join([description, symbol] + [fmt_ci(stats[c].get(key), kind, dp) for c, _ in columns]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(out)
