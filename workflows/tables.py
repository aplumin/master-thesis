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

def fmt(value, kind, dp, horizon=None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    if isinstance(value, float) and np.isinf(value):
        return "---" if horizon is None else rf"$>${horizon:.0f} d"
    if kind == "pct":
        return f"{100 * value:.{dp}f}\\%"
    if kind == "days":
        return f"{value:.{dp}f} d"
    return f"{value:.{dp}f}"


def render_table(groups, caption, short_caption, label, horizon, group_label_fn=str):
    """groups: list of (group_heading, [(row_label, metrics_dict), ...])"""
    header = " & ".join([r"\textbf{scenario}"] + [h for _, h, _, _ in COLUMNS])
    out = [r"\begin{table}[H]", r"\centering", r"\small", r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule", header + r"\\", r"\midrule"]
    for i, (heading, rows) in enumerate(groups):
        out.append(r"\multicolumn{%d}{l}{%s}\\" % (len(COLUMNS) + 1, group_label_fn(heading)))
        for row_label, m in rows:
            cells = [rf"\quad {row_label}"] + [fmt(m.get(key), kind, dp, horizon=horizon) for key, _, kind, dp in COLUMNS]
            out.append(" & ".join(cells) + r" \\")
        if i < len(groups) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", "}", rf"\caption[{short_caption}]{{{caption}}}\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)
