"""
Control theory stability analysis of the warning feedback loop.
"""

from math import comb

import numpy as np
from scipy.optimize import brentq

from models.parameters import Params


def logistic(x):
    """sigma(x) = 1 / (1 + exp(-x)). Clipped to -80 < x < 80."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -80.0, 80.0)))

def arg_L(omega, tau_W, tau_B, n_W=3, n_B=1):
    """Phase of the open-loop transfer function, L(j*omega)."""
    return -n_W*np.arctan(omega*tau_W/n_W) - n_B*np.arctan(omega*tau_B/n_B)

def _operating_point(R, eps_w, k, R_crit=1.0):
    """
    Closed-loop equilibrium of r = R * (1 - eps_w * sigma(k * (r - R_crit))),
    where R = R_0 * rho * S is the reproductive number before the warning response.
    """
    def f(r):
        return R * (1.0 - eps_w * logistic(k * (r - R_crit))) - r
    return brentq(f, 1e-9, max(10.0, 2.0 * R))

def loop_gain(R, eps_w, k, R_crit=1.0, at_midpoint=True):
    """Static loop gain L(0) = K * R_eps = eps_w * k * R_crit / (2 * (2 - eps_w))."""
    if at_midpoint:
        return (eps_w * k * R_crit) / (2.0 * (2.0 - eps_w))
    s = logistic(k * (_operating_point(R, eps_w, k, R_crit) - R_crit))
    return R * eps_w * k * s * (1.0 - s)

def _characteristic_polynomial(ps: Params):
    """Closed-loop characteristic polynomial pW(s) * pB(s) + L0."""
    P = np.convolve(
        np.array([comb(ps.n_W, j) * (ps.tau_W/ps.n_W)**j for j in range(ps.n_W+1)]),
        np.array([comb(ps.n_B, j) * (ps.tau_B/ps.n_B)**j for j in range(ps.n_B+1)])
    )
    P[0] += loop_gain(ps.R_0 * ps.rho, ps.epsilon_w, ps.k, ps.R_crit)
    return P

def dominant_pole(ps: Params):
    """Dominant complex root of characteristic polynomial."""
    roots = np.roots(_characteristic_polynomial(ps)[::-1])
    complex_roots = roots[np.abs(roots.imag) > 1e-9]
    if complex_roots.size == 0: 
        return complex(np.nan, np.nan)
    return complex_roots[np.argmax(complex_roots.real)]

def period_and_damping(t, x, t0=50.0, t1=250.0, peak_threshold=0.2, T_min=4.0, T_max=200.0):
    """Estimate period and damping rate from a trajectory x."""
    t_m = t[(t>t0) & (t<t1)]
    x_m = x[(t>t0) & (t<t1)]
    dt = float(t_m[1] - t_m[0])

    # normalise around 0
    x_m = x_m - x_m.mean()
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

def _root(fn, lo=1e-10, hi=1e10):
    try: return brentq(fn, lo, hi)
    except ValueError: return None

def gain_margin(L0, tau_W, tau_B, n_W=3, n_B=1):
    """M_G = |L(j omega_PC)|^-1 at phase crossover arg L = -pi."""
    if L0 <= 0:
        return np.inf
    omega_PC = _root(lambda w: np.pi + arg_L(w, tau_W, tau_B, n_W, n_B))
    if omega_PC is None:
        return np.inf
    return (1.0/L0) * (1 + (omega_PC*tau_W/n_W)**2)**(n_W/2) * (1 + (omega_PC*tau_B/n_B)**2)**(n_B/2)

def phase_margin(L0, tau_W, tau_B, n_W=3, n_B=1):
    """M_P = pi + arg L(j omega_c) at the gain crossover |L| = 1."""
    def g(w):
        return (L0**2 * (1 + (w*tau_W/n_W)**2)**(-n_W) * (1 + (w*tau_B/n_B)**2)**(-n_B) - 1)
    if g(0) <= 0:
        return np.inf, None
    omega_c = _root(g)
    if omega_c is None:
        return np.inf, None
    return np.pi + arg_L(omega_c, tau_W, tau_B, n_W, n_B), omega_c

def delay_margin(ps: Params, clip_negative=False):
    """M_D = M_P / omega_c."""
    L0 = loop_gain(ps.R_0 * ps.rho, ps.epsilon_w, ps.k, ps.R_crit)
    M_P, omega_c = phase_margin(L0, ps.tau_W, ps.tau_B, ps.n_W, ps.n_B)
    if omega_c is None:
        return np.inf
    M_D = M_P / omega_c
    return max(M_D, 0.0) if clip_negative else M_D

def critical_loop_gain(tau_W=14.0, tau_B=7.0, n_W=3, n_B=1):
    """Static loop gain at the Hopf boundary: L0 s.t. |L(j w_PC)| = 1 at the phase crossover."""
    def magnitude(w):
        return (1.0 + (w * tau_W / n_W) ** 2) ** (-n_W / 2.0) * (1.0 + (w * tau_B / n_B) ** 2) ** (-n_B / 2.0)
    return float(1.0 / magnitude(brentq(lambda w: np.pi + arg_L(w, tau_W, tau_B, n_W, n_B), 1e-10, 1e10)))

def k_crit(L0c, eps_w, R_crit=1.0):
    """k_crit(eps_w) = 2 * L0_crit * (2 - eps_w) / (eps_w * R_crit)."""
    eps_w = np.asarray(eps_w, dtype=float)
    return np.where(eps_w > 0, 2.0 * L0c * (2.0 - eps_w) / (eps_w * R_crit), np.inf)

def eps_w_crit(L0c, k, R_crit=1.0):
    """eps_w_crit(k) = 4 * L0_crit / (k R_crit + 2 * L0_crit)."""
    return float(4.0 * L0c / (k * R_crit + 2.0 * L0c))
