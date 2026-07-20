"""
Stability analysis.
"""

import numpy as np
import jax
from math import comb
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from functools import partial
import models


def arg_L(omega, tau_W, tau_B, n_W=3, n_B=1):
    return -n_W*np.arctan(omega*tau_W/n_W) - n_B*np.arctan(omega*tau_B/n_B)

def _loop_gain(eps_w, k, R_crit=1.0):
    """Static loop gain L(0) = K * R_eps = eps_w * k * R_crit / (2 * (2 - eps_w))."""
    return (eps_w * k * R_crit) / (2 * (2 - eps_w))

def _characteristic_polynomial(tau_W, tau_B, eps_w, k, n_W=3, n_B=1, R_crit=1.0):
    """pW * pB + L0 = 0."""
    P = np.convolve(
        np.array([comb(n_W, j) * (tau_W/n_W)**j for j in range(n_W+1)]),
        np.array([comb(n_B, j) * (tau_B/n_B)**j for j in range(n_B+1)]))
    P[0] += _loop_gain(eps_w, k, R_crit)
    return P

def dominant_pole(tau_W, tau_B, eps_w, k, n_W=3, n_B=1, R_crit=1.0):
    """Dominant complex root of characteristic polynomial."""
    roots = np.roots(_characteristic_polynomial(tau_W, tau_B, eps_w, k, n_W, n_B, R_crit)[::-1])
    complex_roots = roots[np.abs(roots.imag) > 1e-9]
    if complex_roots.size == 0: return np.nan
    return complex_roots[np.argmax(complex_roots.real)]

@partial(jax.jit, static_argnames=['model'])
def compute_rt_grid(model, base_params, taus_W, taus_B, t1=300.0):
    """True Rt in (tau_W, tau_B)."""
    def _rt(tau_W, tau_B):
        params = base_params.update(tau_W=tau_W, tau_B=tau_B)
        _, yy = model(params=params, t1=t1)
        return params.R_0 * params.rho * yy[:,-1] * yy[:,0]
    return jax.vmap(jax.vmap(_rt, in_axes=(None, 0)), in_axes=(0, None))(taus_W, taus_B)

def period_and_damping(t, x, t0=50.0, t1=250.0, smoothing_days=20.0, peak_threshold=0.2, T_min=4.0, T_max=200.0):
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

def _root(fn, lo=1e-10, hi=1000.0):
    try: return brentq(fn, lo, hi)
    except ValueError: return None

def gain_margin(eps_w, tau_W, tau_B, n_W=3, n_B=1, k=10.0, R_crit=1.0):
    """M_G = |L(j omega_PC)|^-1 at phase crossover arg L = -pi."""
    L0 = _loop_gain(eps_w, k, R_crit)
    if L0 <= 0: return np.inf
    omega_PC = _root(lambda w: np.pi + arg_L(omega=w, tau_W=tau_W, tau_B=tau_B, n_W=n_W, n_B=n_B))
    if omega_PC is None: return np.inf
    return (1.0/L0) * (1+(omega_PC*tau_W/n_W)**2)**(n_W/2) * (1+(omega_PC*tau_B/n_B)**2)**(n_B/2)

def phase_margin(eps_w, tau_W, tau_B, n_W=3, n_B=1, k=10.0, R_crit=1.0):
    """M_P = pi + arg L(j omega_c) at gain crossover |L| = 1."""
    L0 = _loop_gain(eps_w, k, R_crit)
    def g(omega):
        return (L0**2 * (1 + (omega*tau_W/n_W)**2)**(-n_W) * (1 + (omega*tau_B/n_B)**2)**(-n_B) - 1)
    if g(0) <= 0:
        return np.inf, None
    omega_c = _root(g)
    if omega_c is None:
        return np.inf, None
    return np.pi + arg_L(omega=omega_c, tau_W=tau_W, tau_B=tau_B, n_W=n_W, n_B=n_B), omega_c

def delay_margin(eps_w, tau_W, tau_B, n_W=3, n_B=1, k=10.0, R_crit=1.0, clip_negative=True):
    """M_D = M_P / omega_c."""
    M_P, omega_c = phase_margin(eps_w, tau_W, tau_B, n_W, n_B, k, R_crit)
    if omega_c is None:
        return np.inf
    M_D = M_P / omega_c
    return max(M_D, 0.0) if clip_negative else M_D
