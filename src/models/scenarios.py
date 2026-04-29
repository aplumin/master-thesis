"""
Functions for running models.
"""

import jax
from functools import partial
from typing import Callable

from models.parameters import Params, logistic_response_function


# TODO: should change from predefined time to stable number after interventions and before natural depletion of susceptibles
@partial(jax.jit, static_argnames=['model', 't1'])
def compute_R_grid(model: Callable, base_params: Params, eps_ww: float, eps_ss: float, t1: float = 100.0, E0: float = 1e-6):
    """Compute a 2D grid of Rt values with wastewater warning response efficacy on the x axis and isolation efficacy on the y axis."""
    def final_R(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        _, yy = model(params=params, t1=t1, E0=E0)
        return params.R_0 * params.rho * yy[-1, -1] * yy[-1, 0]
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid(model: Callable, base_params: Params, eps_ww, eps_ss, t1: float = 100.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline. 
    Wastewater warning response efficacy on the x axis and isolation efficacy on the y axis.
    """
    def I_tot(w, s):
        params = base_params.update(epsilon_w=w, epsilon_s=s)
        _, yy =  model(params=params, t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(eps_ww, eps_ss)
    return I_tot_grid / I_tot(0.0, 0.0)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Rt(model: Callable, base_params: Params, p: float, phi: float, t1: float = 50.0, E0: float = 1e-6):
    """
    Compute a 2D grid of the reproductive number after interventions.
    Asymptomatic proportion p on the x axis and relative infectiousness phi on the y axis.
    """
    def final_R(p, phi):
        params = base_params.update(p=p, phi=phi)
        _, yy = model(params=params, t1=t1, E0=E0)
        Is_final = yy[-1, -(params.n_W + params.n_B + 2)]
        # TODO: this assumes n_B > 0
        return params.R_0 * params.rho * logistic_response_function(yy[-1,-1], params, Is_final) * yy[-1,0]
    return jax.vmap(jax.vmap(final_R, in_axes=(0, None)), in_axes=(None, 0))(p, phi)

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_asymptomatic_grid_Itot(model: Callable, base_params: Params, p: float, phi: float, t1: float = 600.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to a no intervention baseline.
    Asymptomatic proportion p on the x axis and relative infectiousness phi on the y axis.
    """
    def I_tot(p, phi):
        _, yy = model(params=base_params.update(p=p, phi=phi), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(p, phi)
    return I_tot_grid # return absolute fraction infected

@partial(jax.jit, static_argnames=['model', 't1'])
def compute_I_tot_grid_delayed_ww(model: Callable, base_params: Params, taus, I_crit_list, t1: float = 100.0, E0: float = 1e-6):
    """
    Compute a 2D grid of proportion infected relative to baseline across different
    wastewater reporting delays and infection intervention thresholds.
    """
    def I_tot(tau_B, I_crit):
        _, yy = model(params=base_params.update(tau_W=tau_B, I_crit=I_crit), t1=t1, E0=E0)
        return yy[0,0] - yy[-1,0]
    
    I_tot_grid = jax.vmap(jax.vmap(I_tot, in_axes=(0, None)), in_axes=(None, 0))(taus, I_crit_list)
    _, yy_base = model(params=base_params.update(I_crit=0.0), t1=t1, E0=E0)
    return I_tot_grid / (yy_base[0,0] - yy_base[-1,0])
