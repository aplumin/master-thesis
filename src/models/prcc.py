import numpy as np
from scipy.stats import qmc, rankdata
from functools import partial

import jax
import jax.numpy as jnp

from models.compartmental import simulate_SEIPAR_W
from models.parameters import Params, f

def _construct_latin_hypercube(n=10_000):
    """Quasi Monte Carlo sampling from Latin hypercube of SEIPAR_W parameters."""
    params = ['R_0', 'phi', 'gamma_inv', 'sigma_inv', 'mu_a_inv', 'mu_s_inv', 'p', 'epsilon_s', 'epsilon_w', 'tau']
    bounds = {
        'R_0': (1.0, 5.0),
        'phi': (0.0, 1.0),
        'gamma_inv': (0.1, 10.0),
        'sigma_inv': (0.1, 10.0),
        'mu_a_inv': (0.1, 10.0),
        'mu_s_inv': (0.1, 10.0),
        'p': (0.0, 1.0),
        'epsilon_s': (0.0, 1.0),
        'epsilon_w': (0.0, 1.0),
        'tau': (1.0, 50.0)
    }
    latin_hypercube = qmc.scale(
        sample = qmc.LatinHypercube(d=len(params)).random(n=n), 
        l_bounds = np.array([bounds[p][0] for p in params]),
        u_bounds = np.array([bounds[p][1] for p in params])
    )
    return latin_hypercube

def _run_latin_hypercube_sampling(latin_hypercube, base_params=Params.for_SEIPAR(), t1=50.0, E0=1e-6, total_infected=False):
    """
    Run SEIPAR_W model with LHS parameters.
    Return Rt by default. If total_infected, return proportion infected compared to baseline.
    """
    def _single_latin_hypercube_sample(latin_hypercube_row, base_params, t1, E0, total_infected):
        R_0, phi, gamma_inv, sigma_inv, mu_a_inv, mu_s_inv, p, epsilon_s, epsilon_w, tau = latin_hypercube_row
        r = p * phi * mu_a_inv + (1-p) * (sigma_inv + mu_s_inv)
        r_eps = p * phi * mu_a_inv + (1-p) * (sigma_inv + (1-epsilon_s) * mu_s_inv)
        params = base_params._replace(R_0=R_0, phi=phi, gamma_inv=gamma_inv, sigma_inv=sigma_inv, mu_a_inv=mu_a_inv, mu_s_inv=mu_s_inv, p=p, epsilon_s=epsilon_s, epsilon_w=epsilon_w, tau=tau,beta=R_0/r,rho=r_eps/r)
        _, yy = simulate_SEIPAR_W(params=params, t1=t1, E0=E0)
        return yy[0, 0] - yy[-1, 0] if total_infected else params.R_0 * params.rho * f(yy[-1,-1], params) * yy[-1,0]

    sample_func = partial(_single_latin_hypercube_sample, base_params=base_params, t1=t1, E0=E0, total_infected=total_infected)
    return jax.jit(jax.vmap(sample_func))(jnp.array(latin_hypercube))

def _partial_rank_corr_coeff(latin_hypercube, y_output):
    """
    Compute the partial rank correlation coefficients between 
    model parameters and the output (e.g. Rt or proportion infected).
    """
    ranked_data = np.hstack((
        np.apply_along_axis(rankdata, 0, latin_hypercube), 
        rankdata(y_output).reshape(-1, 1)
    ))
    C = np.corrcoef(ranked_data, rowvar=False) 
    W = np.linalg.inv(C) # precision matrix
    prcc = np.array([
        -W[i, -1] / np.sqrt(W[i, i] * W[-1, -1]) # -Wxy / sqrt(Wxx * Wyy) for all params x and output y
            for i in range(latin_hypercube.shape[1])
        ])
    return prcc

def calculate_prcc(params=Params.for_SEIPAR(), t1=50.0, E0=1e-6, total_infected=False):
    """
    Calcuate the partial rank correlation coefficients by doing latin hypercube sampling 
    for SEIPAR_W model parameters and calculating the PRCC between each parameter and the output.
    """
    latin_hypercube = _construct_latin_hypercube()
    y = _run_latin_hypercube_sampling(latin_hypercube=latin_hypercube, base_params=params, t1=t1, E0=E0, total_infected=total_infected)
    return _partial_rank_corr_coeff(latin_hypercube, np.array(y))
