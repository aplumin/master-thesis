"""
Spatial metapopulation model with two demes.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from models.parameters import Params, logistic_response_function
from models.compartmental import chain_derivative, _solve


class SpatialParams(NamedTuple):
    """
    Parameters for spatial metapopulation model with two demes, A and B.

    Attributes:
        epi_params (Params): Epidemiological parameters.
        N_A (float): Population fraction in deme A (N_B = 1 - N_A).
        m (float): Migration rate from A to B. The backmigration rate is scaled to conserve population sizes.
    """
    epi_params: Params
    N_A: float
    m: float

    def update(self, **kwargs) -> "SpatialParams":
        fields = set(Params._fields)
        return self._replace(
            epi_params = self.epi_params.update(**{k: v for k, v in kwargs.items() if k in fields}),
            **{k: v for k, v in kwargs.items() if k not in fields}
        )


def _SEIPAR_spatial(X, prevalence, B_out, ps):
    S, E, Ia, Ip, Is, R = X
    prev_a, prev_p, prev_s = prevalence
    lambda_ = B_out * ps.beta * (ps.phi_a * prev_a + ps.phi_p * prev_p + (1.0 - ps.epsilon_s) * prev_s) * S
    become_infectious = E / ps.gamma_inv
    become_symptomatic = Ip / ps.sigma_inv
    recover_asyx = Ia / ps.mu_a_inv
    recover_syx = Is / ps.mu_s_inv
    return jnp.stack([
        -lambda_,
        lambda_ - become_infectious,
        ps.p * become_infectious - recover_asyx,
        (1.0 - ps.p) * become_infectious - become_symptomatic,
        become_symptomatic - recover_syx,
        recover_asyx + recover_syx,
    ])

def _SEIAR_spatial(X, prevalence, B_out, ps):
    S, E, Ia, Is, R = X
    prev_a, prev_s = prevalence
    lambda_ = B_out * ps.beta * (ps.phi_a * prev_a + (1.0 - ps.epsilon_s) * prev_s) * S
    become_infectious = E / ps.gamma_inv
    recover_asyx = Ia / ps.mu_a_inv
    recover_syx = Is / ps.mu_s_inv
    return jnp.stack([
        -lambda_,
        lambda_ - become_infectious,
        ps.p * become_infectious - recover_asyx,
        (1.0 - ps.p) * become_infectious - recover_syx,
        recover_asyx + recover_syx,
    ])

def _SEIR_spatial(X, prevalence, B_out, ps):
    S, E, II, R = X
    (prev_s,) = prevalence
    lambda_ = B_out * ps.beta * (1.0 - ps.epsilon_s) * prev_s * S
    become_infectious = E / ps.gamma_inv
    recover = II / ps.mu_s_inv
    return jnp.stack([
        -lambda_,
        lambda_ - become_infectious,
        become_infectious - recover,
        recover,
    ])


def _div(num, denom, fill=0.0):
    """Divide where denominator > 0, else fill."""
    div = denom > 0.0
    return jnp.where(div, num / jnp.where(div, denom, 1.0), fill)

_MODELS = { # (function, labels, indices of infectious compartments, index of symptomatics)
    "SEIPAR": (_SEIPAR_spatial, ("S", "E", "Ia", "Ip", "Is", "R"), [2, 3, 4], 4),
    "SEIAR": (_SEIAR_spatial, ("S", "E", "Ia", "Is", "R"), [2, 3], 3),
    "SEIR": (_SEIR_spatial, ("S", "E", "I", "R"), [2], 2),
}

def _simulate_spatial(model_name, spatial_params, t1, E0, n_ts, primary_in_A, ww_in_B, response_in_B_to_A):
    local_fn, labels, infectious_idx, syx_idx = _MODELS[model_name]
    n_flow = len(labels)
    ps = spatial_params.epi_params
    n_W, n_B = ps.n_W, ps.n_B

    N_A = spatial_params.N_A
    N_B = 1.0 - N_A
    m_AB = spatial_params.m
    # back migration s.t. pop sizes stay constant: m_BA = m_AB * N_A / N_B
    m_BA = _div(spatial_params.m * N_A, N_B)

    def _rhs(t, y, args):
        X_A = y[:n_flow]
        X_B = y[n_flow:2 * n_flow]
        i = 2 * n_flow
        W_A = y[i:i + n_W]; i += n_W
        B_A = y[i:i + n_B]; i += n_B
        if ww_in_B:
            W_B = y[i:i + n_W]; i += n_W
            B_B = y[i:i + n_B]; i += n_B

        B_out_A = B_A[-1]
        if response_in_B_to_A: # B responds to warnings in A
            B_out_B = B_A[-1]
        elif ww_in_B: # B has own surveillance
            B_out_B = B_B[-1]
        else: # B has no behavioural response
            B_out_B = jnp.ones_like(B_out_A)

        prev_A = tuple(_div(X_A[j], N_A) for j in infectious_idx)
        prev_B = tuple(_div(X_B[j], N_B) for j in infectious_idx)

        migration = m_BA * X_B - m_AB * X_A # net migration into A
        dX_A = local_fn(X_A, prev_A, B_out_A, ps) + migration
        dX_B = local_fn(X_B, prev_B, B_out_B, ps) - migration
        dFlow = jnp.concatenate([dX_A, dX_B])

        syx_pos = infectious_idx.index(syx_idx)
        Rt_A = ps.R_0 * ps.rho * B_out_A * _div(X_A[0], N_A)
        dW_A = chain_derivative(W_A, Rt_A, n_W / ps.tau_W)
        dB_A = chain_derivative(B_A, logistic_response_function(W_A[-1], ps, prev_A[syx_pos]), n_B / ps.tau_B)
        parts = [dFlow, dW_A, dB_A]

        if ww_in_B:
            Rt_B = ps.R_0 * ps.rho * B_out_B * _div(X_B[0], N_B)
            dW_B = chain_derivative(W_B, Rt_B, n_W / ps.tau_W)
            dB_B = chain_derivative(B_B, logistic_response_function(W_B[-1], ps, prev_B[syx_pos]), n_B / ps.tau_B)
            parts += [dW_B, dB_B]

        return jnp.concatenate(parts)

    E0_A = jnp.where(primary_in_A, E0 * N_A, 0.0)
    E0_B = jnp.where(primary_in_A, 0.0, E0 * N_B)
    y0_flow = jnp.concatenate([
        jnp.stack([N_A - E0_A, E0_A]), jnp.zeros(n_flow - 2),
        jnp.stack([N_B - E0_B, E0_B]), jnp.zeros(n_flow - 2),
    ])
    chains = [jnp.zeros(n_W), jnp.ones(n_B)]
    if ww_in_B:
        chains += [jnp.zeros(n_W), jnp.ones(n_B)]
    y0 = jnp.concatenate([y0_flow] + chains)

    return _solve(_rhs, y0, ps, t1, n_ts)


_SPATIAL_STATIC = ['n_ts', 'ww_in_B', 'response_in_B_to_A', 'primary_in_A']

@partial(jax.jit, static_argnames=_SPATIAL_STATIC)
def simulate_SEIPAR_W_spatial(
    spatial_params: SpatialParams = SpatialParams(epi_params=Params.for_SEIPAR(), N_A=1.0, m=0.0),
    t1: float = 200.0, E0: float = 1e-6, n_ts: int = 5000,
    primary_in_A: bool = True, ww_in_B: bool = False, response_in_B_to_A: bool = False,
):
    """
    Two-deme SEIPAR model with migration.
    States: S_A, E_A, Ia_A, Ip_A, Is_A, R_A, S_B, E_B, Ia_B, Ip_B, Is_B, R_B, W_A(n_W), B_A(n_B), [W_B(n_W), B_B(n_B)].
    The W_B/B_B chains are included only if ww_in_B=True. 
    With response_in_B_to_A=True, deme B has the same response as deme A.
    """
    return _simulate_spatial("SEIPAR", spatial_params, t1, E0, n_ts, primary_in_A, ww_in_B, response_in_B_to_A)

@partial(jax.jit, static_argnames=_SPATIAL_STATIC)
def simulate_SEIAR_W_spatial(
    spatial_params: SpatialParams = SpatialParams(epi_params=Params.for_SEIAR(), N_A=1.0, m=0.0),
    t1: float = 200.0, E0: float = 1e-6, n_ts: int = 5000,
    primary_in_A: bool = True, ww_in_B: bool = False, response_in_B_to_A: bool = False,
):
    """
    Two-deme SEIAR model with migration.
    States: S_A, E_A, Ia_A, Is_A, R_A, S_B, E_B, Ia_B, Is_B, R_B, W_A(n_W), B_A(n_B), [W_B(n_W), B_B(n_B)].
    The W_B/B_B chains are included only if ww_in_B=True. 
    With response_in_B_to_A=True, deme B has the same response as deme A.
    """
    return _simulate_spatial("SEIAR", spatial_params, t1, E0, n_ts, primary_in_A, ww_in_B, response_in_B_to_A)


@partial(jax.jit, static_argnames=_SPATIAL_STATIC)
def simulate_SEIR_W_spatial(
    spatial_params: SpatialParams = SpatialParams(epi_params=Params.for_SEIR(), N_A=1.0, m=0.0),
    t1: float = 200.0, E0: float = 1e-6, n_ts: int = 5000,
    primary_in_A: bool = True, ww_in_B: bool = False, response_in_B_to_A: bool = False,
):
    """
    Two-deme SEIR model with migration.
    States: S_A, E_A, I_A, R_A, S_B, E_B, I_B, R_B, W_A(n_W), B_A(n_B), [W_B(n_W), B_B(n_B)].
    The W_B/B_B chains are included only if ww_in_B=True. 
    With response_in_B_to_A=True, deme B has the same response as deme A.
    """
    return _simulate_spatial("SEIR", spatial_params, t1, E0, n_ts, primary_in_A, ww_in_B, response_in_B_to_A)


def unpack_spatial(ys, epi_params: Params, model: str = "SEIR", ww_in_B: bool = False, response_in_B_to_A: bool = False):
    """Unpack a trajectory into compartment dictionary."""
    if model not in _MODELS:
        raise ValueError(model)
    _, labels, _, _ = _MODELS[model]
    n_flow, n_W, n_B = len(labels), epi_params.n_W, epi_params.n_B
    expected = 2 * n_flow + (2 if ww_in_B else 1) * (n_W + n_B)
    if ys.shape[1] != expected:
        raise ValueError(ys.shape[1])
    out = {f"{lab}_A": ys[:, j] for j, lab in enumerate(labels)}
    out.update({f"{lab}_B": ys[:, n_flow + j] for j, lab in enumerate(labels)})
    idx = 2 * n_flow
    out["W_A"] = ys[:, idx:idx + n_W]; idx += n_W
    out["B_A"] = ys[:, idx:idx + n_B]; idx += n_B
    if ww_in_B:
        out["W_B"] = ys[:, idx:idx + n_W]; idx += n_W
        out["B_B"] = ys[:, idx:idx + n_B]; idx += n_B
    elif response_in_B_to_A:
        out["W_B"], out["B_B"] = out["W_A"], out["B_A"]
    return out


_RUN_FUNCTIONS = {"SEIR": simulate_SEIR_W_spatial, "SEIAR": simulate_SEIAR_W_spatial, "SEIPAR": simulate_SEIPAR_W_spatial}
_DEFAULT_PARAMS = {"SEIR": Params.for_SEIR, "SEIAR": Params.for_SEIAR, "SEIPAR": Params.for_SEIPAR}

def run_spatial(N_A, m, epi_params: Params = None, response_in_B_to_A=False, model="SEIPAR", t1=1000.0, E0=1e-6, ww_in_B=False):
    """
    Run spatial simulation.
    Returns (Itot_A, Itot_B, peak_I_A, peak_I_B, total_infections).
    """
    if model not in _MODELS:
        raise ValueError(model)
    if epi_params is None:
        epi_params = _DEFAULT_PARAMS[model]()
    _, labels, infectious_idx, _ = _MODELS[model]
    sp = SpatialParams(epi_params=epi_params, N_A=N_A, m=m)
    _, ys = _RUN_FUNCTIONS[model](sp, primary_in_A=False, response_in_B_to_A=response_in_B_to_A, ww_in_B=ww_in_B, t1=t1, E0=E0)
    d = unpack_spatial(ys, epi_params, model=model, ww_in_B=ww_in_B, response_in_B_to_A=response_in_B_to_A)
    N_B = 1.0 - N_A
    inf_A = sum(d[f"{labels[j]}_A"] for j in infectious_idx)
    inf_B = sum(d[f"{labels[j]}_B"] for j in infectious_idx)
    Itot_A = _div(d["R_A"][-1], N_A)
    Itot_B = _div(d["R_B"][-1], N_B)
    peak_A = jnp.max(_div(inf_A, N_A))
    peak_B = jnp.max(_div(inf_B, N_B))
    return Itot_A, Itot_B, peak_A, peak_B, d["R_A"][-1] + d["R_B"][-1]
