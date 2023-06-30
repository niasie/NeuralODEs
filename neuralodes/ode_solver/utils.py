from .RKIntegrator import rk_solve, rk_adaptive, rk_adaptive_embedded
from .tableau import (
    ExplicitEuler,
    ExplicitMidpoint,
    ExplicitTrapezoidal,
    ClassicalRK4,
    Kuttas38Method,
    Fehlberg4,
    Fehlberg5,
)
import torch


def get_tableau(name):
    name = name.lower()
    if name == "explicit_euler":
        return ExplicitEuler()
    elif name == "explicit_midpoint":
        return ExplicitMidpoint()
    elif name == "explicit_trapezoidal":
        return ExplicitTrapezoidal()
    elif name == "classikal_rk4":
        return ClassicalRK4()
    elif name == "kutta38":
        return Kuttas38Method()
    elif name == "fehlberg4":
        return Fehlberg4()
    elif name == "fehlberg5":
        return Fehlberg5()
    else:
        raise ValueError(f"integrator {name} is not available")

def get_ode_integrator(
        method_low,
        method_high=None,
        atol=1e-3,
        rtol=1e-3,
        return_all_states=True,
    ):
    tableau_low = get_tableau(method_low)
    tableau_high = (
        get_tableau(method_high) if method_high is not None
        else None
    )
    if tableau_high is None:
        def integrator(f, y0, t0, t1, dt):
            return rk_solve(
                y0_=y0,
                t0_=t0,
                t1_=t1,
                dt_=dt,
                f=f,
                tableau=tableau_low,
                return_all_states=return_all_states,
            )
        print("Using a non-adaptive method.")
        return integrator
    else:
        if torch.equal(tableau_low.a, tableau_high.a):
            def integrator(f, y0, t0, t1, dt):
                return rk_adaptive_embedded(
                    y0_=y0,
                    t0_=t0,
                    t1_=t1,
                    dt_=dt,
                    f=f,
                    tableau_low=tableau_low,
                    tableau_high=tableau_high,
                    return_all_states=return_all_states,
                    atol=atol,
                    rtol=rtol,
                )
            print("Using an adaptive embedded method.")
            return integrator
        else:
            def integrator(f, y0, t0, t1, dt):
                return rk_adaptive(
                    y0_=y0,
                    t0_=t0,
                    t1_=t1,
                    dt_=dt,
                    f=f,
                    tableau_low=tableau_low,
                    tableau_high=tableau_high,
                    return_all_states=return_all_states,
                    atol=atol,
                    rtol=rtol,
                )
            print("Using an adaptive, non-embedded method.")
            return integrator
