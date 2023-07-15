from .RKIntegrator import rk_solve, rk_adaptive, rk_adaptive_embedded
from .tableau import (
    ExplicitEuler,
    ImplicitEuler,
    ExplicitMidpoint,
    ImplicitMidpoint,
    ExplicitTrapezoidal,
    ImplicitTrapezoidal,
    ClassicalRK4,
    Kuttas38Method,
    Fehlberg4,
    Fehlberg5,
)
from .AdjointWrapper import AdjointWrapper
import torch
from scipy.integrate import solve_ivp

def get_tableau(name):
    name = name.lower()
    if name == "explicit_euler":
        return ExplicitEuler()
    elif name == "implicit_euler":
        return ImplicitEuler()
    elif name == "explicit_midpoint":
        return ExplicitMidpoint()
    elif name == "implicit_midpoint":
        return ImplicitMidpoint()
    elif name == "explicit_trapezoidal":
        return ExplicitTrapezoidal()
    elif name == "implicit_trapezoidal":
        return ImplicitTrapezoidal()
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

def get_scipy_integrator(method="RK45", return_all_states=True):
    def integrator(f, z0, t0, t1, dt):
        shape = z0.shape
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SciPy solve_ivp works with 1d Numpy arrays
        # Convert the n-d tensor into a 1d numpy array
        z0 = z0.clone().detach().reshape(-1).cpu().numpy()
        
        # f works with n-d tensors -> need to transform to n-d tensor, apply f, then retransform to 1d numpy array
        f_compat = lambda t, x : f(torch.tensor(t).to(device), torch.tensor(x).to(device).reshape(*shape).float()).clone().detach().reshape(-1).cpu().numpy()
        
        dt = abs(dt.numpy())
        sol = solve_ivp(f_compat, [t0, t1], z0, first_step=dt, method=method, lband=shape[0]//2+1, uband=shape[0]//2+1)

        if return_all_states:
            return torch.tensor(sol.y[:, -1]).to(device).reshape(*shape).float(), list(sol.t), list(sol.y)
        else:
            return torch.tensor(sol.y[-1]).to(device).reshape(*shape).float(), [], []
    
    print(f"Using the Scipy implementation of the {method} method.")
    return integrator

def get_adjoint_integrator(solver, *f_params):
    def adjoint_integrator(f, z0, t0, t1, dt):
        return AdjointWrapper.apply(
            solver, f, z0, t0, t1, dt, *f_params)
    
    return adjoint_integrator