import numpy as np
import torch
from scipy.integrate import solve_ivp
from neuralodes.ode_solver import adjoint_wrapper

def scipy_ode_wrapper(f, z0, t0, t1, *solver_args, **solver_kwargs):
    solver_kwargs['method'] = 'RK45'
    sol = solve_ivp(f, [t0, t1], z0, *solver_args, **solver_kwargs)
    return torch.tensor(sol.y)

weight = torch.tensor(1.).requires_grad_()

def f(t, x):
    return weight * x

solver = scipy_ode_wrapper
optim = torch.optim.Adam([weight], lr=5e-2)

y0 = torch.tensor([1.]).requires_grad_()
t0 = torch.tensor(0.)
t1 = torch.tensor(1.)

# torch.autograd.gradcheck(lambda w: adjoint_wrapper(solver, f, [w], y0, t0, t1), (weight))

for i in range(200):
    optim.zero_grad()

    y1 = adjoint_wrapper(solver, f, [weight], y0, t0, t1)

    loss = torch.abs(y1 - np.exp(5.))

    loss.backward()
    print(weight, weight.grad)
    optim.step()

    print(loss.item())