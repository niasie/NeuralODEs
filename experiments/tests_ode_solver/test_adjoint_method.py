import numpy as np
import torch
from neuralodes.ode_solver import get_ode_integrator, get_scipy_integrator, get_adjoint_integrator


weight = torch.tensor([1.], dtype=torch.float64).requires_grad_()

def f(t, x):
    result = weight * x
    return result


method_low = "fehlberg4"
method_high = None

# solver = get_ode_integrator(
#     method_low=method_low,
#     method_high=method_high,
#     atol=1e-6,
#     rtol=1e-6,
#     return_all_states=True,
# )

solver = get_scipy_integrator()
solver = get_adjoint_integrator(solver, weight)

optim = torch.optim.Adam([weight], lr=1e-1)

y0 = torch.tensor([1.], dtype=torch.float64).requires_grad_()
t0 = torch.tensor(0.)
t1 = torch.tensor(1.)
dt = torch.tensor(1e-3)

for i in range(200):
    optim.zero_grad()

    y1, _, _ = solver(f, y0, t0, t1, dt)

    # torch.autograd.gradcheck(lambda w: adjoint_wrapper(solver, f, [w], y0, t0, t1, dt), (weight))
    # torch.autograd.gradcheck(lambda z0: solver(f, z0, t0, t1, dt)[0], y0)

    loss = (y1 - y0 * np.exp(5.))**2

    loss.backward()
    print(weight, weight.grad)
    optim.step()

    print(loss.item())