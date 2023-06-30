import torch
import numpy as np
from neuralodes.ode_solver import get_ode_integrator
import matplotlib.pyplot as plt


coeff = -10.0
def f_exponential(t, y):
    return coeff * y


def f_exact(t, y0):
    y = y0 * np.exp(coeff * t)
    return y

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
y0 = torch.empty((2, 1, 1, 1), dtype=torch.float32, device=device)
y0[0] = 5.0
y0[1] = 1.0
t0 = torch.tensor(0.0, device=device)
t1 = torch.tensor(1.0, device=device)
dt = torch.tensor(0.1, device=device)
method_low = "fehlberg4"
method_high = "fehlberg5"

integrator = get_ode_integrator(
    method_low=method_low,
    method_high=method_high,
    atol=1e-6,
    rtol=1e-6,
    return_all_states=True,
)
y_final, times, states = integrator(f_exponential, y0, t0, t1, dt)

t = np.linspace(0.0, 1.0, 100)
y_exact_1 = f_exact(t, 5.0)
y_exact_2 = f_exact(t, 1.0)

plt.plot(t, y_exact_1, label="y0=5.0")
plt.plot(t, y_exact_2, label="y0=1.0")

times = np.array(times)
states = np.concatenate([np.expand_dims(state, 0) for state in states])
plt.scatter(times, states[:, 0].flatten(), label="y0=5.0", marker="x")
plt.scatter(times, states[:, 1].flatten(), label="y0=1.0", marker="x")

plt.legend()
plt.show()