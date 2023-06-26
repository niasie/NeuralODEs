import torch
import numpy as np
from tableau import ExplicitEuler, ExplicitMidpoint, ExplicitTrapezoidal, ClassicalRK4, Kuttas38Method, Fehlberg4, Fehlberg5
from RKIntegrator import rk_solve, rk_adaptive_embedded
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
t1 = torch.tensor(10.0, device=device)
dt = torch.tensor(0.1, device=device)
tableau = Fehlberg4()

# y_final, times, states = rk_solve(y0, t0, t1, dt, f_exponential, tableau, return_all_states=True)
y_final, times, states = rk_adaptive_embedded(y0, t0, t1, dt, f_exponential, Fehlberg4(), Fehlberg5(), True, 1e-6, 1e-6)

t = np.linspace(0.0, 10.0, 100)
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