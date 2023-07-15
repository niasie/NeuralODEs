import torch
from neuralodes.ode_solver import get_ode_integrator, get_scipy_integrator
import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class FOptim(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff = torch.nn.Parameter(torch.tensor(-1.0, device=device), requires_grad=True)
    
    def forward(self, t, y, numpy=False):
        y = torch.tensor(y)

        if not numpy:
            return self.coeff * y
        else:
            return (self.coeff * y).detach().numpy()



f_optim = FOptim()
y0 = torch.empty((1, 1), dtype=torch.float32, device=device)
y0[0] = 5.0
y_target = torch.empty((1, 1), dtype=torch.float32, device=device)
y_target[0] = 1.0
t0 = torch.tensor(0.0, device=device)
t1 = torch.tensor(1.0, device=device)
dt = torch.tensor(0.1, device=device)

optimizer = torch.optim.Adam(f_optim.parameters(), lr=1e-2)
epochs = 250

# method_low = "fehlberg4"
# method_high = None

# integrator = get_ode_integrator(
#     method_low=method_low,
#     method_high=method_high,
#     atol=1e-6,
#     rtol=1e-6,
#     return_all_states=True,
# )

integrator = get_scipy_integrator()

l = torch.nn.functional.mse_loss
f_optim.train()
for i in range(epochs):
    optimizer.zero_grad()
    # y_final, times, states = rk_solve(y0, t0, t1, dt, f_optim, method, False)
    y_final, times, states = integrator(f_optim, y0, t0, t1, dt)
    print(f"i = {i}, y_final = {y_final.item()}")
    loss = l(y_final, y_target)
    loss.backward()
    optimizer.step()


f_optim.eval()
def f_exact(t):
    return 5.0 * np.exp(-np.log(5) * t)

t = np.linspace(0.0, 1.0, 100)
y_exact_1 = f_exact(t)
plt.plot(t, y_exact_1, label="y0=5.0")

y_final, times, states = integrator(f_optim, y0, t0, t1, dt)
    
times = np.array(times)
states = np.concatenate([np.expand_dims(state, 0) for state in states])
plt.scatter(times, states[:, 0].flatten(), label="y0=5.0", marker="x")
plt.show()