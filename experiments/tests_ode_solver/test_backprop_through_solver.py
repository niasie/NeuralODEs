import torch
from neuralodes import rk_solve, rk_adaptive_embedded
from neuralodes import ExplicitEuler, ExplicitMidpoint, ExplicitTrapezoidal, Fehlberg4, Fehlberg5
import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ee = ExplicitEuler()
em = ExplicitMidpoint()
et = ExplicitTrapezoidal()
f4 = Fehlberg4()
f5 = Fehlberg5()


class FOptim(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff = torch.nn.Parameter(torch.tensor(-1.0, device=device), requires_grad=True)
    
    def forward(self, t, y):
        return self.coeff * y



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

method = f5
l = torch.nn.functional.mse_loss
f_optim.train()
for i in range(epochs):
    optimizer.zero_grad()
    # y_final, times, states = rk_solve(y0, t0, t1, dt, f_optim, method, False)
    y_final, times, states = rk_adaptive_embedded(y0, t0, t1, dt, f_optim, f4, f5, False)
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

# y_final, times, states = rk_solve(y0, t0, t1, dt, f_optim, method, True)
y_final, times, states = rk_adaptive_embedded(y0, t0, t1, dt, f_optim, f4, f5, True)
    
times = np.array(times)
states = np.concatenate([np.expand_dims(state, 0) for state in states])
plt.scatter(times, states[:, 0].flatten(), label="y0=5.0", marker="x")
plt.show()