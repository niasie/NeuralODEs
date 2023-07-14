from neuralodes.models import ContinousNormalizingFlow
from neuralodes.ode_solver import get_ode_integrator
import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = "circles"
epochs = "12000"
cnf = None
if model == "butterfly":
    cnf = ContinousNormalizingFlow(
        z_size=2,
        n_neurons_param_net=64,
        n_functions=128,
        hidden_size=10,
        ode_solver=get_ode_integrator(
            method_low="fehlberg4",
            method_high="fehlberg5",
            atol=1e-5,
            rtol=1e-5,
            return_all_states=False,
        ),
        t0=0.0,
        t1=1.0,
        dt=0.1,
    ).to(device)
else:
    cnf = ContinousNormalizingFlow(
        z_size=2,
        n_neurons_param_net=64,
        n_functions=128,
        hidden_size=1,
        ode_solver=get_ode_integrator(
            method_low="fehlberg4",
            method_high="fehlberg5",
            atol=1e-5,
            rtol=1e-5,
            return_all_states=False,
        ),
        t0=0.0,
        t1=1.0,
        dt=0.1,
    ).to(device)

latent_distribution = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0], device=device),
    covariance_matrix=torch.tensor(
        [[0.1, 0.0], [0.0, 0.1]], device=device
    ),
)
cnf.load_state_dict(torch.load(f"models/{model}/{epochs}.pth"))

def sample_cnf(n_samples):
    cnf.eval()
    cnf.latent_to_sample = True

    bsize = 20000
    samples = np.empty((0, 2), dtype=np.float32)
    n = 0

    while n < n_samples:
        with torch.no_grad():
            latent_samples = latent_distribution.sample((bsize, )).to(device)
            x, _ = cnf(latent_samples)
            x = x.detach().cpu().numpy().astype(np.float32)
        samples = np.concatenate((samples, x), axis=0)
        n += bsize
    return samples

n_samples = 50000
samples = sample_cnf(n_samples)
plt.figure(figsize=(10, 10))
plt.scatter(samples[:, 0], samples[:, 1], s=0.05, c="black")
plt.xticks([])
plt.yticks([])
plt.savefig(f"report/{model}_samples.png", bbox_inches='tight', pad_inches=0)

for i, t in enumerate(np.linspace(0.0, 1.0, 4)):
    cnf.t1 = torch.tensor(t, dtype=torch.float32)
    print(t)
    samples = sample_cnf(n_samples)
    plt.cla()
    plt.figure(figsize=(10, 10))
    plt.hist2d(samples[:, 0], samples[:, 1], bins=500, density=True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"report/{model}_{i}.png", bbox_inches='tight', pad_inches=0)