from sklearn.datasets import make_circles
import torch
import numpy as np
from neuralodes.ode_solver import get_ode_integrator
from neuralodes.models import ContinousNormalizingFlow
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

noise = 0.05
factor = 0.5
def sample_dataset(n_samples):
    samples, _ = make_circles(n_samples, noise=noise, factor=factor)
    samples = torch.tensor(samples, dtype=torch.float32, device=device)
    return samples


def plot_density_map(samples, label):
    x = samples[:, 0]
    y = samples[:, 1]
    nbins=300
    k = gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
    plt.show()


def create_scatterplots(n_samples):
    true_samples = sample_dataset(n_samples).cpu().numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
    ax0.scatter(true_samples[:, 0], true_samples[:, 1], label="true")
    ax0.set_title("dataset")
    ax0.set_xlim(-1.25, 1.25)
    ax0.set_ylim(-1.25, 1.25)
    
    latent_samples = latent_distribution.sample((n_samples, )).to(device)
    cnf.eval()
    cnf.latent_to_sample = True
    x, _ = cnf(latent_samples)
    x = x.detach().cpu().numpy()
    ax1.scatter(x[:, 0], x[:, 1], label="sampled")
    ax1.set_title("sampled")
    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-1.25, 1.25)
    return fig


cnf = ContinousNormalizingFlow(
    z_size=2,
    n_neurons_param_net=32,
    n_functions=64,
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

epochs = 10000
batchsize = 512
optimizer = torch.optim.Adam(cnf.parameters())
for i in range(1, epochs + 1):
    cnf.train()
    cnf.latent_to_sample = False
    optimizer.zero_grad()

    x = sample_dataset(batchsize)
    z, logpz_modification = cnf(x)
    logpz = latent_distribution.log_prob(z)

    logp_x = logpz - logpz_modification
    loss = -logp_x.mean()

    loss.backward()
    optimizer.step()
    print(f"Epoch {i}: loss = {loss.item()}")
    if i % 100 == 0:
        fig = create_scatterplots(1000)
        plt.savefig(f"figs\\{i}.png")