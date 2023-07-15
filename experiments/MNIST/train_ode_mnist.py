from mnist_dataset import getMNISTTDataset, load_images, load_labels
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
from neuralodes.models import ConvolutionalODEClassifier
from neuralodes.ode_solver import get_ode_integrator, get_scipy_integrator
from neuralodes.utils import (
    train,
    count_parameters,
    CELossModel,
    compute_accuracy,
)

torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = f"{date.today()}_{datetime.now().time()}"
model_name = model_name.replace(":", "_")
model_name = model_name.replace(".", "_")

writer = SummaryWriter(f"logs\\{model_name}")

model = ConvolutionalODEClassifier(
    ode_solver=get_ode_integrator(
        method_low="implicit_euler",
        method_high=None,
        atol=1e-3,
        rtol=1e-3,
        return_all_states=False,
    ),
    # ode_solver=get_scipy_integrator(method="LSODA"),
    adjoint_grads=True,
    in_channels=1,
    n_channels=64,
    output_size=10,
    kernel_size=3,
    activation="relu",
    n_downsampling_blocks=2,
    with_norm=False,
    t0=0.0,
    t1=1.0,
    dt=1.0/6.0,
).to(device)

optimizer = torch.optim.Adam(params=model.parameters())
count_parameters(model)

train_set = getMNISTTDataset(device, "train", 0, 1, True)
test_input = load_images(device, "test")
test_labels = load_labels(device, "test")
test_set = (test_input, test_labels)

train(
    model=model,
    train_set=train_set,
    optimizer=optimizer,
    loss_model=CELossModel(batched=True),
    num_epochs=100,
    writer=writer,
    validation_interval=1,
    validation_set=test_set,
    validation_model=compute_accuracy,
    mode="max",
    checkpoint_path=f"logs\\{model_name}",
)
print(torch.cuda.max_memory_allocated() / (1024 ** 3))