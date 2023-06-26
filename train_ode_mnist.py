from model import ODEClassifier
from mnist_dataset import getMNISTTDataset, load_images, load_labels
from utils import train, compute_accuracy, count_parameters, CELossModel
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
from ode_solver.tableau import ExplicitEuler, ExplicitMidpoint, ExplicitTrapezoidal, Fehlberg4, Fehlberg5


torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = f"{date.today()}_{datetime.now().time()}"
model_name = model_name.replace(":", "_")
model_name = model_name.replace(".", "_")

writer = SummaryWriter(f"logs\\{model_name}")

tableau_low = ExplicitEuler()
tableau_high = None
model = ODEClassifier(
    n_channels=64,
    output_size=10,
    activation="relu",
    with_norm=False,
    tableau_low=tableau_low,
    tableau_high=tableau_high,
    t0=0.0,
    t1=1.0,
    dt=1.0 / 6.0,
    atol=1e-6,
    rtol=1e-6
).to(device)

optimizer = torch.optim.Adam(params=model.parameters())
count_parameters(model)

train_set = getMNISTTDataset(device, "train", 0, 128, True)
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