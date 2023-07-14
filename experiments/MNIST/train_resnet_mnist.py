from mnist_dataset import getMNISTTDataset, load_images, load_labels
from neuralodes.models import ResNetConv
from neuralodes.utils import train, compute_accuracy, count_parameters, CELossModel
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime


torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = f"{date.today()}_{datetime.now().time()}"
model_name = model_name.replace(":", "_")
model_name = model_name.replace(".", "_")

writer = SummaryWriter(f"logs\\{model_name}")

model = ResNetConv(
    num_blocks=6,
    in_channels=1,
    n_channels=64,
    output_size=10,
    activation="relu",
    with_norm=False,
    kernel_size=3,
    n_downsampling_blocks=2,
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
print(torch.cuda.max_memory_allocated() / (1024 ** 3))