from cifar10_dataset import transformations, get_datasets, get_dataloader, get_accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
from neuralodes.models import ConvolutionalODEClassifier
from neuralodes.ode_solver import (
    ExplicitEuler,
    ExplicitMidpoint,
    ExplicitTrapezoidal,
    ClassicalRK4,
    Kuttas38Method,
    Fehlberg4,
    Fehlberg5,
)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 35
batch_size = 32
learning_rate = 0.001

# CIFAR-10 dataset
transform_train, transform_test = transformations()
train_dataset, test_dataset = get_datasets(transform_train, transform_test)
train_loader, test_loader = get_dataloader(train_dataset, test_dataset, batch_size)

tableau_low = ExplicitEuler()
tableau_high = None
model = ConvolutionalODEClassifier(
    in_channels=3,
    n_channels=64,
    output_size=10,
    kernel_size=3,
    activation="relu",
    n_downsampling_blocks=2,
    with_norm=False,
    tableau_low=tableau_low,
    tableau_high=tableau_high,
    t0=0.0,
    t1=1.0,
    dt=1.0/6.0,
    atol=1e-6,
    rtol=1e-6,
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    # Training step.
    for i, (images, labels) in enumerate(train_loader):
        # Prepare input batch.
        images = images.to(device)
        labels = labels.to(device)

        # Forward + backprop + loss.
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()

        # Update model params.
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels, batch_size)
    
    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc / i)) 

test_acc = 0.0
for i, (images, labels) in enumerate(test_loader, 0):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_acc += get_accuracy(outputs, labels, batch_size)
        
print('Test Accuracy: %.2f'% (test_acc / i))