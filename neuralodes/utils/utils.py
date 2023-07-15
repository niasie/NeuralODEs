from typing import Any
import torch
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import os
import torch.nn as nn


def train(
    model: torch.nn.Module,
    train_set,
    optimizer: torch.optim.Optimizer,
    loss_model,
    num_epochs,
    writer: SummaryWriter=None,
    validation_interval=1,
    validation_set=None,
    validation_model=None,
    mode="max",
    checkpoint_path="log/",
):
    checkpoint_path = os.path.join(checkpoint_path, "best_model.pth")

    val_best = -torch.inf if mode == "max" else torch.inf
    n_batch = len(train_set)

    for epoch in range(num_epochs):
        model.train()
        loss_total = 0.0
        for batch in train_set:
            optimizer.zero_grad()
            loss = loss_model(model, batch)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        loss_total /= n_batch

        writer.add_scalar("train_loss", loss_total, epoch)
        
        with torch.no_grad():
            if validation_set is not None:
                if epoch % validation_interval == 0:
                    model.eval()
                    val_score = validation_model(model, validation_set)
                    writer.add_scalar("val_score", val_score, epoch)

                    checkpoint = {
                        "epoch_idx": epoch,
                        "net": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }

                    if mode == "max":
                        if val_score > val_best:
                            val_best = val_score
                            torch.save(checkpoint, checkpoint_path)
                    elif mode == "min":
                        if val_score < val_best:
                            val_best = val_score
                            torch.save(checkpoint, checkpoint_path)


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/23?page=2
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_activation(activation: str):
    activation = activation.lower()

    if activation == "relu":
        return torch.nn.ReLU
    elif activation == "sigmoid":
        return torch.nn.Sigmoid
    elif activation == "tanh":
        return torch.nn.Tanh
    elif activation == "lrelu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"{activation} not supported\n")


class CELossModel():
    def __init__(self, batched=True):
        self.l = torch.nn.functional.cross_entropy
        self.batched = batched
    
    def __call__(self, model, input_output) -> Any:
        if self.batched:
            return self.l(model(input_output[0]), input_output[1])
        else:
            loss = 0.0
            n = len(input_output)
            for in_out in input_output:
                loss += self.l(model(in_out[0]), in_out[1])
            return loss / n


def compute_accuracy(model, dataset):
    preds = model(dataset[0])
    accuracy = torch.mean((torch.argmax(preds, dim=-1) == dataset[1]).float())
    return accuracy


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )