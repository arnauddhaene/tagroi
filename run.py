# import os

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

import aim
import click

from src.models.unet import UNet
from src.training.train import train
from src.data.datasets import ACDCDataset


# os.environ["NCCL_DEBUG"] = "INFO"

@click.command()
@click.option('--tagged', default=True, help="Apply simulated tagging transformation.")
@click.option('--batch-size', default=32, help="Batch size.")
@click.option('--lr', default=1e-2, help="Learning rate.")
@click.option('--decay', default=1e-3, help="Optimizer weight decay.")
@click.option('--momentum', default=0.9, help="Optimizer momentum.")
@click.option('--epochs', default=30, help="Number of training epochs.")
@click.option('--experiment-name', default='', help="Assign run to experiment.")
@click.option('--model-name', default='model', help="Filename for pickled model.")
@click.option('--verbose', default=1, type=int, help="Print out info for debugging purposes.")
def run(
    tagged, batch_size, lr, decay, momentum, epochs,
    experiment_name, model_name, verbose
):
    dataset = ACDCDataset(path='../training/', tagged=tagged, verbose=verbose)

    train_set, val_set = random_split(dataset, [704, 247], generator=torch.Generator().manual_seed(42))
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=1, n_classes=4, bilinear=True).double()
    # Load old saved version of the model as a state dictionary
    saved_model_sd = torch.load('checkpoints/model/model_cine_tag.pt')
    # Extract UNet if saved model is parallelized
    model.load_state_dict(saved_model_sd)

    if device.type == 'cuda':
        model = nn.DataParallel(model)
        model.n_classes = model.module.n_classes

    run = aim.Run(experiment=experiment_name)

    train(model, run=run, loader_train=loader_train, loader_val=loader_val,
          filename=model_name, device=device,
          learning_rate=lr, weight_decay=decay, momentum=momentum,
          epochs=epochs, verbose=verbose)


if __name__ == '__main__':
    run()
