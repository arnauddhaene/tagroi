import os

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

import aim

from src.models.unet import UNet
from src.training.train import train
from src.data.datasets import ACDCDataset


os.environ["NCCL_DEBUG"] = "INFO"

dataset = ACDCDataset(path='../training/', tagged=False, verbose=1)

train_set, val_set = random_split(dataset, [704, 248], generator=torch.Generator().manual_seed(42))
loader_train = DataLoader(train_set, batch_size=16, shuffle=True)
loader_val = DataLoader(val_set, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_channels=1, n_classes=4, bilinear=True).double()
# Load old saved version of the model
# saved_model = torch.load('checkpoints/model/model_v1.pt')
# if isinstance(saved_model, nn.DataParallel):  # Extract UNet if saved model is parallelized
#     saved_model = saved_model.module
# model.load_state_dict(saved_model.state_dict())

if device.type == 'cuda':
    model = nn.DataParallel(model).to(device)
    model.n_classes = model.module.n_classes

run = aim.Run()

train(model, run=run, loader_train=loader_train, loader_val=loader_val, device=device)
