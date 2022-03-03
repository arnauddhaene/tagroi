from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import dice_loss


def train(
    model: nn.Module, train_loader: DataLoader, filename: str,
    learning_rate: float = 1e-2, weight_decay: float = 1e-3, epochs: int = 25,
    verbose: int = 0
) -> None:
    """
    Train model
    
    Args:
        model (nn.Module): model
        train_loader (DataLoader): data loader
        filename (str): location to store trained model
        learning_rate (float, optional): learning rate. Defaults to 1e-2.
        weight_decay (float, optional): weight decay for Adam. Defaults to 1e-3.
        epochs (int, optional): number of epochs. Defaults to 25.
        verbose (int, optional): print info. Defaults to 0.
    """
    REPO_PATH = Path(__file__).parent.parent.parent
    checkpoints = (REPO_PATH / 'checkpoints' / filename).mkdir(parents=True, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        
        acc_loss = 0.

        model.train()
        
        for _input, target in train_loader:
            optimizer.zero_grad()
            target = target.long()  # As target is index of classes
            
            output = model(_input)
            loss = criterion(output, target) \
                + dice_loss(output, F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).double(),
                            multiclass=True)
            
            loss.backward()
            optimizer.step()
            
            acc_loss += loss.item()
            
    torch.save(model, str(checkpoints / 'model.pt'))
