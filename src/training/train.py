from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import aim

from .metrics import dice_loss, dice, evaluate
from ..data.utils import INDEX_TO_CLASS


def train(
    model: nn.Module, run: aim.Run, loader_train: DataLoader, loader_val: DataLoader,
    filename: str = 'model', device: torch.device = 'cpu',
    learning_rate: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 50,
    verbose: int = 0
) -> None:
    """
    Train model
    
    Args:
        model (nn.Module): model
        loader_train (DataLoader): data loader
        loader_val (DataLoader): validation data loader. Defaults to None.
        filename (str): location to store trained model
        learning_rate (float, optional): learning rate. Defaults to 1e-2.
        weight_decay (float, optional): weight decay for Adam. Defaults to 1e-3.
        epochs (int, optional): number of epochs. Defaults to 25.
        device (torch.device, optional): cuda device. Defaults to cpu.
        verbose (int, optional): print info. Defaults to 0.
    """
    REPO_PATH = Path(__file__).parent.parent.parent
    (REPO_PATH / 'checkpoints' / filename).mkdir(parents=True, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    run['hparams'] = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': loader_train.batch_size,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'device': device.type
    }

    dice_score = torch.zeros(4)

    if verbose > 0:
        print(f'Launching training of {model.__class__.__name__} for {epochs} epochs')
    
    pbar = tqdm(range(epochs), unit='epoch', leave=False)
    for epoch in pbar:
        
        acc_loss = 0.

        model.train()
        
        for inputs, targets in tqdm(loader_train, desc='Iterating through training batches...',
                                    total=len(loader_train), unit='batch', leave=False):
            # move to device
            # target is index of classes
            inputs, targets = inputs.to(device), targets.long().to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) + dice_loss(outputs, targets)
            
            dice_score += dice(outputs, targets)
            acc_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Tracking training performance
        run.track(float(acc_loss), name='loss', epoch=epoch)

        train_perf = dice_score / len(loader_train)
        avg_dice = train_perf.mean()

        for i, val in enumerate(train_perf):
            run.track(val, name=f'dice_{INDEX_TO_CLASS[i]}', epoch=epoch, context=dict(subset='train'))

        status = f'Epoch {epoch:03} \t Loss {acc_loss:.4f} \t Dice {avg_dice:.4f}'
        
        # Tracking validation performance
        val_perf = evaluate(model, loader_val, device)
        avg_val_dice = val_perf.mean()

        for i, val in enumerate(val_perf):
            run.track(val, name=f'dice_{INDEX_TO_CLASS[i]}', epoch=epoch, context=dict(subset='val'))

        status += f'\t Val. Dice {avg_val_dice:.4f}'

        pbar.set_description(status)
        
    torch.save(model, str(REPO_PATH / 'checkpoints' / filename / 'model.pt'))
