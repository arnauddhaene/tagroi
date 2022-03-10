from pathlib import Path
from tqdm import tqdm

import kornia.augmentation as K

import torch
from torch import nn
from torch.utils.data import DataLoader

import aim

from .metrics import dice_score, DiceLoss, evaluate
from ..data.utils import INDEX_TO_CLASS


def train(
    model: nn.Module, run: aim.Run, loader_train: DataLoader, loader_val: DataLoader,
    filename: str = 'model', device: torch.device = 'cpu',
    learning_rate: float = 1e-2, weight_decay: float = 1e-3, momentum: float = 0.9,
    epochs: int = 50, verbose: int = 0
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
        momentum (float, optional): momentum for optimizer
        epochs (int, optional): number of epochs. Defaults to 25.
        device (torch.device, optional): cuda device. Defaults to cpu.
        verbose (int, optional): print info. Defaults to 0.
    """
    REPO_PATH = Path(__file__).parent.parent.parent
    (REPO_PATH / 'checkpoints' / 'model').mkdir(parents=True, exist_ok=True)

    amp = True
    model = model.to(device)

    proba: float = .2

    train_aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=proba),
        K.RandomVerticalFlip(p=proba),
        K.RandomElasticTransform(p=proba),
        K.RandomGaussianNoise(p=proba),
        K.RandomSharpness(p=proba),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(.1, .1), p=proba),
        data_keys=['input', 'mask']
    )
    
    # Define loss
    criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss(exclude_bg=True)

    def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return criterion(outputs, targets) + dice_criterion(outputs, targets)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                    momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    run['hparams'] = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': loader_train.batch_size,
        'momentum': momentum,
        'optimizer': optimizer.__class__.__name__,
        'main_criterion': criterion.__class__.__name__,
        'secondary_criterion': dice_criterion.__class__.__name__,
        'device': device.type,
        'augmentation': str(train_aug)
    }

    if verbose > 0:
        print(f'Launching training of {model.__class__.__name__} for {epochs} epochs')
    
    pbar = tqdm(range(epochs), unit='epoch', leave=False)
    for epoch in pbar:
        
        dice = torch.zeros(4).to(device)
        acc_loss = 0.

        model.train()

        batch_pbar = tqdm(loader_train, total=len(loader_train), unit='batch', leave=False)
        for inputs, targets in batch_pbar:

            batch_pbar.set_description(f'Acummulated loss: {acc_loss:.4f}')
            # move to device
            # target is index of classes
            inputs, targets = inputs.double().to(device), targets.to(device)

            # Run augmentation pipeline every batch
            inputs, targets = train_aug(inputs, targets.unsqueeze(1))
            targets = targets.squeeze(1).long()
            
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            dice += dice_score(outputs, targets)
            acc_loss += loss.item()

        acc_loss /= len(loader_train)

        # Tracking training performance
        run.track(acc_loss, name='loss', epoch=epoch, context=dict(subset='train'))

        train_perf = dice / len(loader_train)
        avg_dice = train_perf.mean()

        for i, val in enumerate(train_perf):
            run.track(val, name=f'dice_{INDEX_TO_CLASS[i]}', epoch=epoch, context=dict(subset='train'))

        status = f'Epoch {epoch:03} \t Loss {acc_loss:.4f} \t Dice {avg_dice:.4f}'
        
        # Tracking validation performance
        val_perf, val_loss = evaluate(model, loader_val, loss_fn, track_images=((epoch + 1) % 5 == 0),
                                      run=run, device=device)
        run.track(val_loss, name='loss', epoch=epoch, context=dict(subset='val'))
        avg_val_dice = val_perf.mean()
        scheduler.step(avg_val_dice)

        for i, val in enumerate(val_perf):
            run.track(val, name=f'dice_{INDEX_TO_CLASS[i]}', epoch=epoch, context=dict(subset='val'))

        status += f'\t Val. Dice {avg_val_dice:.4f}'

        pbar.set_description(status)
        
    torch.save(model, str(REPO_PATH / 'checkpoints' / 'model' / f'{filename}.pt'))
