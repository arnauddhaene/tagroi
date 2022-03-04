from tqdm import tqdm
from typing import Dict
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def dice(prediction: torch.Tensor, target: torch.Tensor, exclude_bg: bool = False) -> Dict[int, torch.Tensor]:
    """Dice coefficient for multi-class segmentation set-up.

    Args:
        prediction (torch.Tensor): prediction Tensor of size (batch, n_classes, w, h)
        target (torch.Tensor): target index mask Tensor of size (batch, w, h)
        exclude_bg (bool, optional): Exclude bg class (idx=0). Defaults to True.

    Returns:
        Dict: Dice coefficient for each class. Use data.utils.INDEX_TO_CLASS for conversion.
    """
    
    n_classes = prediction.shape[1]
    # target is one hot encoded to be - batch_size, n_classes, width, height
    target = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2).bool()
    # prediction needs to adhere to multi-class segmentation
    # this means that each pixel should have only one class
    prediction = F.one_hot(prediction.argmax(dim=1), n_classes).permute(0, 3, 1, 2).bool()
    
    # Let's check ability to compare after shaping them correctly
    assert prediction.size() == target.size()
    
    _dice = torch.zeros(n_classes)

    # Calculate Dice coefficient for classes
    # Background is class 0, so either [0, 1, ...] or [1, ...]
    bg_offset = 1 if exclude_bg else 0
    for _class in range(bg_offset, n_classes):
        _dice[_class] = dc(prediction[:, _class, ...], target[:, _class, ...])
        
    return _dice
        
    
def dice_loss(prediction: torch.Tensor, target: torch.Tensor, exclude_bg: bool = False) -> torch.Tensor:
    """Loss based on Dice coefficient. Objective function to minimize.

    Args:
        prediction (torch.Tensor): prediction Tensor of size (batch, n_classes, w, h)
        target (torch.Tensor): target index mask Tensor of size (batch, w, h)
        exclude_bg (bool, optional): Exclude bg class (idx=0). Defaults to True.

    Returns:
        torch.Tensor: Dice Loss. Value between 0. and 1.
    """
    offset = 1 if exclude_bg else 0
    return 1. - dice(prediction, target)[offset:].mean()


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device = 'cpu') -> torch.Tensor:
    """Evaluate segmentation model on DataLoader with Dice coefficient`

    Args:
        model (nn.Module): model
        dataloader (DataLoader): validation or testing dataloader
        device (torch.device): device to evaluate on

    Returns:
        torch.Tensor: average dice score over all batches
    """
    model.eval()
    
    # Aggregate per batch Dice coefficient in master dictionary
    dice_score = torch.zeros(model.n_classes)
    
    n_batches = len(dataloader)
    assert n_batches > 0

    # iterate over the validation set
    with torch.no_grad():
        for images, targets in tqdm(dataloader, total=n_batches, unit='batch', leave=False,
                                    desc='Iterating through validation batches...'):
            images, targets = images.to(device), targets.to(device)
            # predict the mask
            output = model(images)
            dice_score += dice(output, targets)

    model.train()

    return dice_score / n_batches


def counter_mean(counter: Counter, denominator: float) -> Dict[int, float]:
    return {k: v / denominator for k, v in dict(counter).items()}
    

def dc(result: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    This is a modified version of medpy.metric.binary.dc that works with PyTorch
    ---
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    Args:
        result (torch.Tensor): Input data containing objects. Can be any type but will be
            converted into binary: background where 0, object everywhere else.
        reference (torch.Tensor): Input data containing objects. Can be any type but will be
            converted into binary: background where 0, object everywhere else.

    Returns:
        torch.Tensor: The Dice coefficient between the inputs.
            Ranges from 0. (no overlap) to 1. (perfect overlap).
    """
    result = result.bool()
    reference = reference.bool()
    
    intersection = torch.count_nonzero(result & reference)
    
    size_i1 = torch.count_nonzero(result)
    size_i2 = torch.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc
