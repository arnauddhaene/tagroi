from typing import Callable
from tqdm import tqdm

from kornia.utils.one_hot import one_hot

import aim

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def dice_score(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    Based on https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
    -- split into dice score and dice loss to be able to track performance for each class

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed dice score of shape (C)

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = dice_loss(input, target)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. \
            Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. \
            Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. \
            Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1],
                                           device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    return torch.mean(2.0 * intersection / (cardinality + eps), dim=0)


def dice_loss(prediction: torch.Tensor, target: torch.Tensor, exclude_bg: bool = False) -> torch.Tensor:
    """Loss based on Dice coefficient. Objective function to minimize.

    Args:
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        labels: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        exclude_bg (bool, optional): Exclude background from loss term. Defaults to False.

    Returns:
        torch.Tensor: Dice Loss. Value between 0. and 1.
    """
    offset = 1 if exclude_bg else 0
    return torch.mean(-dice_score(prediction, target)[offset:] + 1.)


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        exclude_bg: Exclude background dice from loss term

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, exclude_bg: bool = False) -> None:
        super().__init__()
        self.exclude_bg: bool = exclude_bg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(input, target, self.exclude_bg)


def evaluate(model: nn.Module, dataloader: DataLoader,
             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             track_images: bool = False, run: aim.Run = None, epoch: int = -1,
             device: torch.device = 'cpu') -> torch.Tensor:
    """Evaluate segmentation model on DataLoader with Dice coefficient`

    Args:
        model (nn.Module): model
        dataloader (DataLoader): validation or testing dataloader
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): loss function
        track_images (bool, optional): track images to aim run. Defaults to False.
        run (aim.Run, optional): aim run to send tracked images to. Defaults to None.
        device (torch.device, optional): device to evaluate on. Defaults to cpu.

    Returns:
        torch.Tensor: average dice score over all batches
    """

    if track_images and not isinstance(run, aim.Run):
        raise ValueError(f'Must pass object of type aim.Run to eval function. Passed {type(run)}.')

    model.eval()

    # Aggregate per batch Dice coefficient in master dictionary
    dice = torch.zeros(model.n_classes).to(device)

    n_batches = len(dataloader)
    assert n_batches > 0

    val_loss = 0.

    # iterate over the batches in the dataloader
    with torch.no_grad():
        for images, targets in tqdm(dataloader, total=n_batches, unit='batch', leave=False,
                                    desc='Iterating through validation batches'):
            images, targets = images.double().to(device), targets.long().to(device)
            # predict the mask
            outputs = model(images)

            if track_images:
                masks = F.softmax(outputs, dim=1).argmax(dim=1).detach().cpu().numpy()
                factor = round(256 / model.n_classes)
                for mask, target in zip(masks, targets.detach().cpu().numpy()):
                    mask, target = mask.astype('uint8') * factor, target.astype('uint8') * factor
                    run.track(aim.Image(mask, caption='prediction'), name='images', epoch=epoch)
                    run.track(aim.Image(target, caption='target'), name='images', epoch=epoch)

            dice += dice_score(outputs, targets)
            val_loss += loss_fn(outputs, targets).item()

    model.train()

    return dice / n_batches, val_loss / len(dataloader)
