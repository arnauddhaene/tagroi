import torch

from .cine2gridtag import sim_gridtag, cine2gridtag


class SimulateTags(torch.nn.Module):
    """Simulates tagging.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    """

    def __init__(self, spacing: float = 5, contrast: float = 0.4, label = None, LV_index: int = 2):
        """
        Args:
            spacing (float, optional): spacing between tag lines, in pixels. Defaults to 5.
            contrast (float, optional): exponent to apply to the image reducing contrast. Defaults to 0.4.
            label (Tensor, ndarray, optional): mask of regions of interest 
            LV_index (int, optional): index in the mask of the LV
        """
        super().__init__()
        self.spacing = spacing
        self.contrast = contrast
        self.label = label
        self.LV_index = LV_index

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be tagged.

        Returns:
            PIL Image or Tensor: tagged image.
        """
        return cine2gridtag(img, self.label, self.LV_index, self.contrast, self.spacing)

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(spacing={self.spacing}, contrast={self.contrast})'
