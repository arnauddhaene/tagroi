from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import transforms

from .custom_transforms import SimulateTags
from .utils import Scan, Patient


class ACDCDataset(TensorDataset):
    """Pytorch Dataset wrapper for the ACDC dataset. Available on https://acdc.creatis.insa-lyon.fr/."""
    
    def __init__(
        self,
        path: str = None, recompute: bool = False, tagged: bool = True, name: str = '',
        verbose: int = 0
    ):
        """Constructor

        Args:
            path (str, optional): Path to unzipped downloaded dataset. Defaults to None.
            recompute (bool, optional): Recompute dataset from source files instead of
                fetching from pickled file in repository. Defaults to False.
            tagged (bool, optional): transform to tagged images. Defaults to True.
            name (str, optional): name to be added to saved file. Defaults to ''.
            verbose (int, optional): Print out information. Defaults to 0.

        Raises:
            ValueError: raised if asked to recompute without giving location of raw dataset.
        """
        repo_path = Path(__file__).parent.parent.parent
        (repo_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.tagged: bool = tagged
        filename = 'acdc_dataset_' + ('tagged' if tagged else 'cine') + '.pt'
        self.location: Path = (repo_path / 'checkpoints') / filename
        
        if path is None and recompute:
            raise ValueError('Missing path.')
        
        # Load from pickled file if available
        if not recompute and self.location.is_file():
            self.tensors = torch.load(self.location)
            if verbose > 0:
                print(f'Loaded saved dataset of {len(self)} images from {self.location}')

        else:
            # Get all patient folders from main raw downloaded ACDC directory
            patient_paths = [ppath for ppath in Path(path).iterdir() if ppath.is_dir()]
            
            images: torch.Tensor = torch.Tensor()
            labels: torch.Tensor = torch.Tensor()

            skip_label: int = 0
            skip_nan: int = 0

            accepted_classes: set = set([0., 1., 2., 3.])

            # Iterate over all patients
            patients_pbar = tqdm(patient_paths, leave=True)
            for ppath in patients_pbar:
                if verbose > 0:
                    patients_pbar.set_description(f'Processing {ppath.name}...')
                
                # Loading .nii.gz files in handled in the `Patient` class
                patient = Patient(ppath)
                assert len(patient.images) == len(patient.masks)
                
                # Loop through each patient's list of images (around 10 per patient)
                image_pbar = tqdm(zip(patient.images, patient.masks), leave=False, total=len(patient.images))
                for image, label in image_pbar:
                    if verbose > 0:
                        image_pbar.set_description(f'Processing image of size {image.shape}...')
                    
                    image, label = image.astype(np.float64), label.astype(np.float64)

                    # Preprocess
                    # mu, sigma = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
                    image = image / image.max()  # To [0, 1] range
                    image = self._preprocess_image(0.456, 0.224)(image).unsqueeze(0)
                    
                    label = self._preprocess_label()(label)

                    # Exclude NaNs from dataset
                    if image.isnan().sum().item() > 0 or label.isnan().sum().item() > 0:
                        skip_nan += 1
                        continue

                    # Throw out inconsistent masks
                    _classes = label.unique().numpy()
                    if len(_classes) > 4 or not set(_classes).issubset(accepted_classes):
                        skip_label += 1
                        continue
                            
                    images = torch.cat((images, image), axis=0)
                    labels = torch.cat((labels, label), axis=0)
            
            self.tensors = (images, labels,)

            if verbose > 0:
                print(f'Skipped {skip_label} image(s) due to incoherent label')
                print(f'Skipped {skip_nan} image(s) due to presence of NaN')

            self._save(verbose)

    def _save(self, verbose: int = 0):
        # Save pickled tensors to reload them quicker on next use
        torch.save(self.tensors, self.location)
        if verbose > 0:
            print(f'Saved dataset of {len(self)} images to {self.location}')

    def _preprocess_image(self, mu: float, sigma: float) -> transforms.Compose:
        """Preprocess image

        Args:
            mu (float): average for normalization layer
            sigma (float): standard deviation for normalization layer

        Returns:
            transforms.Compose: transformation callback function
        """
        if self.tagged:
            return transforms.Compose([
                SimulateTags(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=sigma),
                transforms.Resize((256, 256))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=sigma),
                transforms.Resize((256, 256))
            ])
        
    def _preprocess_label(self) -> transforms.Compose:
        """Preprocess mask

        Returns:
            transforms.Compose: transformation callback function
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    
class DMDDataset(Dataset):
    
    def __init__(self):
        
        HEALTHY_DIR, DMD_DIR = Path('../../dmd_roi/healthy'), Path('../../dmd_roi/dmd')
        
        self.images = np.empty((1, 3, 256, 256))
        self.labels = np.empty((1, 1, 256, 256))
        
        for directory in [HEALTHY_DIR, DMD_DIR]:
            # Iterate over all scans for each folder
            scans = [Scan(directory / patient) for patient in directory.iterdir() if directory.is_dir()]
            
            for scan in scans:
                for slic in scan.slices.values():
                    if slic.is_annotated():
                        
                        image = slic.image[0]
                        image = image.astype(np.float32)
                        # Preprocess
                        mu, sigma = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
                        image = self._preprocess_image(mu, sigma)(image).unsqueeze(0)
                        
                        # Repeat channels by 3 to be used in RGB segmentation model
                        image = image.repeat(1, 3, 1, 1)
                        
                        label = (slic.mask['outer'] ^ slic.mask['inner'])
                        label = label.astype(np.float32)
                        label = self._preprocess_label()(label).unsqueeze(0)
                                                                        
                        self.images = np.append(self.images, image, axis=0)
                        self.labels = np.append(self.labels, label, axis=0)
                        
        self.images = torch.from_numpy(self.images).float()
        self.labels = torch.from_numpy(self.labels).float()
        
    def _preprocess_image(self, mu: float, sigma: float) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mu, std=sigma),
            transforms.Resize((256, 256))
        ])
        
    def _preprocess_label(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        
    def __len__(self) -> int:
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
