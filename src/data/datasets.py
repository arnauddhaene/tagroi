from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .custom_transforms import SimulateTags
from .data_utils import Scan, Patient


class ACDCDataset(Dataset):
    
    def __init__(self, path: str, verbose: int = 0):
        
        patient_paths = [ppath for ppath in Path(path).iterdir() if ppath.is_dir()]
        
        self.images = torch.empty((1, 1, 256, 256))
        self.labels = torch.empty((1, 256, 256))

        patients_pbar = tqdm(patient_paths, leave=False)
        for ppath in patients_pbar:
            if verbose > 0:
                patients_pbar.set_description(f'Processing patient {ppath.name}...')
            
            patient = Patient(ppath)
            
            assert len(patient.images) == len(patient.masks)
            
            image_pbar = tqdm(zip(patient.images, patient.masks), leave=False)
            for image, label in image_pbar:
                if verbose > 0:
                    image_pbar.set_description(f'Processing image of size {image.shape}...')
                
                image, label = image.astype(np.float32), label.astype(np.float32)
            
                # Preprocess
                mu, sigma = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
                image = self._preprocess_image(mu, sigma)(image).unsqueeze(0)
                
                label = self._preprocess_label()(label)
                        
                self.images = torch.cat((self.images, image), axis=0)
                self.labels = torch.cat((self.labels, label), axis=0)

    def _preprocess_image(self, mu: float, sigma: float) -> transforms.Compose:
        return transforms.Compose([
            SimulateTags(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mu, std=sigma),
            transforms.Resize((256, 256))
        ])
        
    def _preprocess_label(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        ])
        
    def __len__(self) -> int:
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]

    
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
    