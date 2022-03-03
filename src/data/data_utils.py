from pathlib import Path

import pydicom

import numpy as np
from skimage.draw import polygon, polygon2mask

from src.metrics_acdc import load_nii


class Patient():
    
    def __init__(self, filepath: str):
        
        files = [f for f in Path(filepath).iterdir() if f.suffixes == ['.nii', '.gz']]
        
        for f in files:
            if '_4d' in str(f) or '_gt' in str(f):
                continue
            
            f_gt = self._gt_path(f)
            
            if f_gt in files:
                self.images, self.masks = self.fetch_frames(f, f_gt)
                      
    @staticmethod
    def _gt_path(filepath: str):
        return filepath.parent / (filepath.stem.split('.')[0] + '_gt.nii.gz')
            
    @staticmethod
    def fetch_frames(image_path: str, mask_path: str):
        
        imt, _, _ = load_nii(image_path)
        gt, _, _ = load_nii(mask_path)
        
        return imt.swapaxes(0, 2), gt.swapaxes(0, 2)
        

class Slice:
    
    def __init__(self, path: Path):
        
        self.slice = path.name
        
        dcm_images = [f for f in path.iterdir() if f.is_file() and f.suffix == '.dcm']
        if len(dcm_images) > 0:
            res = map(lambda ds: (ds.InstanceNumber, ds.pixel_array), map(pydicom.dcmread, dcm_images))
            self.image = np.array(list(zip(*sorted(res, key=lambda item: item[0])))[1])
        
        # Extract ROI if it exists
        roi_path = (path / 'roi_pts.npz')
        if roi_path.is_file():
            rois = np.load(roi_path)
            self.roi = {array_key: rois[array_key] for array_key in list(rois.keys())}
            self.mask = {}
            for _roi in ['pts_interp_outer', 'pts_interp_inner']:
                if self.roi[_roi] is not None:
                    # Verify dimensions
                    assert self.roi[_roi].shape[0] == 2
                    
                    pg = polygon(self.roi[_roi][1, :], self.roi[_roi][0, :], self.image.shape[1:])
                    # Save under 'outer' or 'inner' key
                    self.mask[_roi.split('_')[-1]] = polygon2mask(self.image.shape[1:], np.array(pg).T)
                else:
                    self.mask[_roi.split('_')[-1]] = None
            
        else:
            self.roi = None
            
    def is_annotated(self) -> bool:
        return self.roi is not None
            
    def __repr__(self) -> str:
        return self.__str__()
            
    def __str__(self) -> str:
        return f"{self.slice} slice with{'out' if self.roi is None else ''} annotated ROI."
        

class Scan:
    
    def __init__(self, path: Path, label: bool = False):
        
        self.id = path.name
        self.label = label
        self.slices = {s.name: Slice(s) for s in (path / 'clean').iterdir() if s.is_dir()}
