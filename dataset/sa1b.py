import os
import csv
import numpy as np
from glob import glob
import cv2
import torch

from .basedataset import BasePairedDataset, MultiResPairedDataset
from trellis.modules.sparse.basic import SparseTensor


class SA1BDataset_Trellis(BasePairedDataset):
    """
    SA-1B paired dataset for DIRECT stage-1 training.
    """
    def __init__(self, dataset_root, data_limit=None, shuffle_data=False, **kwargs):
        self.dataset_root = dataset_root
        self._build_data_list(metadata_dir=os.path.join(dataset_root, "metadata"))
        
        super().__init__(
            data_limit=data_limit, 
            shuffle_data=shuffle_data,
            **kwargs
        )

    def _build_data_list(self, metadata_dir):
        """
        Read filtered CSV files and keep valid SA-1B training samples.
        """
        data_files = sorted(glob(os.path.join(metadata_dir, "*.csv")))

        all_pairs = []
        
        for csv_path in data_files:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    object_dir = row["object_dir"]
                    image_name = row["image_name"]
                    meta = {"image_name": image_name}
                    object_name_placeholder = f"SA1B_{image_name}"
                    
                    all_pairs.append((object_name_placeholder, object_dir, [meta]))

        self.data = all_pairs

    def _load_raw_data(self, idx):
        """
        Build sample paths and load raw sample data.
        """
        _, object_dir, metas = self.data[idx]
        meta = metas[0]
        
        image_name = meta["image_name"]

        parts = object_dir.strip("/").split("/")

        split = parts[-1]

        tar_image_path = os.path.join(self.dataset_root, "images", split, image_name + '.jpg')
        tar_mask_path = os.path.join(self.dataset_root, "masks", split, image_name + '.jpg.png')
        
        ref_image_path = os.path.join(self.dataset_root, "edited_images", split, image_name + '.png')
        ref_mask_path = os.path.join(self.dataset_root, "edited_masks", split, image_name + '.png')
        target_trellis_dir = os.path.join(self.dataset_root, "trellis", split, image_name)
      
        ref_image = cv2.cvtColor(cv2.imread(ref_image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ref_mask = self.get_mask(ref_mask_path)
        tar_image = cv2.cvtColor(cv2.imread(tar_image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        tar_mask = self.get_mask(tar_mask_path)
        
        try:
            target_w2c = torch.load(os.path.join(target_trellis_dir, "optimized_w2c.pt"), weights_only=True)
            target_slat_data = torch.load(os.path.join(target_trellis_dir, "slat.pt"), weights_only=True)
        except TypeError:
            target_w2c = torch.load(os.path.join(target_trellis_dir, "optimized_w2c.pt"))
            target_slat_data = torch.load(os.path.join(target_trellis_dir, "slat.pt"))
            
        target_slat = SparseTensor(
            feats=target_slat_data['feats'],
            coords=target_slat_data['coords']
        )
  
        # Normal Image
        target_normal_image_path = os.path.join(target_trellis_dir, "normal.png")
        tar_normal_image = cv2.cvtColor(cv2.imread(target_normal_image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        tar_normal_image = tar_normal_image.astype(np.float32) / 255 # RGB, [0, 1]
        
        return {
            "ref_image": ref_image,
            "ref_mask": ref_mask,
            "tar_image": tar_image,
            "tar_mask": tar_mask,
            "target_w2c": target_w2c,
            "target_slat": target_slat,
            "tar_normal_image": tar_normal_image,
            "ref_image_source_path": ref_image_path,
            "tar_image_source_path": tar_image_path
        }
    
    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag
    
    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        if ratio > 0.8 * 0.8  or ratio < 0.1 * 0.1:
            return False
        else:
            return True 

class SA1BDataset_Trellis_MultiRes(MultiResPairedDataset):
    """
    SA-1B paired dataset for DIRECT multi-resolution training.
    """
    def __init__(self, dataset_root, data_limit=None, shuffle_data=False, **kwargs):
        self.dataset_root = dataset_root
        self._build_data_list(metadata_dir=os.path.join(dataset_root, "metadata"))
        
        super().__init__(
            data_limit=data_limit, 
            shuffle_data=shuffle_data,
            **kwargs
        )

    def _build_data_list(self, metadata_dir):
        """
        Read filtered CSV files and keep valid SA-1B training samples.
        """
        data_files = sorted(glob(os.path.join(metadata_dir, "*.csv")))

        all_pairs = []
        
        for csv_path in data_files:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    object_dir = row["object_dir"]
                    image_name = row["image_name"]
                    meta = {"image_name": image_name}
                    object_name_placeholder = f"SA1B_{image_name}"
                    
                    all_pairs.append((object_name_placeholder, object_dir, [meta]))

        self.data = all_pairs

    def _load_raw_data(self, idx):
        """
        Build sample paths and load raw sample data.
        """
        _, object_dir, metas = self.data[idx]
        meta = metas[0]
        
        image_name = meta["image_name"]

        parts = object_dir.strip("/").split("/")

        split = parts[-1]

        tar_image_path = os.path.join(self.dataset_root, "images", split, image_name + '.jpg')
        tar_mask_path = os.path.join(self.dataset_root, "masks", split, image_name + '.jpg.png')
        
        ref_image_path = os.path.join(self.dataset_root, "edited_images", split, image_name + '.png')
        ref_mask_path = os.path.join(self.dataset_root, "edited_masks", split, image_name + '.png')
        target_trellis_dir = os.path.join(self.dataset_root, "trellis", split, image_name)
      
        ref_image = cv2.cvtColor(cv2.imread(ref_image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ref_mask = self.get_mask(ref_mask_path)
        tar_image = cv2.cvtColor(cv2.imread(tar_image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        tar_mask = self.get_mask(tar_mask_path)
        
        try:
            target_w2c = torch.load(os.path.join(target_trellis_dir, "optimized_w2c.pt"), weights_only=True)
            target_slat_data = torch.load(os.path.join(target_trellis_dir, "slat.pt"), weights_only=True)
        except TypeError:
            target_w2c = torch.load(os.path.join(target_trellis_dir, "optimized_w2c.pt"))
            target_slat_data = torch.load(os.path.join(target_trellis_dir, "slat.pt"))
            
        target_slat = SparseTensor(
            feats=target_slat_data['feats'],
            coords=target_slat_data['coords']
        )
  
        # Normal Image
        target_normal_image_path = os.path.join(target_trellis_dir, "normal.png")
        tar_normal_image = cv2.cvtColor(cv2.imread(target_normal_image_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        tar_normal_image = tar_normal_image.astype(np.float32) / 255 # RGB, [0, 1]
        
        return {
            "ref_image": ref_image,
            "ref_mask": ref_mask,
            "tar_image": tar_image,
            "tar_mask": tar_mask,
            "target_w2c": target_w2c,
            "target_slat": target_slat,
            "tar_normal_image": tar_normal_image,
            "ref_image_source_path": ref_image_path,
            "tar_image_source_path": tar_image_path
        }
    
    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag
    
    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        if ratio > 0.8 * 0.8  or ratio < 0.1 * 0.1:
            return False
        else:
            return True 
