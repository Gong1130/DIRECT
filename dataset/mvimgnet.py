import os
import csv
import random
from collections import defaultdict
import numpy as np
from glob import glob
import cv2
import torch

from .data_utils import get_bbox_from_mask
from .basedataset import BasePairedDataset, MultiResPairedDataset
from trellis.modules.sparse.basic import SparseTensor


class MVImgNetDataset_Trellis(BasePairedDataset):
    """
    MVImgNet paired dataset for DIRECT stage-1 training.
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
        Read filtered CSV files and group frames by object.
        """
        data_files = sorted(glob(os.path.join(metadata_dir, "*.csv")))

        object_dict = defaultdict(lambda: {"dir": None, "metas": {}})
        
        for csv_path in data_files:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    object_dir = row["object_dir"]
                    image_name = row["image_name"]
                    object_name = os.path.basename(object_dir)

                    object_dict[object_dir]["dir"] = object_dir
                    object_dict[object_dir]["name"] = object_name
                    object_dict[object_dir]["metas"][image_name] = {"image_name": image_name}
        
        all_objects = []
        for object_dir, data in object_dict.items():
            object_name = data["name"]
            metas = list(data["metas"].values())
            if len(metas) >= 2:
                all_objects.append((object_name, object_dir, metas))
        
        self.data = all_objects

    def _load_raw_data(self, idx):
        """
        Build sample paths and load raw sample data.
        """
        object_name, object_dir, metas = self.data[idx] 
        
        ref_meta, tar_meta = random.sample(metas, 2)
        ref_image_name = ref_meta["image_name"]
        tar_image_name = tar_meta["image_name"]

        source_image_dir = os.path.join(self.dataset_root, "images")
        source_mask_dir = os.path.join(self.dataset_root, "rmbg_masks")
        trellis_dir = os.path.join(self.dataset_root, "trellis")
        
        parts = object_dir.strip("/").split("/")
        
        rel_object_dir = parts[-3:]
        ref_image_source_path = os.path.join(source_image_dir, *rel_object_dir, "images", ref_image_name + '.jpg')
        tar_image_source_path = os.path.join(source_image_dir, *rel_object_dir, "images", tar_image_name + '.jpg')
        ref_mask_path = os.path.join(source_mask_dir, *rel_object_dir, ref_image_name + '.jpg.png')
        tar_mask_path = os.path.join(source_mask_dir, *rel_object_dir, tar_image_name + '.jpg.png')
        target_trellis_dir = os.path.join(trellis_dir, *rel_object_dir, tar_image_name)
      
        ref_image = cv2.cvtColor(cv2.imread(ref_image_source_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ref_mask = self.get_mask(ref_mask_path)
        tar_image = cv2.cvtColor(cv2.imread(tar_image_source_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        tar_mask = self.get_mask(tar_mask_path)
        assert self.check_mask_area(ref_mask) == True
        assert self.check_mask_area(tar_mask)  == True

        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = 0.90, mode = 'max') == True

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
            "ref_image_source_path": ref_image_source_path,
            "tar_image_source_path": tar_image_source_path
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

class MVImgNetDataset_Trellis_MultiRes(MultiResPairedDataset):
    """
    MVImgNet paired dataset for DIRECT multi-resolution training.
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
        Read filtered CSV files and group frames by object.
        """
        data_files = sorted(glob(os.path.join(metadata_dir, "*.csv")))

        object_dict = defaultdict(lambda: {"dir": None, "metas": {}})
        
        for csv_path in data_files:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    object_dir = row["object_dir"]
                    image_name = row["image_name"]
                    object_name = os.path.basename(object_dir)

                    object_dict[object_dir]["dir"] = object_dir
                    object_dict[object_dir]["name"] = object_name
                    object_dict[object_dir]["metas"][image_name] = {"image_name": image_name}
        
        all_objects = []
        for object_dir, data in object_dict.items():
            object_name = data["name"]
            metas = list(data["metas"].values())
            if len(metas) >= 2:
                all_objects.append((object_name, object_dir, metas))
        
        self.data = all_objects

    def _load_raw_data(self, idx):
        """
        Build sample paths and load raw sample data.
        """
        object_name, object_dir, metas = self.data[idx] 
        
        ref_meta, tar_meta = random.sample(metas, 2)
        ref_image_name = ref_meta["image_name"]
        tar_image_name = tar_meta["image_name"]

        source_image_dir = os.path.join(self.dataset_root, "images")
        source_mask_dir = os.path.join(self.dataset_root, "rmbg_masks")
        trellis_dir = os.path.join(self.dataset_root, "trellis")
        
        parts = object_dir.strip("/").split("/")
        
        rel_object_dir = parts[-3:]
        ref_image_source_path = os.path.join(source_image_dir, *rel_object_dir, "images", ref_image_name + '.jpg')
        tar_image_source_path = os.path.join(source_image_dir, *rel_object_dir, "images", tar_image_name + '.jpg')
        ref_mask_path = os.path.join(source_mask_dir, *rel_object_dir, ref_image_name + '.jpg.png')
        tar_mask_path = os.path.join(source_mask_dir, *rel_object_dir, tar_image_name + '.jpg.png')
        target_trellis_dir = os.path.join(trellis_dir, *rel_object_dir, tar_image_name)
      
        ref_image = cv2.cvtColor(cv2.imread(ref_image_source_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ref_mask = self.get_mask(ref_mask_path)
        tar_image = cv2.cvtColor(cv2.imread(tar_image_source_path).astype(np.uint8), cv2.COLOR_BGR2RGB)
        tar_mask = self.get_mask(tar_mask_path)
        assert self.check_mask_area(ref_mask) == True
        assert self.check_mask_area(tar_mask)  == True

        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = 0.90, mode = 'max') == True

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
            "ref_image_source_path": ref_image_source_path,
            "tar_image_source_path": tar_image_source_path
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
