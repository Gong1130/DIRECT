from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import os
import random
from glob import glob

import cv2
import numpy as np

from .data_utils import (
    expand_bbox,
    expand_bbox_asymmetrical,
    expand_image_mask,
    expand_square_box,
    find_largest_inner_rectangle,
    get_bbox_from_mask,
    mask_score,
    pad_to_square,
    rotate_image_and_mask,
)
from trellis.modules.sparse.basic import SparseTensor, sparse_cat

class BasePairedDataset(Dataset, ABC):
    """
    Base class for fixed-resolution paired training datasets.
    """
    def __init__(self, data_limit=None, shuffle_data=False, **kwargs):
        super().__init__()

        self.size = kwargs.get('resolution', 512)        
        
        self.crop_strategy = kwargs.get('crop_strategy', None)
        if self.crop_strategy == 'mask_ratio':
            self.min_ratio = kwargs.get('min_ratio', 0.1)
            self.max_ratio = kwargs.get('max_ratio', 0.5)

        self.constraint_inpainting_area = kwargs.get('constraint_inpainting_area', False)
        self.asymmetrical_constraint = kwargs.get('asymmetrical_constraint', False)

        self.rotate_ref = kwargs.get('rotate_ref', False)

        self.mask_template_paths = self._load_mask_templates(kwargs.get('mask_template_path'))

        self._apply_data_limit(data_limit, shuffle_data)

    def _load_mask_templates(self, mask_template_path):
        if not mask_template_path:
            return []
        if isinstance(mask_template_path, str):
            mask_template_path = [mask_template_path]
        all_templates = []
        for path in mask_template_path:
            all_templates.extend(
                glob(os.path.join(path, "**/*.png"), recursive=True)
            )
            all_templates.extend(
                glob(os.path.join(path, "**/*.jpg"), recursive=True)
            )
        all_templates.sort()        
        return all_templates

    def _apply_data_limit(self, data_limit, shuffle_data):
        if data_limit is not None and data_limit < len(self.data):
            if shuffle_data:
                random.shuffle(self.data)
                print(f"Dataset limit: Shuffled and sampled {data_limit} objects.")
            else:
                print(f"Dataset limit: Truncated to {data_limit} objects.")
            self.data = self.data[:data_limit]
        print(f"Dataset {self.__class__.__name__} final size: {len(self.data)} objects.")

    @abstractmethod
    def _build_data_list(self, **kwargs):
        """
        Subclasses must populate self.data with sample metadata.
        """
        pass

    @abstractmethod
    def _load_raw_data(self, idx):
        """
        Subclasses must load the raw sample fields required by get_sample.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.get_sample(idx)
                break
            except Exception as e:
                print(e)
                idx = np.random.randint(0, len(self.data))
                continue
        return sample
    
    def get_sample(self, idx):
        """
        Load a raw sample and apply common image/mask processing.
        """
        item = self._load_raw_data(idx)

        assert mask_score(item['ref_mask']) > 0.90
        (
            item['masked_ref_image'], 
            item['ref_image'],
            item['ref_mask']
        ) = self.mask_ref_image(item['ref_image'], item['ref_mask'])

        (
            item['target_image'],
            item['masked_target_image'], 
            item['mask_out_target_image'], 
            item['inpainting_mask'], 
            item['object_mask'], 
            item['full_masked_target_image']
        ) = self.process_target_image(item['tar_image'], item['tar_mask'])

        return item

    '''
        item = {
            "masked_ref_image": masked_ref_image, # 512, 512, 3, [-1, 1]
            "ref_mask": ref_mask, # 512, 512, 1 [0,1]
            "target_image": target_image, # 512, 512, 3, [-1, 1]
            "masked_target_image": masked_target_image, # 512, 512, 3, [-1, 1]
            "inpainting_mask": inpainting_mask, # 512, 512, 1 [0,1]
            "object_mask": object_mask, # 512, 512, 1 [0,1]
            "target_w2c": target_w2c, # 4, 4
            "target_slat": target_slat, # 1, 8
            "target_normal_image": tar_normal_image, # 512, 512, 3, [0, 1]
            "mask_out_target_image": mask_out_target_image, # 512, 512, 3, [-1, 1]
            "full_masked_target_image": full_masked_target_image, # H, W, 3, [0, 255]
            "ref_image": ref_image,
            "ref_image_source_path": raw_data['ref_image_source_path'],
            "tar_image_source_path": raw_data['tar_image_source_path']
        }
    '''        

    def get_mask(self, mask_path):
        image = cv2.imread( mask_path, cv2.IMREAD_UNCHANGED)

        if len(image.shape) == 2:  # H x W
            mask = (image > 128).astype(np.uint8)
            return mask
        elif image.shape[2] == 4:
            mask = (image[:,:,-1] > 128).astype(np.uint8)
            return mask
        else:
            raise ValueError(f"Unsupported mask format: shape={image.shape}")     
    
    def mask_ref_image(self, ref_image, ref_mask):
        # Get bbox from mask
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        y1, y2, x1, x2 = ref_box_yyxx

        # Crop image and mask
        cropped_image = ref_image[y1:y2, x1:x2, :]
        cropped_mask = ref_mask[y1:y2, x1:x2]

        if self.rotate_ref:
            rotation_range = 90
            angle = random.uniform(-rotation_range, rotation_range)            
            rotated_image, rotated_mask = rotate_image_and_mask(cropped_image, cropped_mask, angle)
            rot_box = get_bbox_from_mask(rotated_mask)
            ry1, ry2, rx1, rx2 = rot_box
            cropped_image = rotated_image[ry1:ry2, rx1:rx2, :]
            cropped_mask = rotated_mask[ry1:ry2, rx1:rx2]

        # Expand image and mask
        ratio = 1.2
        expanded_image, expanded_mask = expand_image_mask(cropped_image, cropped_mask, ratio=ratio)

        # Make square
        padded_image = pad_to_square(expanded_image, pad_value=0, random=False)
        padded_mask = pad_to_square(expanded_mask.astype(np.uint8) * 255, pad_value=0, random=False)

        # Resize to 512x512
        resized_image = cv2.resize(padded_image.astype(np.uint8), (self.size, self.size))
        resized_mask = cv2.resize(padded_mask.astype(np.uint8), (self.size, self.size))
        resized_mask = (resized_mask > 127).astype(np.uint8)  # Ensure binary mask

        # Generate masked image
        mask_3ch = np.stack([resized_mask] * 3, axis=-1)
        masked_image = resized_image * mask_3ch

        # Normalize both images to [-1, 1]
        ref_image = (resized_image.astype(np.float32) / 127.5) - 1.0
        masked_ref_image = (masked_image.astype(np.float32) / 127.5) - 1.0
        ref_mask = resized_mask[:, :, None]

        return masked_ref_image, ref_image, ref_mask

    def process_target_image(self, tar_image, tar_mask):
        img_h, img_w = tar_image.shape[:2]

        if self.crop_strategy == 'mask_ratio':
            crop_tar_image, crop_tar_mask, (oy, ox, cs) = self.crop_with_mask_ratio(tar_image, tar_mask, min_ratio=self.min_ratio, max_ratio=self.max_ratio) # 512, 512, 3, [0, 255]; 512, 512, [0,1]
        elif self.crop_strategy == 'square_short_edge':
            crop_tar_image, crop_tar_mask, (oy, ox, cs) = self.crop_square_short_edge(tar_image, tar_mask)
        else:
            crop_tar_image, crop_tar_mask, (oy, ox, cs) = self.random_crop(tar_image, tar_mask) # 512, 512, 3, [0, 255]; 512, 512, [0,1]

        target_h, target_w = crop_tar_image.shape[:2]
        valid_mask_crop_res = np.zeros((cs, cs), dtype=np.uint8)
        v_top = max(0, -oy)
        v_left = max(0, -ox)
        v_bottom = min(cs, img_h - oy)
        v_right = min(cs, img_w - ox)
        if v_bottom > v_top and v_right > v_left:
            valid_mask_crop_res[v_top:v_bottom, v_left:v_right] = 1
        valid_mask = cv2.resize(valid_mask_crop_res, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        valid_mask = valid_mask[:, :, None]

        y1, y2, x1, x2 = get_bbox_from_mask(crop_tar_mask)

        template_path = random.choice(self.mask_template_paths)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_bin = self.preprocess_template_mask(template)
        y1, y2, x1, x2 = expand_bbox(crop_tar_mask, (y1, y2, x1, x2), ratio=[1.0, 1.1], min_expand=10)
        inpainting_mask = self.warp_template_to_bbox(template_bin, crop_tar_mask, (y1, y2, x1, x2), use_inner_box=True) # 512, 512 [0,1]

        target_image = (crop_tar_image.astype(np.float32) / 127.5) - 1.0 # [-1, 1]
        crop_tar_mask = crop_tar_mask[:, :, None]
        mask_out_target_image = crop_tar_image * crop_tar_mask
        mask_out_target_image = (mask_out_target_image.astype(np.float32) / 127.5) - 1.0 # [-1, 1]

        inpainting_mask = inpainting_mask[:, :, None] # 512, 512, 1
        inpainting_mask = inpainting_mask * valid_mask
        masked_target_image = crop_tar_image * (1 - inpainting_mask) 

        local_masked_full_size = cv2.resize(masked_target_image, (cs, cs), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        full_masked_image = tar_image.copy()

        t_start, t_end = max(0, oy), min(img_h, oy + cs)
        l_start, l_end = max(0, ox), min(img_w, ox + cs)

        s_t_start = max(0, -oy)
        s_l_start = max(0, -ox)
        h_len = t_end - t_start
        w_len = l_end - l_start

        if h_len > 0 and w_len > 0:
            full_masked_image[t_start:t_end, l_start:l_end] = local_masked_full_size[s_t_start:s_t_start+h_len, s_l_start:s_l_start+w_len]
        full_masked_target_image = full_masked_image
        
        masked_target_image = (masked_target_image.astype(np.float32) / 127.5) - 1.0 # [-1, 1]

        return target_image, masked_target_image, mask_out_target_image, inpainting_mask, crop_tar_mask, full_masked_target_image

    def preprocess_template_mask(self, template_gray, long_dim=256, short_dim=256):
        """
        Binarize a template mask, keep its largest component, pad it, and
        resize it to the requested long/short side dimensions.
        """
        mask = (template_gray > 128).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return mask 
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        x, y, w, h = stats[largest_label][:4]
        
        # Padding
        pad = 20 
        H, W = mask.shape
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        clean_mask = (labels == largest_label).astype(np.uint8)
        cropped_mask = clean_mask[y1:y2, x1:x2]
        
        h_c, w_c = cropped_mask.shape
        if max(h_c, w_c) > 0:
            if h_c >= w_c:
                new_h = long_dim
                new_w = short_dim
            else:
                new_h = short_dim
                new_w = long_dim
            
            resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = cropped_mask

        return resized_mask

    def crop_with_mask_ratio(
        self,
        image,
        mask,
        min_ratio=0.3,
        max_ratio=0.6,
        _pad_offset=(0, 0)
    ):
        H, W = mask.shape

        y1, y2, x1, x2 = get_bbox_from_mask(mask)

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        min_crop_size = int(max(bbox_h, bbox_w) / max_ratio)
        max_crop_size = int(max(bbox_h, bbox_w) / min_ratio)
        min_crop_size = max(min(min_crop_size, min(H, W)), max(bbox_h, bbox_w))
        max_crop_size = min(max_crop_size, min(H, W))

        if min_crop_size > max_crop_size:
            pad_top = pad_bottom = 0
            pad_left = pad_right = 0
            
            if H < min_crop_size:
                pad_total = min_crop_size - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top

            if W < min_crop_size:
                pad_total = min_crop_size - W
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

            pad_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            pad_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            new_offset = (_pad_offset[0] + pad_top, _pad_offset[1] + pad_left)
            
            return self.crop_with_mask_ratio(
                pad_img, pad_mask,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                _pad_offset=new_offset
            )

        crop_size = np.random.randint(min_crop_size, max_crop_size + 1)

        # Valid crop origin range to include bbox
        y_min = max(0, y2 - crop_size)
        y_max = min(y1, H - crop_size)
        x_min = max(0, x2 - crop_size)
        x_max = min(x1, W - crop_size)

        crop_y1 = np.random.randint(y_min, y_max + 1)
        crop_x1 = np.random.randint(x_min, x_max + 1)
        crop_y2 = crop_y1 + crop_size
        crop_x2 = crop_x1 + crop_size

        orig_y1 = crop_y1 - _pad_offset[0]
        orig_x1 = crop_x1 - _pad_offset[1]

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize
        resized_img = cv2.resize(crop_img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask, (orig_y1, orig_x1, crop_size)

    def random_crop(
        self,
        image,
        mask,
        _pad_offset=(0, 0)
    ):
        H, W = mask.shape
        y1, y2, x1, x2 = get_bbox_from_mask(mask)

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        # Clamp to valid image bounds
        min_crop_size = max(bbox_h, bbox_w) + 10
        max_crop_size = min(H, W)

        if min_crop_size > max_crop_size:            
            pad_top = pad_bottom = 0
            pad_left = pad_right = 0

            if H < min_crop_size:
                pad_total = min_crop_size - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top


            if W < min_crop_size:
                pad_total = min_crop_size - W
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

            pad_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            pad_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            new_offset = (_pad_offset[0] + pad_top, _pad_offset[1] + pad_left)

            return self.random_crop(
                pad_img, pad_mask,
                _pad_offset=new_offset
            )

        crop_size = np.random.randint(min_crop_size, max_crop_size + 1)

        y_min = max(0, y2 - crop_size)
        y_max = min(y1, H - crop_size)
        x_min = max(0, x2 - crop_size)
        x_max = min(x1, W - crop_size)

        crop_y1 = np.random.randint(y_min, y_max + 1)
        crop_x1 = np.random.randint(x_min, x_max + 1)
        crop_y2 = crop_y1 + crop_size
        crop_x2 = crop_x1 + crop_size

        orig_y1 = crop_y1 - _pad_offset[0]
        orig_x1 = crop_x1 - _pad_offset[1]

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize
        resized_img = cv2.resize(crop_img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask, (orig_y1, orig_x1, crop_size)

    def crop_square_short_edge(
        self,
        image,
        mask,
        _pad_offset=(0, 0)
    ):
        H, W = mask.shape
        y1, y2, x1, x2 = get_bbox_from_mask(mask)

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        base_size = min(H, W)
        
        max_bbox_dim = max(bbox_h, bbox_w) + 10
        crop_size = max(base_size, max_bbox_dim)

        if H < crop_size or W < crop_size:
            pad_top = pad_bottom = 0
            pad_left = pad_right = 0

            if H < crop_size:
                pad_total = crop_size - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top

            if W < crop_size:
                pad_total = crop_size - W
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

            pad_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            pad_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            new_offset = (_pad_offset[0] + pad_top, _pad_offset[1] + pad_left)

            return self.crop_square_short_edge(
                pad_img, pad_mask,
                _pad_offset=new_offset
            )

        y_min = max(0, y2 - crop_size)
        y_max = min(y1, H - crop_size)
        x_min = max(0, x2 - crop_size)
        x_max = min(x1, W - crop_size)

        crop_y1 = np.random.randint(y_min, y_max + 1)
        crop_x1 = np.random.randint(x_min, x_max + 1)
        crop_y2 = crop_y1 + crop_size
        crop_x2 = crop_x1 + crop_size

        orig_y1 = crop_y1 - _pad_offset[0]
        orig_x1 = crop_x1 - _pad_offset[1]

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        resized_img = cv2.resize(crop_img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask, (orig_y1, orig_x1, crop_size)
    
    def warp_template_to_bbox(self, template_mask, target_mask, bbox, use_inner_box):
        H, W = target_mask.shape
        y1, y2, x1, x2 = bbox
        target_box = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        if use_inner_box:
            sy1, sy2, sx1, sx2 = find_largest_inner_rectangle(template_mask)
        else:
            sy1, sy2, sx1, sx2 = get_bbox_from_mask(template_mask)
            sy1, sy2, sx1, sx2 = expand_bbox(template_mask, (sy1, sy2, sx1, sx2), ratio=[1.0, 1.1], min_expand=20)
        
        source_box = np.array([
            [sx1, sy1],
            [sx2, sy1],
            [sx2, sy2],
            [sx1, sy2]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(source_box, target_box)
        warped_template = cv2.warpPerspective(template_mask, M, (W, H), flags=cv2.INTER_LINEAR)
        warped_binary = (warped_template > 0).astype(np.uint8)
        if self.constraint_inpainting_area:
            if self.asymmetrical_constraint:
                ey1, ey2, ex1, ex2 = expand_bbox_asymmetrical(target_mask, (y1, y2, x1, x2), h_ratio_range=[0, 0.3],
                                                          w_ratio_range=[0, 0.3], min_pad_px=20)
            else:
                ey1, ey2, ex1, ex2 = expand_bbox(target_mask, (y1, y2, x1, x2), ratio=[1.2, 1.3])
                ey1, ey2, ex1, ex2 = expand_square_box((ey1, ey2, ex1, ex2), H, W)
            bbox_mask = np.zeros_like(target_mask, dtype=np.uint8)
            bbox_mask[ey1:ey2, ex1:ex2] = 1
            warped_binary = warped_binary * bbox_mask
        return warped_binary

class MultiResPairedDataset(Dataset, ABC):
    """
    Base class for multi-resolution paired training datasets.
    """
    def __init__(self, data_limit=None, shuffle_data=False, **kwargs):
        super().__init__()
        self.resolutions = [
            (672, 1568),
            (688, 1504),
            (720, 1456),
            (752, 1392),
            (800, 1328),
            (832, 1248),
            (880, 1184),
            (944, 1104),
            (1024, 1024),
            (1104, 944),
            (1184, 880),
            (1248, 832),
            (1328, 800),
            (1392, 752),
            (1456, 720),
            (1504, 688),
            (1568, 672),
        ]
        self.crop_strategy = kwargs.get('crop_strategy', None)
        if self.crop_strategy == 'mask_ratio':
            self.min_ratio = kwargs.get('min_ratio', 0.1)
            self.max_ratio = kwargs.get('max_ratio', 0.5)

        self.constraint_inpainting_area = kwargs.get('constraint_inpainting_area', False)
        self.asymmetrical_constraint = kwargs.get('asymmetrical_constraint', False)

        self.rotate_ref = kwargs.get('rotate_ref', False)

        self.mask_template_paths = self._load_mask_templates(kwargs.get('mask_template_path'))

        self._apply_data_limit(data_limit, shuffle_data)

    def _load_mask_templates(self, mask_template_path):
        if not mask_template_path:
            return []
        if isinstance(mask_template_path, str):
            mask_template_path = [mask_template_path]
        all_templates = []
        for path in mask_template_path:
            all_templates.extend(
                glob(os.path.join(path, "**/*.png"), recursive=True)
            )
            all_templates.extend(
                glob(os.path.join(path, "**/*.jpg"), recursive=True)
            )
        all_templates.sort()        
        return all_templates

    def _apply_data_limit(self, data_limit, shuffle_data):
        if data_limit is not None and data_limit < len(self.data):
            if shuffle_data:
                random.shuffle(self.data)
                print(f"Dataset limit: Shuffled and sampled {data_limit} objects.")
            else:
                print(f"Dataset limit: Truncated to {data_limit} objects.")
            self.data = self.data[:data_limit]
        print(f"Dataset {self.__class__.__name__} final size: {len(self.data)} objects.")

    @abstractmethod
    def _build_data_list(self, **kwargs):
        """
        Subclasses must populate self.data with sample metadata.
        """
        pass

    @abstractmethod
    def _load_raw_data(self, idx):
        """
        Subclasses must load the raw sample fields required by get_sample.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.get_sample(idx)
                break
            except Exception as e:
                print(e)
                idx = np.random.randint(0, len(self.data))
                continue
        return sample
    
    def get_sample(self, idx):
        """
        Load a raw sample and apply common image/mask processing.
        """
        item = self._load_raw_data(idx)

        assert mask_score(item['ref_mask']) > 0.90
        (
            item['target_image'],
            item['masked_target_image'], 
            item['mask_out_target_image'], 
            item['inpainting_mask'], 
            item['object_mask'], 
            item['full_masked_target_image']
        ) = self.process_target_image(item['tar_image'], item['tar_mask'])
        size = item['target_image'].shape[:2]
        (
            item['masked_ref_image'], 
            item['ref_image'],
            item['ref_mask']
        ) = self.mask_ref_image(item['ref_image'], item['ref_mask'], size)
        
        return item

    def get_mask(self, mask_path):
        image = cv2.imread( mask_path, cv2.IMREAD_UNCHANGED)

        if len(image.shape) == 2:  # H x W
            mask = (image > 128).astype(np.uint8)
            return mask
        elif image.shape[2] == 4:
            mask = (image[:,:,-1] > 128).astype(np.uint8)
            return mask
        else:
            raise ValueError(f"Unsupported mask format: shape={image.shape}")     
    
    def mask_ref_image(self, ref_image, ref_mask, size):
        # Get bbox from mask
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        y1, y2, x1, x2 = ref_box_yyxx

        # Crop image and mask
        cropped_image = ref_image[y1:y2, x1:x2, :]
        cropped_mask = ref_mask[y1:y2, x1:x2]

        if self.rotate_ref:
            rotation_range = 90
            angle = random.uniform(-rotation_range, rotation_range)            
            rotated_image, rotated_mask = rotate_image_and_mask(cropped_image, cropped_mask, angle)
            rot_box = get_bbox_from_mask(rotated_mask)
            ry1, ry2, rx1, rx2 = rot_box
            cropped_image = rotated_image[ry1:ry2, rx1:rx2, :]
            cropped_mask = rotated_mask[ry1:ry2, rx1:rx2]

        # Expand image and mask
        ratio = 1.2
        expanded_image, expanded_mask = expand_image_mask(cropped_image, cropped_mask, ratio=ratio)

        target_H, target_W = size
        target_aspect = target_W / target_H
        h, w = expanded_image.shape[:2]
        curr_aspect = w / h

        pad_top = pad_bottom = pad_left = pad_right = 0
        if abs(curr_aspect - target_aspect) > 1e-6:
            if curr_aspect < target_aspect:
                desired_w = int(np.ceil(h * target_aspect))
                pad_w = max(0, desired_w - w)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
            else:
                desired_h = int(np.ceil(w / target_aspect))
                pad_h = max(0, desired_h - h)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

        if any((pad_top, pad_bottom, pad_left, pad_right)):
            padded_image = cv2.copyMakeBorder(
                expanded_image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
            padded_mask = cv2.copyMakeBorder(
                expanded_mask.astype(np.uint8), pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
        else:
            padded_image = expanded_image
            padded_mask = expanded_mask

        resized_image = cv2.resize(padded_image.astype(np.uint8), (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(padded_mask.astype(np.uint8), (target_W, target_H), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 0).astype(np.uint8)

        mask_3ch = np.stack([resized_mask] * 3, axis=-1)
        masked_image = resized_image * mask_3ch

        ref_image_out = (resized_image.astype(np.float32) / 127.5) - 1.0
        masked_ref_image_out = (masked_image.astype(np.float32) / 127.5) - 1.0
        ref_mask_out = resized_mask[:, :, None] # (H, W, 1)

        return masked_ref_image_out, ref_image_out, ref_mask_out

    def process_target_image(self, tar_image, tar_mask):
        img_h, img_w = tar_image.shape[:2]

        if self.crop_strategy == 'mask_ratio':
            crop_tar_image, crop_tar_mask, (oy, ox, ch, cw) = self.crop_with_mask_ratio(tar_image, tar_mask, min_ratio=self.min_ratio, max_ratio=self.max_ratio) # 512, 512, 3, [0, 255]; 512, 512, [0,1]
        elif self.crop_strategy == 'crop_full_image':
            crop_tar_image, crop_tar_mask, (oy, ox, ch, cw) = self.crop_full_image(tar_image, tar_mask)
        else:
            crop_tar_image, crop_tar_mask, (oy, ox, ch, cw) = self.random_crop(tar_image, tar_mask) # 512, 512, 3, [0, 255]; 512, 512, [0,1]
        
        target_h, target_w = crop_tar_image.shape[:2]
        valid_mask_crop_res = np.zeros((ch, cw), dtype=np.uint8)
        v_top = max(0, -oy)
        v_left = max(0, -ox)
        v_bottom = min(ch, img_h - oy)
        v_right = min(cw, img_w - ox)
        if v_bottom > v_top and v_right > v_left:
            valid_mask_crop_res[v_top:v_bottom, v_left:v_right] = 1
        valid_mask = cv2.resize(valid_mask_crop_res, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        valid_mask = valid_mask[:, :, None]

        y1, y2, x1, x2 = get_bbox_from_mask(crop_tar_mask)

        template_path = random.choice(self.mask_template_paths)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_bin = self.preprocess_template_mask(template)
        y1, y2, x1, x2 = expand_bbox(crop_tar_mask, (y1, y2, x1, x2), ratio=[1.0, 1.1], min_expand=10)
        inpainting_mask = self.warp_template_to_bbox(template_bin, crop_tar_mask, (y1, y2, x1, x2), use_inner_box=True) # 512, 512 [0,1]

        target_image = (crop_tar_image.astype(np.float32) / 127.5) - 1.0 # [-1, 1]
        crop_tar_mask = crop_tar_mask[:, :, None]
        mask_out_target_image = crop_tar_image * crop_tar_mask
        mask_out_target_image = (mask_out_target_image.astype(np.float32) / 127.5) - 1.0 # [-1, 1]

        inpainting_mask = inpainting_mask[:, :, None] # 512, 512, 1
        inpainting_mask = inpainting_mask * valid_mask
        masked_target_image = crop_tar_image * (1 - inpainting_mask) 

        # for full_masked_target_image
        local_masked_full_size = cv2.resize(masked_target_image, (cw, ch), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        full_masked_image = tar_image.copy()

        t_start, t_end = max(0, oy), min(img_h, oy + ch)
        l_start, l_end = max(0, ox), min(img_w, ox + cw)

        s_t_start = max(0, -oy)
        s_l_start = max(0, -ox)
        h_len = t_end - t_start
        w_len = l_end - l_start

        if h_len > 0 and w_len > 0:
            full_masked_image[t_start:t_end, l_start:l_end] = local_masked_full_size[s_t_start:s_t_start+h_len, s_l_start:s_l_start+w_len]
        full_masked_target_image = full_masked_image

        masked_target_image = (masked_target_image.astype(np.float32) / 127.5) - 1.0 # [-1, 1]

        return target_image, masked_target_image, mask_out_target_image, inpainting_mask, crop_tar_mask, full_masked_target_image

    def preprocess_template_mask(self, template_gray, long_dim=256, short_dim=256):
        """
        Binarize a template mask, keep its largest component, pad it, and
        resize it to the requested long/short side dimensions.
        """
        mask = (template_gray > 128).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return mask 
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        x, y, w, h = stats[largest_label][:4]
        
        # Padding
        pad = 20 
        H, W = mask.shape
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        clean_mask = (labels == largest_label).astype(np.uint8)
        cropped_mask = clean_mask[y1:y2, x1:x2]
        
        h_c, w_c = cropped_mask.shape
        if max(h_c, w_c) > 0:
            if h_c >= w_c:
                new_h = long_dim
                new_w = short_dim
            else:
                new_h = short_dim
                new_w = long_dim
            
            resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = cropped_mask

        return resized_mask

    def crop_with_mask_ratio(
        self,
        image,
        mask,
        min_ratio=0.3,
        max_ratio=0.6,
        _pad_offset=(0, 0),
        target_res=None
    ):
        H, W = mask.shape
        if target_res is None:
            orig_aspect = W / H
            target_res = min(self.resolutions, key=lambda res: abs(orig_aspect - res[0]/res[1]))
        y1, y2, x1, x2 = get_bbox_from_mask(mask)
        y1, y2, x1, x2 = expand_bbox(mask, (y1, y2, x1, x2), ratio=[1.0, 1.1], min_expand=10)
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        # Calculate crop size range based on desired mask ratio
        target_width, target_height = target_res
        target_aspect = target_width / target_height
        # crop_h = min(int(round(W / target_aspect)), H)
        # crop_w = min(int(round(H * target_aspect)), W)

        bbox_aspect = bbox_w / bbox_h
        if target_aspect > bbox_aspect:
            min_crop_h = min(int(bbox_h / max_ratio), H)
            max_crop_h = min(int(bbox_h / min_ratio), H, int(np.ceil(W / target_aspect)))
            min_crop_w = int(np.ceil(min_crop_h * target_aspect))
            max_crop_w = int(np.ceil(max_crop_h * target_aspect))
        else:
            min_crop_w = min(int(bbox_w / max_ratio), W)
            max_crop_w = min(int(bbox_w / min_ratio), W, int(H * target_aspect))
            min_crop_h = int(np.ceil(min_crop_w / target_aspect))
            max_crop_h = int(np.ceil(max_crop_w / target_aspect))

        if min_crop_h > max_crop_h or min_crop_w > max_crop_w:
            pad_top = pad_bottom = 0
            pad_left = pad_right = 0

            if H < min_crop_h:
                pad_total = min_crop_h - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top

            if W < min_crop_w:
                pad_total = min_crop_w - W
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

            pad_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            pad_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            new_offset = (_pad_offset[0] + pad_top, _pad_offset[1] + pad_left)
           
            return self.crop_with_mask_ratio(
                pad_img, pad_mask,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                _pad_offset=new_offset,
                target_res=target_res
            )

        if target_aspect > bbox_aspect:
            crop_h = np.random.randint(min_crop_h, max_crop_h + 1)
            crop_w = int(round(crop_h * target_aspect))
        else:
            crop_w = np.random.randint(min_crop_w, max_crop_w + 1)
            crop_h = int(round(crop_w / target_aspect))

        y_min = max(0, y2 - crop_h)
        y_max = min(y1, H - crop_h)
        x_min = max(0, x2 - crop_w)
        x_max = min(x1, W - crop_w)

        crop_y1 = np.random.randint(y_min, y_max + 1)
        crop_x1 = np.random.randint(x_min, x_max + 1)
        crop_y2 = crop_y1 + crop_h
        crop_x2 = crop_x1 + crop_w

        orig_y1 = crop_y1 - _pad_offset[0]
        orig_x1 = crop_x1 - _pad_offset[1]

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize
        resized_img = cv2.resize(crop_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        return resized_img, resized_mask, (orig_y1, orig_x1, crop_h, crop_w)

    def random_crop(
        self,
        image,
        mask,
        _pad_offset=(0, 0)
    ):
        H, W = mask.shape
        y1, y2, x1, x2 = get_bbox_from_mask(mask)

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        # Clamp to valid image bounds
        orig_aspect = W / H
        # target_res: (width, height)
        target_res = min(self.resolutions, key=lambda res: abs(orig_aspect - res[0]/res[1]))
        target_width, target_height = target_res
        target_aspect = target_width / target_height

        min_crop_h = max(bbox_h + 10, int(np.ceil((bbox_w + 10) / target_aspect)))
        min_crop_w = int(round(min_crop_h * target_aspect))
        max_crop_h = min(H, int(W / target_aspect))

        if min_crop_h > max_crop_h:            
            pad_top = pad_bottom = 0
            pad_left = pad_right = 0

            if H < min_crop_h:
                pad_total = min_crop_h - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top

            if W < min_crop_w:
                pad_total = min_crop_w - W
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

            pad_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            pad_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            new_offset = (_pad_offset[0] + pad_top, _pad_offset[1] + pad_left)

            return self.random_crop(
                pad_img, pad_mask,
                _pad_offset=new_offset
            )

        crop_h = np.random.randint(min_crop_h, max_crop_h + 1)
        crop_w = int(round(crop_h * target_aspect))

        y_min = max(0, y2 - crop_h)
        y_max = min(y1, H - crop_h)
        x_min = max(0, x2 - crop_w)
        x_max = min(x1, W - crop_w)

        crop_y1 = np.random.randint(y_min, y_max + 1)
        crop_x1 = np.random.randint(x_min, x_max + 1)
        crop_y2 = crop_y1 + crop_h
        crop_x2 = crop_x1 + crop_w

        orig_y1 = crop_y1 - _pad_offset[0]
        orig_x1 = crop_x1 - _pad_offset[1]

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize
        resized_img = cv2.resize(crop_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask, (orig_y1, orig_x1, crop_h, crop_w)
   
    def crop_full_image(
        self,
        image,
        mask,
        _pad_offset=(0, 0),
        target_res=None
    ):
        H, W = mask.shape[:2]
        if target_res is None:
            orig_aspect = W / H
            target_res = min(self.resolutions, key=lambda res: abs(orig_aspect - res[0]/res[1]))
       
        y1, y2, x1, x2 = get_bbox_from_mask(mask)
        y1, y2, x1, x2 = expand_bbox(mask, (y1, y2, x1, x2), ratio=[1.0, 1.1], min_expand=10)
        
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        
        target_width, target_height = target_res
        target_aspect = target_width / target_height
        
        min_crop_h = min(bbox_h + 10, H)
        min_crop_w = min(bbox_w + 10, W)

        crop_h = min(int(round(W / target_aspect)), H)
        crop_w = min(int(round(H * target_aspect)), W)

        if min_crop_h > crop_h or min_crop_w > crop_w:
            pad_top = pad_bottom = 0
            pad_left = pad_right = 0
            if min_crop_h > crop_h:
                pad_total = int(np.ceil(min_crop_h * target_aspect)) - W
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
            if min_crop_w > crop_w:
                pad_total = int(np.ceil(min_crop_w / target_aspect)) - H
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top

            pad_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            pad_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            new_offset = (_pad_offset[0] + pad_top, _pad_offset[1] + pad_left)

            return self.crop_full_image(
                pad_img, pad_mask,
                _pad_offset=new_offset,
                target_res=target_res
            )
        
        y_min = max(0, y2 - crop_h)
        y_max = min(y1, H - crop_h)
        x_min = max(0, x2 - crop_w)
        x_max = min(x1, W - crop_w)

        crop_y1 = np.random.randint(y_min, y_max + 1)
        crop_x1 = np.random.randint(x_min, x_max + 1)
        crop_y2 = crop_y1 + crop_h
        crop_x2 = crop_x1 + crop_w
        
        orig_y1 = crop_y1 - _pad_offset[0]
        orig_x1 = crop_x1 - _pad_offset[1]

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        resized_img = cv2.resize(crop_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask, (orig_y1, orig_x1, crop_h, crop_w)

    def warp_template_to_bbox(self, template_mask, target_mask, bbox, use_inner_box):
        H, W = target_mask.shape
        y1, y2, x1, x2 = bbox
        target_box = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        if use_inner_box:
            sy1, sy2, sx1, sx2 = find_largest_inner_rectangle(template_mask)
        else:
            sy1, sy2, sx1, sx2 = get_bbox_from_mask(template_mask)
            sy1, sy2, sx1, sx2 = expand_bbox(template_mask, (sy1, sy2, sx1, sx2), ratio=[1.0, 1.1], min_expand=20)
        
        source_box = np.array([
            [sx1, sy1],
            [sx2, sy1],
            [sx2, sy2],
            [sx1, sy2]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(source_box, target_box)
        warped_template = cv2.warpPerspective(template_mask, M, (W, H), flags=cv2.INTER_LINEAR)
        warped_binary = (warped_template > 0).astype(np.uint8)
        if self.constraint_inpainting_area:
            if self.asymmetrical_constraint:
                ey1, ey2, ex1, ex2 = expand_bbox_asymmetrical(target_mask, (y1, y2, x1, x2), h_ratio_range=[0, 0.3],
                                                          w_ratio_range=[0, 0.3], min_pad_px=20)
            else:
                ey1, ey2, ex1, ex2 = expand_bbox(target_mask, (y1, y2, x1, x2), ratio=[1.2, 1.3])
                ey1, ey2, ex1, ex2 = expand_square_box((ey1, ey2, ex1, ex2), H, W)
            bbox_mask = np.zeros_like(target_mask, dtype=np.uint8)
            bbox_mask[ey1:ey2, ex1:ex2] = 1
            warped_binary = warped_binary * bbox_mask
        return warped_binary


from torch.utils.data._utils.collate import default_collate
from collections.abc import Mapping


def direct_collate_fn(batch):
    """
    Collate SparseTensor fields with sparse_cat and defer other fields to
    PyTorch's default collate behavior.
    """
    elem = batch[0]

    if isinstance(elem, SparseTensor):
        return sparse_cat(batch)

    if isinstance(elem, Mapping):
        return {
            key: direct_collate_fn([d[key] for d in batch])
            for key in elem
        }

    try:
        return default_collate(batch)
    except Exception:
        return batch
  
