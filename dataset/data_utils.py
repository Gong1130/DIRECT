import math

import cv2
import numpy as np


def mask_score(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    return np.max(contour_areas) / sum(contour_areas)


def expand_image_mask(image, mask, ratio=1.4, random=False, image_pad_value=0):
    h, w = image.shape[:2]
    H, W = int(h * ratio), int(w * ratio)
    if random:
        h1 = np.random.randint(0, int(H - h))
        w1 = np.random.randint(0, int(W - w))
    else:
        h1 = int((H - h) // 2)
        w1 = int((W - w) // 2)
    h2 = H - h - h1
    w2 = W - w - w1

    image_pad = ((h1, h2), (w1, w2), (0, 0))
    mask_pad = ((h1, h2), (w1, w2))
    image = np.pad(image, image_pad, "constant", constant_values=image_pad_value)
    mask = np.pad(mask, mask_pad, "constant", constant_values=0)
    return image, mask


def get_bbox_from_mask(mask):
    h, w = mask.shape[:2]

    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return y1, y2, x1, x2


def rotate_image_and_mask(image, mask, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    radians = math.radians(angle)
    sin_a = math.sin(radians)
    cos_a = math.cos(radians)
    new_w = int((h * abs(sin_a)) + (w * abs(cos_a)))
    new_h = int((h * abs(cos_a)) + (w * abs(sin_a)))

    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    transform[0, 2] += (new_w / 2) - center[0]
    transform[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(
        image,
        transform,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    mask_255 = mask * 255
    rotated_mask_255 = cv2.warpAffine(
        mask_255,
        transform,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    rotated_mask = (rotated_mask_255 > 127).astype(np.uint8)

    return rotated_image, rotated_mask


def find_largest_inner_rectangle(mask):
    H, W = mask.shape
    mask_bin = (mask > 0).astype(np.uint8)

    height = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            if mask_bin[i, j] == 0:
                height[i, j] = 0
            else:
                height[i, j] = height[i - 1, j] + 1 if i > 0 else 1

    max_area = 0
    best_box = None
    for i in range(H):
        stack = []
        j = 0
        while j <= W:
            h = height[i, j] if j < W else 0
            if not stack or h >= height[i, stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                w = j if not stack else j - stack[-1] - 1
                area = height[i, top] * w
                if area > max_area:
                    max_area = area
                    h_ = height[i, top]
                    y2 = i + 1
                    y1 = y2 - h_
                    x2 = j
                    x1 = x2 - w
                    best_box = (y1, y2, x1, x2)
    return best_box


def expand_bbox(mask, yyxx, ratio=(1.2, 2.0), min_crop=0, min_expand=0):
    y1, y2, x1, x2 = yyxx
    sampled_ratio = np.random.randint(int(ratio[0] * 10), int(ratio[1] * 10)) / 10
    H, W = mask.shape[:2]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = sampled_ratio * (y2 - y1 + 1)
    w = sampled_ratio * (x2 - x1 + 1)
    h = max(h, min_crop, min_expand + y2 - y1 + 1)
    w = max(w, min_crop, min_expand + x2 - x1 + 1)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    return y1, y2, x1, x2


def expand_bbox_asymmetrical(
    mask,
    yyxx,
    h_ratio_range=(0.1, 0.8),
    w_ratio_range=(0.1, 0.8),
    min_pad_px=20,
):
    y1, y2, x1, x2 = yyxx
    H, W = mask.shape[:2]

    orig_h = y2 - y1 + 1
    orig_w = x2 - x1 + 1

    pad_top = max(int(orig_h * np.random.uniform(*h_ratio_range)), min_pad_px)
    pad_bottom = max(int(orig_h * np.random.uniform(*h_ratio_range)), min_pad_px)
    pad_left = max(int(orig_w * np.random.uniform(*w_ratio_range)), min_pad_px)
    pad_right = max(int(orig_w * np.random.uniform(*w_ratio_range)), min_pad_px)

    new_y1 = max(0, y1 - pad_top)
    new_y2 = min(H, y2 + pad_bottom)
    new_x1 = max(0, x1 - pad_left)
    new_x2 = min(W, x2 + pad_right)

    return new_y1, new_y2, new_x1, new_x2


def expand_square_box(bbox, H, W):
    y1, y2, x1, x2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width > height:
        diff = width - height
        sy1 = max(y1 - diff // 2, 0)
        sy2 = min(y2 + diff // 2, H)
        return sy1, sy2, x1, x2

    diff = height - width
    sx1 = max(x1 - diff // 2, 0)
    sx2 = min(x2 + diff // 2, W)
    return y1, y2, sx1, sx2


def pad_to_square(image, pad_value=255, random=False):
    H, W = image.shape[:2]
    if H == W:
        return image

    pad_total = abs(H - W)
    if random:
        pad_1 = int(np.random.randint(0, pad_total))
    else:
        pad_1 = pad_total // 2
    pad_2 = pad_total - pad_1

    if image.ndim == 2:
        if H > W:
            pad_width = ((0, 0), (pad_1, pad_2))
        else:
            pad_width = ((pad_1, pad_2), (0, 0))
    elif image.ndim == 3:
        if H > W:
            pad_width = ((0, 0), (pad_1, pad_2), (0, 0))
        else:
            pad_width = ((pad_1, pad_2), (0, 0), (0, 0))
    else:
        raise ValueError(f"Unsupported image rank: {image.ndim}")

    return np.pad(image, pad_width, mode="constant", constant_values=pad_value)
