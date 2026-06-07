import copy
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import utils3d
import viser.transforms as tf
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


@dataclass
class GenerationInputs:
    reference_image: Image.Image
    composite_image: Image.Image
    inpaint_mask: Image.Image
    geometry_image: Image.Image
    context_image: Image.Image
    model_input_mask: Image.Image
    background_reference_image: Image.Image
    ideal_bbox: tuple
    target_side: int


def random_seed():
    return int(np.random.randint(0, 2**31 - 1))


def extract_gaussian_params(gaussian_obj):
    with torch.no_grad():
        centers_np = gaussian_obj.get_xyz.detach().cpu().numpy()
        opacities_np = gaussian_obj.get_opacity.detach().cpu().numpy()
        colors_dc_np = gaussian_obj._features_dc.detach().cpu().numpy().squeeze(axis=1)
        rgbs_np = np.clip(0.5 + 0.28209479177387814 * colors_dc_np, 0.0, 1.0)
        scales_np = gaussian_obj.get_scaling.detach().cpu().numpy()
        quats_np = gaussian_obj.get_rotation.detach().cpu().numpy()
        Rs = tf.SO3(quats_np).as_matrix()
        S_diag = np.eye(3)[None, :, :] * scales_np[:, None, :] ** 2
        covariances_np = np.einsum("nij,njk,nlk->nil", Rs, S_diag, Rs)
    return centers_np, covariances_np, rgbs_np, opacities_np


def get_mask_bbox(mask_pil, threshold=128):
    rows, cols = np.where(np.array(mask_pil) > threshold)
    if rows.size == 0 or cols.size == 0:
        return None
    return (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)


def get_smart_crop_bbox(mask_pil, min_ratio=0.3, max_ratio=0.6, model_input_size=512):
    bbox = get_mask_bbox(mask_pil)
    if bbox is None:
        return (0, 0, model_input_size, model_input_size), model_input_size

    min_x, min_y, max_x, max_y = bbox
    mask_w = max_x - min_x
    mask_h = max_y - min_y
    mask_area = mask_w * mask_h
    target_crop_side = int(math.sqrt(mask_area / ((min_ratio + max_ratio) / 2.0)))
    target_crop_side = max(target_crop_side, max(mask_w, mask_h) + 20)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    half_side = target_crop_side // 2
    x1 = center_x - half_side
    y1 = center_y - half_side
    return (x1, y1, x1 + target_crop_side, y1 + target_crop_side), target_crop_side


def crop_and_pad(image, bbox, target_side):
    x1, y1, x2, y2 = bbox
    W, H = image.size
    valid_crop = image.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    new_img = Image.new(image.mode, (target_side, target_side), 0)
    new_img.paste(valid_crop, (max(0, -x1), max(0, -y1)))
    return new_img


def uniform_dilate_mask(mask, expand_radius=5):
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    kernel_size = expand_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return (cv2.dilate(mask_uint8, kernel, iterations=1) > 0).astype(np.uint8)


def adain_color_fix(target_pil, source_pil, mask_pil):
    to_tensor = ToTensor()
    t_tensor = to_tensor(target_pil).unsqueeze(0)
    s_tensor = to_tensor(source_pil).unsqueeze(0)
    m_tensor = to_tensor(mask_pil).unsqueeze(0)
    eps = 1e-5
    res = t_tensor.clone()

    for ch in range(3):
        bg_indices = m_tensor[0, 0] < 0.1
        if bg_indices.sum() < 10:
            continue
        s_pixels = s_tensor[0, ch][bg_indices]
        t_pixels = t_tensor[0, ch][bg_indices]
        s_mean, s_std = s_pixels.mean(), s_pixels.std() + eps
        t_mean, t_std = t_pixels.mean(), t_pixels.std() + eps
        res[0, ch] = (t_tensor[0, ch] - t_mean) * (s_std / t_std) + s_mean

    return ToPILImage()(res.squeeze(0).clamp(0.0, 1.0))


def get_bbox_from_mask(mask: np.ndarray):
    fg = mask > 0
    if not np.any(fg):
        H, W = mask.shape
        return 0, H, 0, W
    ys, xs = np.where(fg)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def expand_image_mask(cropped_image: np.ndarray, cropped_mask: np.ndarray, ratio=1.2):
    h, w = cropped_image.shape[:2]
    side = max(int(np.ceil(max(h, w) * ratio)), 1)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    canvas_img = np.zeros((side, side, 3), dtype=cropped_image.dtype)
    canvas_mask = np.zeros((side, side), dtype=cropped_mask.dtype)
    canvas_img[pad_y:pad_y + h, pad_x:pad_x + w] = cropped_image
    canvas_mask[pad_y:pad_y + h, pad_x:pad_x + w] = cropped_mask
    return canvas_img, canvas_mask


def pad_to_square(arr: np.ndarray, pad_value=0):
    H, W = arr.shape[:2]
    if H == W:
        return arr
    if H > W:
        diff = H - W
        left = diff // 2
        right = diff - left
        pad_width = ((0, 0), (left, right)) if arr.ndim == 2 else ((0, 0), (left, right), (0, 0))
    else:
        diff = W - H
        top = diff // 2
        bottom = diff - top
        pad_width = ((top, bottom), (0, 0)) if arr.ndim == 2 else ((top, bottom), (0, 0), (0, 0))
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def center_reference_image_pil(img_pil, ratio=1.2, out_size=512):
    img_np = np.array(img_pil)
    mask = (img_np.sum(axis=2) > 0).astype(np.uint8)
    y1, y2, x1, x2 = get_bbox_from_mask(mask)
    if y2 <= y1 or x2 <= x1:
        return img_pil.resize((out_size, out_size)), Image.fromarray(mask * 255)

    cropped_img = img_np[y1:y2, x1:x2, :]
    cropped_msk = mask[y1:y2, x1:x2]
    expanded_img, expanded_msk = expand_image_mask(cropped_img, cropped_msk, ratio=ratio)
    padded_img = pad_to_square(expanded_img, pad_value=0)
    padded_msk = pad_to_square(expanded_msk, pad_value=0)
    resized_img = cv2.resize(padded_img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    resized_msk = cv2.resize(padded_msk, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(resized_img), Image.fromarray((resized_msk > 0).astype(np.uint8) * 255)


def postprocess_rendered_image(gauss_bchw: torch.Tensor, ratio=1.2, out_size=512):
    assert gauss_bchw.ndim == 4 and gauss_bchw.shape[1] == 3
    device = gauss_bchw.device
    weight_dtype = gauss_bchw.dtype
    gauss_bchw = gauss_bchw.to(torch.float32)
    out_imgs = []
    out_masks = []

    for b in range(gauss_bchw.shape[0]):
        img_tensor = gauss_bchw[b].permute(1, 2, 0).detach().cpu().numpy()
        img = (np.clip(img_tensor, 0, 1) * 255.0).astype(np.uint8)
        mask = (img.sum(axis=2) > 0).astype(np.uint8)
        y1, y2, x1, x2 = get_bbox_from_mask(mask)

        if y2 <= y1 or x2 <= x1:
            resized_img = np.zeros((out_size, out_size, 3), dtype=np.uint8)
            resized_msk = np.zeros((out_size, out_size), dtype=np.uint8)
        else:
            cropped_img = img[y1:y2, x1:x2, :]
            cropped_msk = mask[y1:y2, x1:x2]
            expanded_img, expanded_msk = expand_image_mask(cropped_img, cropped_msk, ratio=ratio)
            padded_img = pad_to_square(expanded_img, pad_value=0)
            padded_msk = pad_to_square(expanded_msk, pad_value=0)
            resized_img = cv2.resize(padded_img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            resized_msk = cv2.resize(padded_msk, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        out_imgs.append(torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0)
        out_masks.append(torch.from_numpy((resized_msk > 0).astype(np.float32))[None, ...])

    imgs_bchw = torch.stack(out_imgs, dim=0).to(device).to(weight_dtype)
    masks_bchw = torch.stack(out_masks, dim=0).to(device).to(weight_dtype)
    return imgs_bchw, masks_bchw


def prepare_preview_background(bg_pil):
    orig_w, orig_h = bg_pil.size
    render_resolution = max(orig_w, orig_h)
    bg_square_pil = Image.new("RGB", (render_resolution, render_resolution), (0, 0, 0))
    pad_x = (render_resolution - orig_w) // 2
    pad_y = (render_resolution - orig_h) // 2
    bg_square_pil.paste(bg_pil, (pad_x, pad_y))
    crop_box = (pad_x, pad_y, pad_x + orig_w, pad_y + orig_h)
    return bg_square_pil, crop_box, render_resolution


def apply_gaussian_transform(gaussian_obj, wxyz, position, device):
    R_obj = tf.SO3(wxyz).as_matrix()
    T_obj = position
    R_torch = torch.tensor(R_obj, device=device, dtype=torch.float32)
    T_torch = torch.tensor(T_obj, device=device, dtype=torch.float32)

    gaussian_placed = copy.deepcopy(gaussian_obj)
    xyz_orig = gaussian_obj.get_xyz
    xyz_new = (R_torch @ xyz_orig.transpose(0, 1)).transpose(0, 1) + T_torch
    gaussian_placed.from_xyz(xyz_new)

    quats_orig = gaussian_obj.get_rotation
    rot_mats_orig = utils3d.torch.quaternion_to_matrix(quats_orig)
    rot_mats_new = R_torch @ rot_mats_orig
    quats_new = utils3d.torch.matrix_to_quaternion(rot_mats_new)
    gaussian_placed.from_rotation(quats_new)
    return gaussian_placed


def get_fixed_view_matrix(eye, device):
    eye = torch.tensor(eye, device=device, dtype=torch.float32)
    target_look_at = eye + torch.tensor([0.0, 0.0, -1.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)

    z_axis = target_look_at - eye
    z_axis = z_axis / torch.norm(z_axis)
    x_axis = torch.cross(z_axis, up)
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = torch.cross(x_axis, z_axis)
    y_axis = y_axis / torch.norm(y_axis)
    y_axis = -y_axis

    R = torch.stack([x_axis, y_axis, z_axis])
    T = -torch.matmul(R, eye)
    view_mat = torch.eye(4, device=device)
    view_mat[:3, :3] = R
    view_mat[:3, 3] = T
    return view_mat


def render_gaussian_preview(gaussian_obj, transform_handle, camera, render_resolution, device):
    from trellis.renderers import GaussianRenderer

    gaussian_placed = apply_gaussian_transform(
        gaussian_obj,
        wxyz=transform_handle.wxyz,
        position=transform_handle.position,
        device=device,
    )
    view_matrix = get_fixed_view_matrix(camera.position, device=device)

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.far = 2000.0
    renderer.rendering_options.near = 0.1
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1

    fov_rad = camera.fov
    intr = utils3d.torch.intrinsics_from_fov_xy(
        torch.tensor(fov_rad, device=device),
        torch.tensor(fov_rad, device=device)
    ).to(device)
    return renderer.render(gaussian_placed, view_matrix, intr)["color"]


def make_condition_image(res_color, target_res):
    gauss_bchw = res_color.unsqueeze(0)
    cond_imgs_bchw, _ = postprocess_rendered_image(gauss_bchw, ratio=1.2, out_size=target_res)
    cond_img_tensor = cond_imgs_bchw[0]
    cond_img_np = (cond_img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(cond_img_np)


def make_composite_preview(bg_square_pil, res_color, crop_box):
    mask_tensor_hd = (res_color > 1e-3).any(dim=0).float()
    mask_np_hd = (mask_tensor_hd.detach().cpu().numpy() * 255).astype(np.uint8)
    mask_pil_hd = Image.fromarray(mask_np_hd, mode="L")

    fg_tensor_hd = torch.clamp(res_color, 0, 1)
    fg_np_hd = (fg_tensor_hd.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    fg_pil_hd = Image.fromarray(fg_np_hd)

    composite_square = bg_square_pil.copy()
    composite_square.paste(fg_pil_hd, (0, 0), mask_pil_hd)
    return composite_square.crop(crop_box), mask_pil_hd.crop(crop_box)


def refine_mask_holes(mask_bool_arr, kernel_size=5):
    mask_uint8 = mask_bool_arr.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(closed_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask > 127


def build_generation_inputs(bg_pil, composite_pil, mask_pil, reference_image, geometry_image, target_res):
    processed_reference_image, _ = center_reference_image_pil(reference_image, ratio=1.2, out_size=target_res)

    mask_np = np.array(mask_pil)
    dilated_mask_01 = uniform_dilate_mask(mask_np, expand_radius=10)
    dilated_mask_pil = Image.fromarray(dilated_mask_01 * 255, mode="L")

    full_bg_arr = np.array(bg_pil)
    mask_expanded = dilated_mask_01[:, :, None]
    masked_full_bg_arr = full_bg_arr * (1 - mask_expanded)
    context_image = Image.fromarray(masked_full_bg_arr.astype(np.uint8))

    ideal_bbox, target_side = get_smart_crop_bbox(
        dilated_mask_pil,
        min_ratio=0.02,
        max_ratio=0.3,
        model_input_size=target_res,
    )

    patch_composite = crop_and_pad(composite_pil, ideal_bbox, target_side)
    patch_mask = crop_and_pad(dilated_mask_pil, ideal_bbox, target_side)
    patch_background_reference = crop_and_pad(bg_pil, ideal_bbox, target_side)
    patch_mask_orig = crop_and_pad(mask_pil, ideal_bbox, target_side)

    comp_arr = np.array(patch_composite)
    mask_dilated_arr = np.array(patch_mask) > 127
    raw_mask_orig_arr = np.array(patch_mask_orig) > 127
    mask_orig_arr = refine_mask_holes(raw_mask_orig_arr, kernel_size=7)

    diff_region = mask_dilated_arr & (~mask_orig_arr)
    comp_arr[diff_region] = [0, 0, 0]

    patch_composite = Image.fromarray(comp_arr)
    patch_mask = Image.fromarray(np.array(patch_mask).astype(np.uint8))
    composite_image = patch_composite.resize((target_res, target_res), Image.Resampling.LANCZOS)
    model_input_mask = patch_mask.resize((target_res, target_res), Image.Resampling.NEAREST)
    if geometry_image.size != (target_res, target_res):
        geometry_image = geometry_image.resize((target_res, target_res), Image.Resampling.LANCZOS)

    mask_arr = np.array(model_input_mask)
    inpaint_mask_01 = (mask_arr > 0).astype(np.float32)
    inpaint_mask = Image.fromarray((inpaint_mask_01 * 255).astype(np.uint8))
    background_reference_image = patch_background_reference.resize((target_res, target_res), Image.Resampling.LANCZOS)

    return GenerationInputs(
        reference_image=processed_reference_image,
        composite_image=composite_image,
        inpaint_mask=inpaint_mask,
        geometry_image=geometry_image,
        context_image=context_image,
        model_input_mask=model_input_mask,
        background_reference_image=background_reference_image,
        ideal_bbox=ideal_bbox,
        target_side=target_side,
    )


def paste_generated_patch(bg_pil, generated_patch, generation_inputs):
    fixed_patch = adain_color_fix(
        target_pil=generated_patch,
        source_pil=generation_inputs.background_reference_image,
        mask_pil=generation_inputs.model_input_mask,
    )
    fixed_patch_physical = fixed_patch.resize(
        (generation_inputs.target_side, generation_inputs.target_side),
        Image.Resampling.LANCZOS,
    )
    crop_min_x, crop_min_y, crop_max_x, crop_max_y = generation_inputs.ideal_bbox

    orig_w, orig_h = bg_pil.size
    pad_left = max(0, -crop_min_x)
    pad_top = max(0, -crop_min_y)
    valid_w = min(orig_w, crop_max_x) - max(0, crop_min_x)
    valid_h = min(orig_h, crop_max_y) - max(0, crop_min_y)
    patch_valid = fixed_patch_physical.crop((pad_left, pad_top, pad_left + valid_w, pad_top + valid_h))

    final_result = bg_pil.copy()
    final_result.paste(patch_valid, (max(0, crop_min_x), max(0, crop_min_y)))
    return final_result
