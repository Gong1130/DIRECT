import math

import torch
import torch.nn.functional as F
import utils3d
from trellis.renderers import GaussianRenderer, MeshRenderer


def paste_geometry_condition(
    masked_target_image,
    object_mask,
    target_gaussian_image,
    target_gaussian_mask,
    inpainting_mask,
):
    B, C, H, W = masked_target_image.shape
    dtype = masked_target_image.dtype

    updated_image = masked_target_image.clone()
    updated_inpainting_mask = inpainting_mask.clone()

    for i in range(B):
        gs_mask = target_gaussian_mask[i, 0] > 0.5
        y_idx_s, x_idx_s = torch.where(gs_mask)
        if len(y_idx_s) == 0:
            continue
        y1_s, y2_s = y_idx_s.min(), y_idx_s.max()
        x1_s, x2_s = x_idx_s.min(), x_idx_s.max()
        h_s, w_s = (y2_s - y1_s + 1).item(), (x2_s - x1_s + 1).item()
        area_s = h_s * w_s
        gs_crop = target_gaussian_image[i:i + 1, :, y1_s:y2_s + 1, x1_s:x2_s + 1]
        gs_mask_crop = target_gaussian_mask[i:i + 1, :, y1_s:y2_s + 1, x1_s:x2_s + 1]

        obj_mask_bool = object_mask[i, 0] > 0.5
        y_idx_t, x_idx_t = torch.where(obj_mask_bool)
        if len(y_idx_t) == 0:
            continue
        y1_t, y2_t = y_idx_t.min(), y_idx_t.max()
        x1_t, x2_t = x_idx_t.min(), x_idx_t.max()
        h_t, w_t = (y2_t - y1_t + 1).item(), (x2_t - x1_t + 1).item()
        area_t = h_t * w_t
        center_y_t, center_x_t = (y1_t + y2_t) // 2, (x1_t + x2_t) // 2

        if area_s <= 0:
            continue
        scale = (area_t / area_s) ** 0.5
        new_h, new_w = int(round(h_s * scale)), int(round(w_s * scale))
        if new_h < 1 or new_w < 1:
            continue

        gs_crop_resized = F.interpolate(gs_crop, size=(new_h, new_w), mode="bilinear", align_corners=False)
        gs_mask_resized = F.interpolate(gs_mask_crop, size=(new_h, new_w), mode="bilinear", align_corners=False)

        paste_y1 = int(center_y_t - new_h // 2)
        paste_x1 = int(center_x_t - new_w // 2)
        paste_y2, paste_x2 = paste_y1 + new_h, paste_x1 + new_w

        valid_y1, valid_x1 = max(0, paste_y1), max(0, paste_x1)
        valid_y2, valid_x2 = min(H, paste_y2), min(W, paste_x2)

        if valid_y1 >= valid_y2 or valid_x1 >= valid_x2:
            continue

        src_y1, src_x1 = valid_y1 - paste_y1, valid_x1 - paste_x1
        src_y2 = src_y1 + (valid_y2 - valid_y1)
        src_x2 = src_x1 + (valid_x2 - valid_x1)

        gs_roi = gs_crop_resized[:, :, src_y1:src_y2, src_x1:src_x2]
        gs_mask_roi = gs_mask_resized[:, :, src_y1:src_y2, src_x1:src_x2]
        target_roi = updated_image[i:i + 1, :, valid_y1:valid_y2, valid_x1:valid_x2]

        gs_mask_roi_bin = (gs_mask_roi > 0.5).to(dtype)
        pasted_roi = gs_roi * gs_mask_roi_bin + target_roi * (1 - gs_mask_roi_bin)
        updated_image[i, :, valid_y1:valid_y2, valid_x1:valid_x2] = pasted_roi.squeeze(0)

        mask_roi = updated_inpainting_mask[i:i + 1, :, valid_y1:valid_y2, valid_x1:valid_x2]
        new_mask_roi = torch.max(mask_roi, gs_mask_roi_bin)
        updated_inpainting_mask[i, :, valid_y1:valid_y2, valid_x1:valid_x2] = new_mask_roi.squeeze(0)

    return updated_image, updated_inpainting_mask


def render_normal_from_slat(slats, extrs, decoder):
    device = extrs.device

    with torch.no_grad():
        meshs_all = []
        chunk_size = 2
        B = slats.shape[0]
        for s in range(0, B, chunk_size):
            e = min(B, s + chunk_size)
            meshs_chunk = decoder(slats[s:e])
            meshs_all.extend(meshs_chunk)
            del meshs_chunk
            torch.cuda.empty_cache()
    meshs = meshs_all

    renderer = MeshRenderer(device=device)
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 1

    results = []
    for i in range(len(meshs)):
        mesh = meshs[i]
        fov = torch.deg2rad(torch.tensor(float(40), device=device))
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        res = renderer.render(mesh, extrs[i], intr, return_types=["mask", "normal"])
        mask = res["mask"].detach().unsqueeze(0)
        normal = torch.clamp(res["normal"].detach(), 0, 1)
        normal_masked = normal * mask
        results.append(normal_masked)

    del res, mask, normal, normal_masked
    torch.cuda.empty_cache()
    return torch.stack(results)


def crop_pad_expand_and_resize(color, mask, size, expand_factor=1.2):
    device = color.device
    target_H, target_W = size
    target_aspect = target_W / target_H

    if mask.dtype == torch.bool:
        mask = mask.float()

    mask_2d = mask.squeeze(0) > 0.5
    if not mask_2d.any():
        return (
            torch.zeros((3, target_H, target_W), device=device, dtype=torch.float32),
            torch.zeros((1, target_H, target_W), device=device, dtype=torch.float32),
        )

    coords = torch.nonzero(mask_2d, as_tuple=False)
    ys = coords[:, 0]
    xs = coords[:, 1]
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1
    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1

    crop_color = color[:, y1:y2, x1:x2]
    crop_mask = mask[:, y1:y2, x1:x2]

    ch = crop_color.shape[1]
    cw = crop_color.shape[2]

    new_h = int(math.ceil(ch * expand_factor))
    new_w = int(math.ceil(cw * expand_factor))
    pad_h_total = max(0, new_h - ch)
    pad_w_total = max(0, new_w - cw)
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    if any((pad_top, pad_bottom, pad_left, pad_right)):
        pad_params = (pad_left, pad_right, pad_top, pad_bottom)
        crop_color = F.pad(crop_color, pad_params, mode="constant", value=0.0)
        crop_mask = F.pad(crop_mask, pad_params, mode="constant", value=0.0)
        ch = crop_color.shape[1]
        cw = crop_color.shape[2]

    current_aspect = float(cw) / float(ch)
    pad_top2 = pad_bottom2 = pad_left2 = pad_right2 = 0
    if abs(current_aspect - target_aspect) > 1e-6:
        if current_aspect < target_aspect:
            desired_w = int(math.ceil(ch * target_aspect))
            pad_w = max(0, desired_w - cw)
            pad_left2 = pad_w // 2
            pad_right2 = pad_w - pad_left2
        else:
            desired_h = int(math.ceil(cw / target_aspect))
            pad_h = max(0, desired_h - ch)
            pad_top2 = pad_h // 2
            pad_bottom2 = pad_h - pad_top2

        if any((pad_top2, pad_bottom2, pad_left2, pad_right2)):
            pad_params2 = (pad_left2, pad_right2, pad_top2, pad_bottom2)
            crop_color = F.pad(crop_color, pad_params2, mode="constant", value=0.0)
            crop_mask = F.pad(crop_mask, pad_params2, mode="constant", value=0.0)

    crop_color_b = crop_color.unsqueeze(0)
    crop_mask_b = crop_mask.unsqueeze(0)

    resized_color = F.interpolate(crop_color_b, size=(target_H, target_W), mode="bilinear", align_corners=False)
    resized_mask = F.interpolate(crop_mask_b, size=(target_H, target_W), mode="nearest")

    return resized_color.squeeze(0), resized_mask.squeeze(0)


def render_gaussian_from_slat_arbitrary_size(slats, extrs, size, decoder, return_mask=False):
    device = extrs.device

    with torch.no_grad():
        gaussians = decoder(slats)

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(min(*size))
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = 0.1
    renderer.pipe.use_mip_gaussian = True

    results = []
    masks = []
    for i in range(len(gaussians)):
        gaussian = gaussians[i]
        fov = torch.deg2rad(torch.tensor(float(40), device=device))
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        res = renderer.render(gaussian, extrs[i], intr)
        res = torch.clamp(res["color"].detach(), 0, 1)
        mask = (res > 1e-3).any(dim=0).unsqueeze(0)
        res, mask = crop_pad_expand_and_resize(res, mask, size)
        results.append(res)
        masks.append(mask)
    if return_mask:
        return torch.stack(results), torch.stack(masks)
    return torch.stack(results)
