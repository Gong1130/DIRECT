import argparse
import json
import os

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "auto")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


def load_image_and_mask(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_np = np.array(image)
    mask_np = (np.array(mask) > 128).astype(np.uint8)
    if mask_np.sum() == 0:
        raise ValueError(f"Mask is empty: {mask_path}")

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    y2 += 1
    x2 += 1

    h = y2 - y1
    w = x2 - x1
    cy = (y1 + y2) / 2
    cx = (x1 + x2) / 2
    side = int(np.ceil(max(h, w) * 1.2))

    crop_y1 = int(round(cy - side / 2))
    crop_x1 = int(round(cx - side / 2))
    crop_y2 = crop_y1 + side
    crop_x2 = crop_x1 + side

    pad_top = max(0, -crop_y1)
    pad_left = max(0, -crop_x1)
    pad_bottom = max(0, crop_y2 - image_np.shape[0])
    pad_right = max(0, crop_x2 - image_np.shape[1])
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        image_np = np.pad(
            image_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        mask_np = np.pad(
            mask_np,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        crop_y1 += pad_top
        crop_y2 += pad_top
        crop_x1 += pad_left
        crop_x2 += pad_left

    image_crop = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
    mask_crop = mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
    image_crop = np.array(Image.fromarray(image_crop).resize((512, 512), Image.BICUBIC))
    mask_crop = np.array(Image.fromarray(mask_crop * 255).resize((512, 512), Image.NEAREST))
    mask_crop = (mask_crop > 128).astype(np.uint8)

    masked_image = image_crop * mask_crop[..., None] + 255 * (1 - mask_crop[..., None])
    return Image.fromarray(masked_image.astype(np.uint8)), Image.fromarray((mask_crop * 255).astype(np.uint8))


def umeyama(src, dst, with_scaling=True):
    assert src.shape == dst.shape and src.shape[1] == 3, "Shape mismatch (N, 3)"
    n = src.shape[0]
    mu_src = src.mean(dim=0)
    mu_dst = dst.mean(dim=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    R = U @ Vh
    if torch.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = U @ Vh
    if with_scaling:
        var_src = (src_c ** 2).sum() / n
        s = S.sum() / var_src
    else:
        s = torch.tensor(1.0, dtype=src.dtype, device=src.device)
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def decompose_c2w(transform):
    return transform[:3, :3], transform[:3, 3]


def compose_c2w(rotation, translation):
    transform = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def align_poses(pred_c2w, render_c2w):
    assert len(pred_c2w) == len(render_c2w) + 1 and len(pred_c2w) >= 3

    target_c2w = pred_c2w[-1]
    pred_c2w = pred_c2w[:-1]

    pred_centers = []
    render_centers = []
    for pred_pose, render_pose in zip(pred_c2w, render_c2w):
        _, pred_t = decompose_c2w(pred_pose)
        _, render_t = decompose_c2w(render_pose)
        pred_centers.append(pred_t)
        render_centers.append(render_t)

    pred_centers = torch.stack(pred_centers, dim=0)
    render_centers = torch.stack(render_centers, dim=0)
    s, rotation_align, translation_align = umeyama(pred_centers, render_centers, with_scaling=True)

    aligned_c2w = []
    for pose in [*pred_c2w, target_c2w]:
        rotation, translation = decompose_c2w(pose)
        aligned_translation = s * (rotation_align @ translation) + translation_align
        aligned_rotation = rotation_align @ rotation
        aligned_c2w.append(compose_c2w(aligned_rotation, aligned_translation))

    return torch.stack(aligned_c2w, dim=0)


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    zero = torch.zeros_like(w0)
    return torch.stack(
        [
            torch.stack([zero, -w2, w1], dim=-1),
            torch.stack([w2, zero, -w0], dim=-1),
            torch.stack([-w1, w0, zero], dim=-1),
        ],
        dim=-2,
    )


def taylor_A(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        if i > 0:
            denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_B(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_C(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def se3_to_SE3(w, v):
    delta = torch.zeros((4, 4), device=w.device, dtype=torch.float32)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)
    eye = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    delta[:3, :3] = eye + A * wx + B * wx @ wx
    delta[:3, 3] = (eye + B * wx + C * wx @ wx) @ v
    delta[3, 3] = 1.0
    return delta


class CameraOptimizationModel(nn.Module):
    def __init__(self, gaussian, renderer, init_w2c, intr, device):
        super().__init__()
        self.gaussian = gaussian
        self.renderer = renderer
        self.init_w2c = init_w2c
        self.t = nn.Parameter(torch.zeros(3, device=device))
        self.w = nn.Parameter(torch.zeros(3, device=device))
        self.v = nn.Parameter(torch.zeros(3, device=device))
        self.intr = intr
        self.mode = "t_only"

    def forward(self):
        if self.mode == "t_only":
            cur_w2c = self.init_w2c.clone()
            cur_w2c[:3, 3] = cur_w2c[:3, 3] + self.t
        elif self.mode == "wv":
            delta = se3_to_SE3(self.w, self.v)
            cur_w2c = delta @ self.init_w2c
        else:
            raise ValueError(f"Unknown optimization mode: {self.mode}")

        render_output = self.renderer.alpha_render(self.gaussian, cur_w2c, self.intr)
        render_image = render_output["color"] * 2 - 1
        render_alpha = render_output["alpha"]
        return render_image, render_alpha, cur_w2c


def compute_iou(alpha1, alpha2, threshold=0.5):
    mask1 = (alpha1 > threshold).detach().cpu().numpy()
    mask2 = (alpha2 > threshold).detach().cpu().numpy()
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def should_rerun(render_alpha, iou):
    return render_alpha.max() < 0.05 or iou < 0.9


def optimize_camera(model, alpha_ref, mode):
    if mode == "wv_only":
        t_stage_steps = 0
        wv_stage_steps = 100
    elif mode == "t_then_wv":
        t_stage_steps = 100
        wv_stage_steps = 50
    else:
        raise ValueError(f"Unknown optimization mode: {mode}")

    iteration = t_stage_steps + wv_stage_steps
    optimizer = torch.optim.Adam([model.w, model.v], lr=0.1) if mode == "wv_only" else torch.optim.Adam([model.t], lr=0.1)
    model.mode = "wv" if mode == "wv_only" else "t_only"
    latest_w2c = None

    for i in tqdm(range(iteration), desc=f"Optimizing [{mode}]", disable=True):
        if i == t_stage_steps and mode == "t_then_wv":
            model.mode = "wv"
            model.init_w2c = latest_w2c.clone().detach()
            optimizer = torch.optim.Adam([model.w, model.v], lr=0.05)

        optimizer.zero_grad()
        render_image, render_alpha, latest_w2c = model()
        loss = ((render_alpha - alpha_ref) ** 2).mean()
        loss.backward()
        optimizer.step()

        if i == 10 and render_alpha.max().item() < 0.05:
            return None, render_alpha, None, 0.0

    final_image = (render_image.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    iou = compute_iou(render_alpha[0], alpha_ref[0])
    return latest_w2c.detach().cpu(), render_alpha, final_image, iou


def detach_gaussian(gaussian):
    gaussian._xyz = gaussian._xyz.detach().cpu()
    gaussian._features_dc = gaussian._features_dc.detach().cpu()
    gaussian._features_rest = gaussian._features_rest.detach().cpu()
    gaussian._opacity = gaussian._opacity.detach().cpu()
    gaussian._scaling = gaussian._scaling.detach().cpu()
    gaussian._rotation = gaussian._rotation.detach().cpu()
    return gaussian


def process_single_image(args):
    import torchvision.transforms as tv_transforms
    import torchvision.utils as vutils
    import utils3d
    from direct.geometry import render_normal_from_slat
    from trellis.modules import sparse as sp
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.renderers import GaussianRenderer
    from trellis.utils import render_utils
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    image_pil, mask_pil = load_image_and_mask(args.image_path, args.mask_path)
    image_pil.save(os.path.join(args.output_dir, "input_processed.png"))

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model_path)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    vggt_model = VGGT.from_pretrained(args.vggt_model_path).to(device).eval()

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (1, 1, 1)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = 0.1
    renderer.pipe.use_mip_gaussian = True

    with torch.no_grad():
        slat_gpu = pipeline.get_slat(
            image_pil,
            mask_image=mask_pil,
            seed=1,
            preprocess_image=True,
            apply_rmbg=False,
        )[0]
        gaussian = pipeline.decode_slat(slat_gpu, ["gaussian"])["gaussian"][0]

    slat_cpu = sp.SparseTensor(
        feats=slat_gpu.feats.detach().cpu(),
        coords=slat_gpu.coords.detach().cpu(),
    )
    torch.save({"feats": slat_cpu.feats, "coords": slat_cpu.coords}, os.path.join(args.output_dir, "slat.pt"))

    video, extrinsics, _ = render_utils.render_6view(
        gaussian,
        resolution=518,
        bg_color=(1, 1, 1),
    )
    render_w2c = torch.stack(extrinsics, dim=0).to(device)

    vggt_input = [tv_transforms.ToTensor()(frame) for frame in video]
    vggt_input.append(tv_transforms.ToTensor()(image_pil.resize((518, 518), Image.Resampling.BICUBIC)))
    vggt_input = torch.stack(vggt_input).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=device.type == "cuda"):
            predictions = vggt_model(vggt_input)
    pred_w2c, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], vggt_input.shape[-2:])
    pred_w2c = pred_w2c.squeeze(0)
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=pred_w2c.dtype, device=device)
    bottom_row = bottom_row.view(1, 1, 4).expand(pred_w2c.shape[0], 1, 4)
    pred_w2c = torch.cat([pred_w2c, bottom_row], dim=1)

    aligned_c2w = align_poses(torch.linalg.inv(pred_w2c), torch.linalg.inv(render_w2c))
    init_w2c = torch.linalg.inv(aligned_c2w[-1])

    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    alpha_ref = torch.from_numpy(mask_np).to(device).unsqueeze(0)

    fov = torch.deg2rad(torch.tensor(40.0, device=device))
    default_intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

    model1 = CameraOptimizationModel(gaussian, renderer, init_w2c.detach(), default_intr, device)
    w2c1, alpha1, img1, iou1 = optimize_camera(model1, alpha_ref, mode="wv_only")
    if should_rerun(alpha1, iou1):
        model2 = CameraOptimizationModel(gaussian, renderer, init_w2c.detach(), default_intr, device)
        w2c2, alpha2, img2, iou2 = optimize_camera(model2, alpha_ref, mode="t_then_wv")
        if iou1 >= iou2:
            w2c, alpha_final, img_final, iou_final = w2c1, alpha1, img1, iou1
        else:
            w2c, alpha_final, img_final, iou_final = w2c2, alpha2, img2, iou2
    else:
        w2c, alpha_final, img_final, iou_final = w2c1, alpha1, img1, iou1

    black_image = bool(alpha_final.max().item() < 0.05)
    if w2c is not None:
        torch.save(w2c, os.path.join(args.output_dir, "optimized_w2c.pt"))

    if img_final is not None:
        Image.fromarray(img_final).save(os.path.join(args.output_dir, "optimized_render.png"))

    if args.save_normal and not black_image and w2c is not None:
        normal_image = render_normal_from_slat(
            slat_gpu,
            w2c.unsqueeze(0).to(device),
            decoder=pipeline.models["slat_decoder_mesh"],
        ).detach().cpu()
        vutils.save_image(normal_image, os.path.join(args.output_dir, "normal.png"))

    if args.save_gaussian:
        torch.save(detach_gaussian(gaussian), os.path.join(args.output_dir, "gaussian.pt"))

    metadata = {
        "image_path": args.image_path,
        "mask_path": args.mask_path,
        "iou": float(iou_final),
        "black_image": black_image,
        "saved_slat": os.path.join(args.output_dir, "slat.pt"),
        "saved_w2c": os.path.join(args.output_dir, "optimized_w2c.pt") if w2c is not None else None,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Reference DIRECT preprocessing for a single image and mask.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trellis_model_path", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--vggt_model_path", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_gaussian", action="store_true")
    parser.add_argument("--save_normal", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    process_single_image(parse_args())
