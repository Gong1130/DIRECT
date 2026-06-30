# DIRECT Preprocess

This folder provides an example single-image preprocessing pipeline for *Geometric Alignment*, described in Section 3.4 of our [paper](https://arxiv.org/abs/2606.06601).

Given an object image and its binary object mask, the script reconstructs a 3D proxy with TRELLIS, estimates the image pose in the reconstructed 3D object with VGGT, and refines the pose by matching the rendered alpha mask to the input mask. The resulting files can be used as DIRECT training metadata.

<p align="center">
  <img src="../assets/alignment.png" alt="DIRECT geometric alignment" width="900">
</p>

<details>
<summary>Preprocessing pipeline</summary>

1. Crop the object image and mask around the foreground object.
2. Reconstruct a TRELLIS 3D proxy from the cropped object image.
3. Render six canonical views of the 3D proxy.
4. Run VGGT on the rendered views and the input image to estimate camera poses.
5. Align the VGGT camera coordinate system to the TRELLIS render camera system.
6. Refine the target image pose by differentiable alpha rendering.
7. Save the sparse latent, optimized pose, quality metrics, and optional normal rendering.

</details>

## Input

The example script expects:

- `image_path`: an RGB object image.
- `mask_path`: a binary object mask aligned with the image.


## Output

For each input image, the script writes:

- `slat.pt`: TRELLIS sparse latent for the reconstructed 3D proxy.
- `optimized_w2c.pt`: optimized world-to-camera pose for the input image.
- `metadata.json`: preprocessing metrics and output paths.
- `input_processed.png`: cropped and masked input image used by TRELLIS and VGGT.
- `optimized_render.png`: final rendered view after pose optimization.
- `normal.png`: rendered normal map, when `--save_normal` is enabled.
- `gaussian.pt`: decoded Gaussian proxy, when `--save_gaussian` is enabled.

The released DIRECT dataset uses `slat.pt`, `optimized_w2c.pt`, and `normal.png` during training.

The quality metrics in `metadata.json` include:

- `iou`: IoU between the optimized render alpha and the input object mask.
- `black_image`: whether the optimized render has near-zero alpha.

These metrics can be used to filter failed preprocessing results when adapting DIRECT to new datasets.

## Usage

Install the main DIRECT environment first. The preprocess script also requires the custom differentiable Gaussian rasterizer used for alpha rendering:

```bash
git clone --recursive https://github.com/Gong1130/alpha_camera_diff_gaussian_rasterization.git third_party/alpha_camera_diff_gaussian_rasterization
pip install -e third_party/alpha_camera_diff_gaussian_rasterization --no-build-isolation
```

Run the reference pipeline:

```bash
python preprocess/preprocess_example.py \
  --image_path <path-to-image> \
  --mask_path <path-to-mask> \
  --output_dir <path-to-output> \
  --save_normal
```

On the first run, the script will automatically download [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) and [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) from Hugging Face.

## Acknowledgements

This preprocessing pipeline builds on [TRELLIS](https://github.com/microsoft/TRELLIS) and [VGGT](https://github.com/facebookresearch/vggt).
