import argparse
import base64
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from PIL import Image
import torch
import viser

from demo_utils import (
    build_generation_inputs,
    extract_gaussian_params,
    make_composite_preview,
    make_condition_image,
    paste_generated_patch,
    prepare_preview_background,
    random_seed,
    render_gaussian_preview,
)

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

MODEL_INPUT_RESOLUTION = 1024

# Model paths
DIRECT_MODEL_PATH = "superGong/DIRECT"
FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-Fill-dev"
SIGLIP_MODEL_PATH = "google/siglip2-so400m-patch14-384"
TRELLIS_MODEL_PATH = "microsoft/TRELLIS-image-large"


def parse_args():
    parser = argparse.ArgumentParser(description="DIRECT Gradio demo")
    parser.add_argument("--gradio_port", type=int, default=7860, help="Port for the Gradio app.")
    parser.add_argument("--viser_port", type=int, default=8081, help="Port for the Viser viewer.")
    return parser.parse_args()


with open(PROJECT_ROOT / "assets/direct_logo1.png", "rb") as f:
    LOGO_B64 = base64.b64encode(f.read()).decode("utf-8")

device = None
direct_pipeline = None
i23dpipe = None
viser_server = None
global_transform_handle = None
# Page styles
custom_css = """
footer {visibility: hidden}

.gradio-container {
    align-items: center;
}

#viser_container_wrapper {
    max-width: 100%; 
    background: #111;
    position: relative;
    height: 500px; 
    margin-left: auto;
    margin-right: auto;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

#viser_container_wrapper iframe {
    width: 100%;
    height: 100%;
    border: none;
    display: block;
    border-radius: 8px;
}

#seed_control {
    position: relative;
}

#seed_control input[type="number"] {
    padding-right: 58px !important;
}

#seed_random_btn {
    position: absolute !important;
    right: 14px;
    bottom: 12px;
    width: 38px !important;
    min-width: 38px !important;
    height: 38px !important;
    z-index: 5;
}

#seed_random_btn button {
    width: 38px !important;
    min-width: 38px !important;
    height: 38px !important;
    padding: 0 !important;
    border-radius: 8px !important;
}

.direct-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin: 8px auto 28px;
    text-align: center;
}

.direct-header img {
    width: 96px;
    height: 96px;
    object-fit: contain;
}

.direct-header h1 {
    margin: 0;
    font-size: 34px;
    line-height: 1.18;
    font-weight: 700;
    letter-spacing: 0;
}

.direct-tutorial {
    max-width: 900px;
    margin: -8px auto 30px;
    padding: 16px 22px;
    text-align: left;
    border: 1px solid #ddd6fe;
    border-radius: 8px;
    background: #faf5ff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.direct-tutorial h2 {
    margin: 0 0 10px;
    font-size: 20px;
    line-height: 1.25;
    font-weight: 650;
    letter-spacing: 0;
}

.direct-tutorial ol {
    margin: 0;
    padding-left: 22px;
    list-style-position: outside;
    color: #4b5563;
    font-size: 15px;
    line-height: 1.55;
}

.direct-tutorial li {
    margin: 4px 0;
}

.direct-tutorial .step-note {
    display: block;
    margin-top: 4px;
}

"""

# Page head
favicon_head_code = f"""
<link rel="icon" type="image/png" href="data:image/png;base64,{LOGO_B64}">
<link rel="apple-touch-icon" href="data:image/png;base64,{LOGO_B64}">
"""

js_head_code = """
<script>
    window.viserGlobal = {
        aspectRatio: 1.0,
        observer: null
    };

    function updateViser(dims) {
        if (!dims || dims.length < 2) return;

        const w_img = dims[0];
        const h_img = dims[1];
        window.viserGlobal.aspectRatio = h_img / w_img;

        const wrapper = document.getElementById("viser_container_wrapper");
        const parent = wrapper ? wrapper.parentElement : null;
        const setupPanel = document.getElementById("setup_panel");
        
        if (!wrapper || !parent) {
            setTimeout(() => updateViser(dims), 100);
            return;
        }

        const applyDimensions = () => {
            const availableWidth = parent.clientWidth;
            const setupHeight = setupPanel ? setupPanel.getBoundingClientRect().height : 0;
            const availableHeight = setupHeight > 0 ? setupHeight : window.innerHeight * 0.8;
            
            const targetRatio = window.viserGlobal.aspectRatio;

            let finalW = availableWidth;
            let finalH = finalW * targetRatio;

            if (finalH > availableHeight) {
                finalH = availableHeight;
                finalW = finalH / targetRatio;
            }

            wrapper.style.width = finalW + "px";
            wrapper.style.height = finalH + "px";
            wrapper.style.margin = "0 auto";
        };

        applyDimensions();

        if (window.viserGlobal.observer) window.viserGlobal.observer.disconnect();
        
        window.viserGlobal.observer = new ResizeObserver(() => {
            window.requestAnimationFrame(applyDimensions);
        });
        window.viserGlobal.observer.observe(parent);
        if (setupPanel) window.viserGlobal.observer.observe(setupPanel);
    }
</script>
"""


def load_models():
    global device, direct_pipeline, i23dpipe

    from direct import DirectPipeline
    from trellis.pipelines import TrellisImageTo3DPipeline

    print("Loading Models...")
    device = torch.device("cuda")
    direct_pipeline = DirectPipeline.from_pretrained(
        direct_model_path=DIRECT_MODEL_PATH,
        flux_model_path=FLUX_MODEL_PATH,
        siglip_model_path=SIGLIP_MODEL_PATH,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    i23dpipe = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_PATH)
    i23dpipe.cuda()
    i23dpipe.init_rmbg_model()
    print("Models Loaded.")


def setup_viser_server(viser_port):
    global viser_server

    viser_server = viser.ViserServer(port=viser_port)
    print(f"Viser server listening on port {viser_port}.")

    @viser_server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client {client.client_id} connected.")

        client.camera.position = np.array([0.0, 0.0, 3.0])
        client.camera.look_at = np.array([0.0, 0.0, 0.0])
        client.camera.up_direction = np.array([0.0, 1.0, 0.0])

        @client.camera.on_update
        def _(cam: viser.CameraHandle):
            current_z = max(0.5, cam.position[2])
            target_pos = np.array([0.0, 0.0, current_z])
            target_look_at = np.array([0.0, 0.0, 0.0])

            if (not np.allclose(cam.position, target_pos, atol=1e-2)) or \
               (not np.allclose(cam.look_at, target_look_at, atol=1e-2)):
                client.camera.position = target_pos
                client.camera.look_at = target_look_at
                client.camera.up_direction = np.array([0.0, 1.0, 0.0])


# Demo callbacks

def on_upload_object(image):
    if image is None: return None, None
    pil_image = Image.fromarray(image)
    return i23dpipe.preprocess_image(pil_image, apply_rmbg=True, return_hd_output=True)

def send_bg_to_viser(bg_image_pil):
    if bg_image_pil is None: return None
    w, h = bg_image_pil.size
    preview_img = bg_image_pil.copy()
    preview_img.thumbnail((1500, 1500)) 
    bg_np = np.array(preview_img)
    for client in viser_server.get_clients().values():
        client.scene.set_background_image(bg_np, format="jpeg")
    return [w, h]

def step1_generate_and_place_3d(image_np, seed):
    if image_np is None: raise gr.Error("Please upload an object image first.")
    global global_transform_handle
    
    pil_image = Image.fromarray(image_np)
    outputs = i23dpipe.run(pil_image, seed=seed, formats=["gaussian"], preprocess_image=False)
    gaussian_obj = outputs['gaussian'][0]
    centers, covs, rgbs, opacities = extract_gaussian_params(gaussian_obj)
    
    with viser_server.atomic():
        if global_transform_handle is not None:
            global_transform_handle.remove()
            
        global_transform_handle = viser_server.scene.add_transform_controls("/object_controls", scale=1)
        viser_server.scene.add_gaussian_splats(
            "/object_controls/splats", 
            centers=centers, covariances=covs, rgbs=rgbs, opacities=opacities
        )

    return gr.update(), gaussian_obj

def step2_preview_render(bg_pil, gaussian_obj):
    target_res = MODEL_INPUT_RESOLUTION
    if bg_pil is None: raise gr.Error("Please upload a background image.")
    if gaussian_obj is None: raise gr.Error("Please generate the 3D object first.")
    if global_transform_handle is None: raise gr.Error("Viser controls are not initialized.")

    bg_square_pil, crop_box, render_resolution = prepare_preview_background(bg_pil)

    clients = viser_server.get_clients()
    if not clients: raise gr.Error("Viser is not connected.")
    client = next(iter(clients.values()))

    res_color = render_gaussian_preview(
        gaussian_obj,
        transform_handle=global_transform_handle,
        camera=client.camera,
        render_resolution=render_resolution,
        device=device,
    )
    geometry_image = make_condition_image(res_color, target_res=target_res)
    final_composite, mask_final = make_composite_preview(bg_square_pil, res_color, crop_box)

    return (
        final_composite, 
        geometry_image,
        mask_final
    )

def prepare_generation_inputs(
    bg_pil: Image.Image,
    composite_pil: Image.Image,
    mask_pil: Image.Image,
    reference_image: Image.Image,
    geometry_image: Image.Image,
):
    target_res = MODEL_INPUT_RESOLUTION
    if bg_pil is None: raise gr.Error("Background image is missing.")
    if composite_pil is None or mask_pil is None or geometry_image is None:
        raise gr.Error("Please confirm the pose and preview the composite first.")
    if reference_image is None:
        raise gr.Error("Reference object image is missing.")

    print("[Flux Pipeline] Preparing Reference Image and Inpainting Inputs...")
    return build_generation_inputs(
        bg_pil=bg_pil,
        composite_pil=composite_pil,
        mask_pil=mask_pil,
        reference_image=reference_image,
        geometry_image=geometry_image,
        target_res=target_res,
    )

def step3_flux_inference(
    bg_pil: Image.Image,
    generation_inputs,
    seed: int,
    reference_guidance_scale: float,
    num_steps: int
):
    """
    Step 3: run Flux inpainting and paste the generated patch back to the source image.
    """
    target_res = MODEL_INPUT_RESOLUTION
    if bg_pil is None: raise gr.Error("Background image is missing.")
    if generation_inputs is None:
        raise gr.Error("Please confirm the pose and preview the composite first.")

    try:
        final_images = direct_pipeline(
            composite_image=generation_inputs.composite_image,
            inpaint_mask=generation_inputs.inpaint_mask,
            reference_image=generation_inputs.reference_image,
            geometry_image=generation_inputs.geometry_image,
            context_image=generation_inputs.context_image,
            seed=int(seed),
            guidance_scale=30,
            num_inference_steps=num_steps, 
            height=target_res,
            width=target_res,
            use_autocast=True,
            reference_guidance_scale=reference_guidance_scale
        )
        generated_patch = final_images[0]
    except Exception as e:
        raise gr.Error(f"Flux generation failed: {e}")

    return paste_generated_patch(bg_pil, generated_patch, generation_inputs)

def build_demo(viser_port):
    with gr.Blocks(title="DIRECT", css=custom_css, head=favicon_head_code + js_head_code) as demo:
        state_gaussian = gr.State()
        state_reference_image = gr.State()
        hidden_dims = gr.JSON(value=[1, 1], visible=False)

        state_geometry = gr.State()
        state_mask = gr.State()
        state_generation_inputs = gr.State()

        gr.HTML(
            f"""
            <div class="direct-header">
                <img src="data:image/png;base64,{LOGO_B64}" alt="DIRECT logo">
                <h1>Direct 3D-Aware Object Insertion via Decomposed Visual Proxies</h1>
            </div>
            <div class="direct-tutorial">
                <ol>
                    <li>Upload a background image and an object image.</li>
                    <li>Click <strong>Generate 3D &amp; Place in Scene</strong>. This lifts the object to 3D and displays it in the Viser viewer on the right.<span class="step-note"><strong>Note:</strong> The image-to-3D model is seed-sensitive, so try a different seed and generate again if the 3D result is not satisfactory.</span></li>
                    <li>Drag and rotate the object in the Viser viewer to choose the desired insertion location and pose, then click <strong>Confirm Pose &amp; Preview</strong>.</li>
                    <li>Click <strong>Generate</strong> to produce the final realistic insertion result!</li>
                </ol>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_id="setup_panel"):
                bg_input = gr.Image(
                    label="Background Image",
                    type="pil",
                    height=320,
                    format="png",
                    show_download_button=True,
                )
                img_input = gr.Image(
                    label="Object Image",
                    type="numpy",
                    height=320,
                    format="png",
                    show_download_button=True,
                )
                with gr.Group(elem_id="seed_control"):
                    seed_num = gr.Number(label="Seed", precision=0)
                    seed_random_btn = gr.Button("🎲", elem_id="seed_random_btn", min_width=38)
                btn_gen = gr.Button("Generate 3D & Place in Scene", variant="primary")

            with gr.Column(scale=2):
                viser_iframe_html = f"""
                <div id="viser_container_wrapper">
                    <iframe src="http://localhost:{viser_port}?t={int(time.time())}" ></iframe>
                </div>
                """
                viser_out = gr.HTML(label="Interactive Scene", value=viser_iframe_html)

        with gr.Row():
            btn_preview = gr.Button("Confirm Pose & Preview", variant="secondary")
            btn_generate = gr.Button("Generate!", variant="primary")

        with gr.Row():
            with gr.Accordion("🛠️ Generation Settings", open=True):
                with gr.Row():
                    reference_guidance_scale = gr.Slider(
                        label="Reference Guidance Scale",
                        minimum=1.0, maximum=5.0, value=2.0, step=0.1
                    )
                    step_slider = gr.Slider(
                        label="Inference Steps",
                        minimum=15, maximum=50, value=28, step=1
                    )

        with gr.Row():
            with gr.Column():
                out_composite = gr.Image(
                    label="Composite Preview",
                    type="pil",
                    height=500,
                    format="png",
                    show_download_button=True,
                )
            with gr.Column():
                out_final = gr.Image(
                    label="Final Result",
                    type="pil",
                    height=500,
                    format="png",
                    show_download_button=True,
                )

        gr.Markdown(
            r"""
---
📧 **Contact**
<br>
If you have any questions, please feel free to contact us at <b>jingbogong@mail.nankai.edu.cn</b>.
We are also actively improving DIRECT, and we welcome any failure cases or feedback encountered during use!
"""
        )

        # Event bindings
        demo.load(fn=random_seed, outputs=seed_num)
        seed_random_btn.click(fn=random_seed, outputs=seed_num)

        bg_input.change(fn=send_bg_to_viser, inputs=bg_input, outputs=hidden_dims) \
            .then(fn=None, inputs=[hidden_dims], outputs=None, js="(d) => updateViser(d)")

        img_input.upload(
            fn=on_upload_object,
            inputs=img_input,
            outputs=[img_input, state_reference_image]
        )

        btn_gen.click(
            fn=step1_generate_and_place_3d,
            inputs=[img_input, seed_num],
            outputs=[viser_out, state_gaussian]
        )

        btn_preview.click(
            fn=step2_preview_render,
            inputs=[bg_input, state_gaussian],
            outputs=[
                out_composite,
                state_geometry,
                state_mask
            ]
        ).then(
            fn=prepare_generation_inputs,
            inputs=[
                bg_input,
                out_composite,
                state_mask,
                state_reference_image,
                state_geometry,
            ],
            outputs=state_generation_inputs,
        )

        btn_generate.click(
            fn=step3_flux_inference,
            inputs=[
                bg_input,
                state_generation_inputs,
                seed_num,
                reference_guidance_scale,
                step_slider
            ],
            outputs=out_final
        )

    return demo


def main():
    args = parse_args()
    load_models()
    setup_viser_server(args.viser_port)
    demo = build_demo(args.viser_port)
    demo.launch(server_port=args.gradio_port)


if __name__ == "__main__":
    main()
