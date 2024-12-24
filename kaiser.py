import os
import shutil
import numpy as np
import torch
import imageio
from typing import Tuple, List, Literal
from PIL import Image
from easydict import EasyDict as edict
import gradio as gr

from gradio_litmodel3d import LitModel3D

# Your local Trellis imports...
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

# Constants
MAX_SEED = np.iinfo(np.int32).max
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(THIS_DIR, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

########################################################################
# Utility Functions
########################################################################

def start_session(req: gr.Request):
    """Create a user-specific temporary folder on session start."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    """Clean up the user-specific temporary folder when session ends."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Placeholder for any image preprocessing logic you want.
    In the original code, we rely on pipeline.preprocess_image().
    """
    return pipeline.preprocess_image(image)

def preprocess_images(images: List[Image.Image]) -> List[Image.Image]:
    """
    Preprocess a list of input images.
    """
    processed_images = [pipeline.preprocess_image(img) for img in images]
    return processed_images

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    """
    Convert Gaussian + mesh into a dictionary that can be serialized by Gradio states.
    """
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    """
    Convert dictionary state back into Gaussian + mesh.
    """
    # Rebuild Gaussian
    gs_dict = state['gaussian']
    gs = Gaussian(
        aabb=gs_dict['aabb'],
        sh_degree=gs_dict['sh_degree'],
        mininum_kernel_size=gs_dict['mininum_kernel_size'],
        scaling_bias=gs_dict['scaling_bias'],
        opacity_bias=gs_dict['opacity_bias'],
        scaling_activation=gs_dict['scaling_activation'],
    )
    gs._xyz = torch.tensor(gs_dict['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(gs_dict['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(gs_dict['_scaling'], device='cuda')
    gs._rotation = torch.tensor(gs_dict['_rotation'], device='cuda')
    gs._opacity = torch.tensor(gs_dict['_opacity'], device='cuda')

    # Rebuild mesh
    mesh_dict = state['mesh']
    mesh = edict(
        vertices=torch.tensor(mesh_dict['vertices'], device='cuda'),
        faces=torch.tensor(mesh_dict['faces'], device='cuda'),
    )
    return gs, mesh

def get_seed(randomize_seed: bool, seed: int) -> int:
    """Return a random seed if requested, else return the user-specified seed."""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

########################################################################
# Model / Pipeline Inference
########################################################################

def image_to_3d(
    image: Image.Image,
    multiimages: List[Image.Image],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert an image (or multiple images) to a 3D model using the Trellis pipeline.
    Returns a state dict and path to a rendered video.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))

    # Prepare pipeline arguments
    sparse_params = {"steps": ss_sampling_steps, "cfg_strength": ss_guidance_strength}
    slat_params   = {"steps": slat_sampling_steps, "cfg_strength": slat_guidance_strength}

    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params=sparse_params,
            slat_sampler_params=slat_params,
        )
    else:
        outputs = pipeline.run_multi_image(
            multiimages,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params=sparse_params,
            slat_sampler_params=slat_params,
            mode=multiimage_algo,
        )

    # Render a 120-frame video of the color and geometry side by side
    gauss_obj = outputs['gaussian'][0]
    mesh_obj = outputs['mesh'][0]

    video_color = render_utils.render_video(gauss_obj, num_frames=120)['color']
    video_geo   = render_utils.render_video(mesh_obj, num_frames=120)['normal']
    merged_video = [np.concatenate([video_color[i], video_geo[i]], axis=1)
                    for i in range(len(video_color))]

    # Save video
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, merged_video, fps=15)

    # Pack outputs into a Gradio state
    state = pack_state(gauss_obj, mesh_obj)
    torch.cuda.empty_cache()
    return state, video_path

def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Convert the stored Gaussian + Mesh to a glTF (GLB) file.
    Returns the path twice: (for gr.Model3D, for gr.DownloadButton).
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)

    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, 
                                      texture_size=texture_size, verbose=False)

    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path

def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Save the Gaussian representation to a PLY file for debugging or advanced usage.
    Returns the path twice: (for gr.Model3D, for gr.DownloadButton).
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path

########################################################################
# Multi-Image Example Helpers
########################################################################

def prepare_multi_example() -> List[Image.Image]:
    """
    Read multi-image examples (3 views each) and horizontally stitch them for the
    Gradio examples gallery. Then the user can split them up later.
    Adjust the path 'assets/example_multi_image' to your own if needed.
    """
    multi_dir = os.path.join(THIS_DIR, "assets", "example_multi_image")
    if not os.path.exists(multi_dir):
        return []

    # We expect file names like: "case1_1.png", "case1_2.png", "case1_3.png" etc.
    all_files = [f for f in os.listdir(multi_dir) if f.endswith('.png')]
    multi_cases = list(set(file.split('_')[0] for file in all_files))

    example_images = []
    for case in multi_cases:
        # Collect 3 images: e.g. case_1.png, case_2.png, case_3.png
        images_for_case = []
        for i in range(1, 4):
            filename = f"{case}_{i}.png"
            img_path = os.path.join(multi_dir, filename)
            if not os.path.exists(img_path):
                break
            img = Image.open(img_path)
            # Resize to a consistent height
            w, h = img.size
            target_h = 512
            new_w = int(w / h * target_h)
            img = img.resize((new_w, target_h))
            images_for_case.append(np.array(img))

        if len(images_for_case) == 3:
            # Join horizontally
            combined = np.concatenate(images_for_case, axis=1)
            example_images.append(Image.fromarray(combined))
    return example_images

def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split a single horizontally stitched image into three separate images
    for multi-view input.
    """
    image_array = np.array(image)
    # fallback: 3-split by width
    width = image_array.shape[1]
    split_width = width // 3
    sub_images = []
    for i in range(3):
        sub_im = image_array[:, i * split_width:(i+1)*split_width, :]
        sub_images.append(Image.fromarray(sub_im))
    return [preprocess_image(img) for img in sub_images]

########################################################################
# Gradio Interface
########################################################################

# Initialize your pipeline once here:
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

#############################################################################
# Create and Configure the Dark Theme
#############################################################################

dark_theme = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
)

# Use only valid .set parameters based on your docstring
dark_theme = dark_theme.set(
    body_background_fill="#121212",  # Dark overall background
    body_text_color="white",         # White text
    block_background_fill="#1E1E1E", # Dark fill for blocks
)


#############################################################################
# Build the Interface
#############################################################################
with gr.Blocks(theme=dark_theme) as demo:

    gr.Markdown(
        """
        <h1 style="color:white;text-align:center;">
          TRELLIS 2D-to-3D Conversion (Dark Theme)
        </h1>
        <p style="color:lightgray;text-align:center;">
          By <strong>David Kaiser</strong> (davidcorykaiser@gmail.com)
        </p>
        <p style="color:lightgray;">
          Convert one or multiple images into a 3D asset using the TRELLIS pipeline. 
          Optionally remove backgrounds with alpha channels or user-provided masks.
        </p>
        """,
        elem_id="title"
    )

    with gr.Row():
        with gr.Column():
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                    image_prompt = gr.Image(
                        label="Image Prompt",
                        format="png",
                        image_mode="RGBA",
                        type="pil",
                        height=300,
                        elem_id="single-image",
                    )
                with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(
                        label="Multi-View Images",
                        format="png",
                        type="pil",
                        height=300,
                        columns=3,
                        elem_id="multi-gallery",
                    )
                    gr.Markdown(
                        """
                        <span style="color:lightgray;">
                        Input different views of the same object, each image representing 
                        a unique perspective. This is an experimental algorithm and 
                        may not produce the best results for highly dissimilar images.
                        </span>
                        """
                    )

            # Generation Settings
            with gr.Accordion(label="Generation Settings", open=True):
                randomize_seed = gr.Checkbox(
                    label="Randomize Seed", value=True, 
                    info="Enable to get a random seed each time."
                )
                seed = gr.Slider(
                    minimum=0, 
                    maximum=MAX_SEED, 
                    label="Seed", 
                    value=0, 
                    step=1,
                    info="If Randomize Seed is disabled, the generation will be deterministic."
                )

                gr.Markdown("**Stage 1: Sparse Structure Generation**", elem_id="stage1-label")
                ss_guidance_strength = gr.Slider(
                    0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1
                )
                ss_sampling_steps = gr.Slider(
                    1, 50, label="Sampling Steps", value=12, step=1
                )

                gr.Markdown("**Stage 2: Structured Latent Generation**", elem_id="stage2-label")
                slat_guidance_strength = gr.Slider(
                    0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1
                )
                slat_sampling_steps = gr.Slider(
                    1, 50, label="Sampling Steps", value=12, step=1
                )

                multiimage_algo = gr.Radio(
                    choices=["stochastic", "multidiffusion"], 
                    label="Multi-Image Algorithm", 
                    value="stochastic",
                    info="Choose how the pipeline fuses multiple views."
                )

            generate_btn = gr.Button("Generate 3D Model", variant="primary")

            # GLB Extraction Settings
            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(
                    0.9, 0.98, label="Simplify", value=0.95, step=0.01
                )
                texture_size = gr.Slider(
                    512, 2048, label="Texture Size", value=1024, step=512
                )

            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian (PLY)", interactive=False)

            gr.Markdown(
                """
                <span style="color:lightgray;">
                Gaussian file can be quite large (~50MB), so it may take a while 
                to display or download.
                </span>
                """
            )

        # Right Column: Video + 3D Model
        with gr.Column():
            video_output = gr.Video(
                label="Generated 3D Preview (split: color | normals)",
                autoplay=True,
                loop=True,
                height=300,
                elem_id="video-output",
            )
            model_output = LitModel3D(
                label="3D Model Preview (GLB or PLY)",
                exposure=10.0,
                height=400,
                elem_id="model-3d-preview",
            )

            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)

    # Hidden states
    is_multiimage = gr.State(False)
    output_buf = gr.State()

    ####################################################################
    # Examples
    ####################################################################
    with gr.Row(visible=True) as single_image_example:
        gr.Markdown("<h3 style='color:white;'>Single Image Examples</h3>")
        examples = gr.Examples(
            examples=[
                os.path.join("assets", "example_image", img)
                for img in os.listdir(os.path.join(THIS_DIR, "assets", "example_image"))
            ],
            inputs=[image_prompt],
            outputs=[image_prompt],
            fn=preprocess_image,
            run_on_click=True,
            examples_per_page=6,
            elem_id="single-examples",
        )

    with gr.Row(visible=False) as multiimage_example:
        gr.Markdown("<h3 style='color:white;'>Multi-Image Examples</h3>")
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            outputs=[multiimage_prompt],
            fn=split_image,
            run_on_click=True,
            examples_per_page=3,
            elem_id="multi-examples",
        )

    ####################################################################
    # Gradio Event Wiring
    ####################################################################
    demo.load(start_session)
    demo.unload(end_session)

    def activate_single_tab():
        return (False, gr.Row.update(visible=True), gr.Row.update(visible=False))

    def activate_multi_tab():
        return (True, gr.Row.update(visible=False), gr.Row.update(visible=True))

    single_image_input_tab.select(fn=activate_single_tab, 
                                  outputs=[is_multiimage, single_image_example, multiimage_example])
    multiimage_input_tab.select(fn=activate_multi_tab, 
                                outputs=[is_multiimage, single_image_example, multiimage_example])

    # Handle uploads
    image_prompt.upload(preprocess_image, inputs=[image_prompt], outputs=[image_prompt])
    multiimage_prompt.upload(preprocess_images, inputs=[multiimage_prompt], outputs=[multiimage_prompt])

    # Generate button logic
    generate_btn.click(
        fn=get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed]
    ).then(
        fn=image_to_3d,
        inputs=[
            image_prompt,
            multiimage_prompt,
            is_multiimage,
            seed,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            multiimage_algo
        ],
        outputs=[output_buf, video_output],
    ).then(
        fn=lambda: (gr.Button.update(interactive=True), gr.Button.update(interactive=True)),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # Clear button logic
    video_output.clear(
        fn=lambda: (gr.Button.update(interactive=False), gr.Button.update(interactive=False)),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # Extract GLB
    extract_glb_btn.click(
        fn=extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        fn=lambda: gr.Button.update(interactive=True),
        outputs=[download_glb],
    )

    # Extract Gaussian
    extract_gs_btn.click(
        fn=extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        fn=lambda: gr.Button.update(interactive=True),
        outputs=[download_gs],
    )

    # Clear 3D model display
    model_output.clear(
        fn=lambda: gr.Button.update(interactive=False),
        outputs=[download_glb],
    )

# Run if called directly
if __name__ == "__main__":
    demo.launch()
