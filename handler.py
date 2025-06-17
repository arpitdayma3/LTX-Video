import os
import uuid
import yaml
import torch
import runpod
import requests
import io
from PIL import Image
import q8_kernels # Added for FP8 support

# Adjust imports based on actual functions needed from inference.py
from inference import (
    create_ltx_video_pipeline,
    load_media_file,
    prepare_conditioning,
    get_unique_filename,
    calculate_padding,
    seed_everething,
    get_device,
    LTXMultiScalePipeline, # Added
    create_latent_upsampler # Added
)
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, SkipLayerStrategy
from pathlib import Path

# Global variable to store the initialized pipeline
PIPELINE = None
CONFIG = None

def _load_config():
    global CONFIG
    if CONFIG is None:
        config_path = "configs/ltxv-13b-0.9.7-distilled.yaml"
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Pipeline config file {config_path} does not exist")
        with open(config_path, "r") as f:
            CONFIG = yaml.safe_load(f)
    return CONFIG

def _initialize_pipeline():
    global PIPELINE
    if PIPELINE is None:
        config = _load_config()
        device = get_device() # Use device detection from inference.py

        # Model paths and downloads (logic adapted from inference.py's infer function)
        models_dir = "MODEL_DIR" # Or any other preferred local cache directory
        Path(models_dir).mkdir(parents=True, exist_ok=True)

        # Use the specific FP8 model
        ltxv_model_name_or_path = "ltxv-13b-0.9.7-distilled-fp8.safetensors"
        # Update config to use this model path for consistency if other parts of config are used
        config["checkpoint_path"] = ltxv_model_name_or_path

        if not os.path.isfile(ltxv_model_name_or_path):
            # Check if it's in MODEL_DIR first (if downloaded manually or by another run)
            local_model_path_check = Path(models_dir) / ltxv_model_name_or_path
            if local_model_path_check.is_file():
                 ltxv_model_path = str(local_model_path_check)
            else:
                from huggingface_hub import hf_hub_download # Local import
                print(f"Downloading model {ltxv_model_name_or_path} from Hugging Face Hub...")
                ltxv_model_path = hf_hub_download(
                    repo_id="Lightricks/LTX-Video", # Assuming same repo_id
                    filename=ltxv_model_name_or_path,
                    local_dir=models_dir,
                    repo_type="model",
                )
                print(f"Model downloaded to {ltxv_model_path}")
        else:
            # This case implies ltxv_model_name_or_path was a full path already and exists
            ltxv_model_path = ltxv_model_name_or_path

        # Ensure the config reflects the actual path being used, especially if it was downloaded
        config["checkpoint_path"] = ltxv_model_path

        spatial_upscaler_model_name_or_path = config.get("spatial_upscaler_model_path")
        if spatial_upscaler_model_name_or_path and not os.path.isfile(spatial_upscaler_model_name_or_path):
            from huggingface_hub import hf_hub_download # Local import
            spatial_upscaler_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=spatial_upscaler_model_name_or_path,
                local_dir=models_dir,
                repo_type="model",
            )
        else:
            spatial_upscaler_model_path = spatial_upscaler_model_name_or_path
        if spatial_upscaler_model_path:
             config["spatial_upscaler_model_path"] = spatial_upscaler_model_path


        # Prepare pipeline arguments (adapted from inference.py)
        precision = config["precision"]
        text_encoder_model_name_or_path = config["text_encoder_model_name_or_path"]
        sampler = config["sampler"]
        # Assuming prompt enhancement is not enabled by default or handled by config
        enhance_prompt = False
        prompt_enhancer_image_caption_model_name_or_path = config.get("prompt_enhancer_image_caption_model_name_or_path")
        prompt_enhancer_llm_model_name_or_path = config.get("prompt_enhancer_llm_model_name_or_path")


        pipeline = create_ltx_video_pipeline(
            ckpt_path=config["checkpoint_path"], # Use updated local path
            precision=precision,
            text_encoder_model_name_or_path=text_encoder_model_name_or_path,
            sampler=sampler,
            device=device,
            enhance_prompt=enhance_prompt, # Keep false for now unless explicitly requested
            prompt_enhancer_image_caption_model_name_or_path=prompt_enhancer_image_caption_model_name_or_path,
            prompt_enhancer_llm_model_name_or_path=prompt_enhancer_llm_model_name_or_path,
        )

        if config.get("pipeline_type") == "multi-scale":
            if not config.get("spatial_upscaler_model_path"):
                raise ValueError("Spatial upscaler model path is required for multi-scale rendering.")
            latent_upsampler = create_latent_upsampler(config["spatial_upscaler_model_path"], pipeline.device)
            pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

        PIPELINE = pipeline
    return PIPELINE

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def handler(event):
    pipeline = _initialize_pipeline()
    config = _load_config() # Get the loaded global config

    job_input = event["input"]

    # Extract and validate inputs
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Prompt is required."}

    image_urls = job_input.get("image_urls")
    if not image_urls or not isinstance(image_urls, list) or len(image_urls) != 2:
        return {"error": "image_urls must be a list of two URLs (first and last frame)."}

    height = job_input.get("height", 704) # Default from inference.py
    width = job_input.get("width", 1216)  # Default from inference.py
    num_frames = job_input.get("num_frames", 121) # Default from inference.py
    seed = job_input.get("seed", 171198) # Default from inference.py
    frame_rate = job_input.get("frame_rate", 30)
    negative_prompt = job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    image_cond_noise_scale = job_input.get("image_cond_noise_scale", 0.15) # Default from inference.py

    # RunPod specific output path
    output_dir = Path(f"/tmp/outputs/{uuid.uuid4()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everething(seed)
    device = get_device()

    # Download keyframe images
    temp_image_paths = []
    for i, url in enumerate(image_urls):
        temp_path = output_dir / f"keyframe_{i}.png" # Assuming png, could be jpg etc.
        downloaded_path = download_image(url, temp_path)
        if not downloaded_path:
            return {"error": f"Failed to download image from {url}"}
        temp_image_paths.append(str(downloaded_path))

    # Prepare conditioning items
    conditioning_media_paths = temp_image_paths
    conditioning_start_frames = [0, num_frames - 1] # First image at frame 0, second at last frame
    conditioning_strengths = [1.0, 1.0] # Default strength

    # Adjust dimensions for padding (from inference.py)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
    padding = calculate_padding(height, width, height_padded, width_padded)

    conditioning_items_list = prepare_conditioning(
        conditioning_media_paths=conditioning_media_paths,
        conditioning_strengths=conditioning_strengths,
        conditioning_start_frames=conditioning_start_frames,
        height=height, # Original height for loading
        width=width,   # Original width for loading
        num_frames=num_frames_padded, # Padded frames for pipeline
        padding=padding,
        pipeline=pipeline,
    )

    # STG mode from config (from inference.py)
    stg_mode_str = config.get("stg_mode", "attention_values")
    if stg_mode_str.lower() == "stg_av" or stg_mode_str.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode_str.lower() == "stg_as" or stg_mode_str.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode_str.lower() == "stg_r" or stg_mode_str.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode_str.lower() == "stg_t" or stg_mode_str.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode_str}")

    # Prepare pipeline call arguments (merging config and job inputs)
    pipeline_call_args = {**config} # Start with base config
    pipeline_call_args.update({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height_padded,
        "width": width_padded,
        "num_frames": num_frames_padded,
        "frame_rate": frame_rate,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "output_type": "pt",  # Get tensors first
        "conditioning_items": conditioning_items_list,
        "skip_layer_strategy": skip_layer_strategy,
        "image_cond_noise_scale": image_cond_noise_scale,
        "device": device,
        "offload_to_cpu": False, # Assuming RunPod has enough VRAM
        "enhance_prompt": False, # Defaulting to false
        # Ensure these are not passed if None or not applicable
        "media_items": None,
        "vae_per_channel_normalize": True, # Common default
        "mixed_precision": (config.get("precision") == "mixed_precision"),
    })

    # Remove keys from pipeline_call_args that are not direct inputs to pipeline __call__
    # These are part of the main config file but not direct pipeline() args
    keys_to_remove_from_call = [
        "checkpoint_path", "spatial_upscaler_model_path", "text_encoder_model_name_or_path",
        "sampler", "prompt_enhancement_words_threshold",
        "prompt_enhancer_image_caption_model_name_or_path", "prompt_enhancer_llm_model_name_or_path",
        "stg_mode", "pipeline_type" # pipeline_type is used to wrap pipeline, not pass to it
    ]
    for key_to_remove in keys_to_remove_from_call:
        pipeline_call_args.pop(key_to_remove, None)


    # Call the pipeline
    try:
        print(f"Calling pipeline with args: { {k: v for k, v in pipeline_call_args.items() if k not in ['generator']} }") # Avoid printing generator object
        generated_tensors = pipeline(**pipeline_call_args).images
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Pipeline execution failed: {str(e)}"}


    # Crop padded images (from inference.py)
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    unpad_bottom = -pad_bottom if pad_bottom != 0 else generated_tensors.shape[3]
    unpad_right = -pad_right if pad_right != 0 else generated_tensors.shape[4]

    generated_tensors = generated_tensors[:, :, :num_frames, pad_top:unpad_bottom, pad_left:unpad_right]

    # Save the output video
    # Assuming one video per prompt for serverless
    video_np = generated_tensors[0].permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).astype("uint8")

    # Use get_unique_filename to place it in the temp output_dir
    # The base name can be simpler as it's in a UUID folder
    output_filename_obj = get_unique_filename(
        base="video_output",
        ext=".mp4",
        prompt=prompt[:30], # Shortened prompt for filename
        seed=seed,
        resolution=(height, width, num_frames), # Original resolution
        dir=output_dir
    )
    output_video_path = str(output_filename_obj)

    try:
        import imageio # Local import for safety
        with imageio.get_writer(output_video_path, fps=frame_rate) as video_writer:
            for frame_idx in range(video_np.shape[0]):
                video_writer.append_data(video_np[frame_idx])
    except Exception as e:
        print(f"Error saving video: {e}")
        return {"error": f"Failed to save video: {str(e)}"}

    # For now, return the path. User can configure RunPod to upload this.
    # Or, could implement S3 upload here if credentials are provided via env vars.
    return {"video_path": output_video_path}


if __name__ == "__main__":
    # This part is for local testing if needed, not used by RunPod directly
    # print("Handler script started. Initializing pipeline for local test...")
    # _initialize_pipeline()
    # print("Pipeline initialized.")
    # # Example local test event (replace with actual URLs and prompt)
    # test_event = {
    #     "input": {
    #         "prompt": "A futuristic cityscape",
    #         "image_urls": [
    #             "https://example.com/first_frame.jpg", # Replace with a real, small image URL
    #             "https://example.com/last_frame.jpg"  # Replace with a real, small image URL
    #         ],
    #         "num_frames": 17 # Padded will be 17
    #     }
    # }
    # print(f"Test event: {test_event}")
    # result = handler(test_event)
    # print(f"Handler result: {result}")

    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
