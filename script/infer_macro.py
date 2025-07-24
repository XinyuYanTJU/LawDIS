import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import json
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Use LawDIS in macro mode for language-guided DIS."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="stable-diffusion-2",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        default="data/DIS5K",
        help="Path to the input image folder."
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="DIS-TE4",
        help="Name of the DIS subset to process, e.g., DIS-VD, DIS-TE1, DIS-TE2, DIS-TE3, DIS-TE4."
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default='data/json',
        help="Path to the prompt json folder."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/output-macro", help="Output directory."
    )
    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=1024,
        help="Resolution of processing.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and dis predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )
    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )

    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    subset_name = args.subset_name
    prompt_dir = args.prompt_dir
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    half_precision = args.half_precision
    processing_res = args.processing_res
    match_input_res = True
    resample_method = args.resample_method
    seed = args.seed
    batch_size = args.batch_size


    # -------------------- Preparation --------------------
    # Output directories
    output_dir = os.path.join(output_dir, subset_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")


    prompt_path = os.path.join(prompt_dir, subset_name+'.json')
    with open(prompt_path, "r") as f:
            filenames = [
                (json.loads(line)['img'], json.loads(line)['prompt']) for line in f
            ]
    n_images = len(filenames)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        print(" Running with full precision (fp16)")
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        print(" Running with full precision (fp32)")
        variant = None

    pipe: LawDISMacroPipeline = LawDISMacroPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("enable_xformers_memory_efficient_attention")
    except ImportError:
        pass  # run without xformers
    

    pipe = pipe.to(device)

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}. "
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(filenames, desc="Estimating dis", leave=True, disable=True):
            # Read input image
            rgb_path = os.path.join(input_rgb_dir, filename[0])
            prompt = filename[1]
            input_image = Image.open(rgb_path)
            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Predict dis
            dis_pred = pipe(
                input_image,
                prompt,
                denoising_steps=denoise_steps,
                processing_res=processing_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator
            )

            # Save
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            dis_to_save = (dis_pred * 255.0).astype(np.uint8)
            png_save_path = os.path.join(output_dir, f"{rgb_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(dis_to_save).save(png_save_path, mode="L")
