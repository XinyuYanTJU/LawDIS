import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from lawdis.lawdis_macro_pipeline import LawDISMacroPipeline

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

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
        "--input_img_path",
        type=str,
        default="data/imgs/2#Aircraft#7#UAV#16522310810_468dfa447a_o.jpg",
        help="Path to the input image."
    )

    parser.add_argument(
    "--prompts",
    nargs="+",
    type=str,
    default=[
        "Black professional camera drone with a high-definition camera mounted on a gimbal.",
        "Three men beside a UAV"
    ],
    help="Prompts."
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="output/output-macro-single", help="Output directory."
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
        help="Inference batch size.",
    )

    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    input_img_path = args.input_img_path
    prompts = args.prompts
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
    output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    n_prompts = len(prompts)
    if n_prompts > 0:
        logging.info(f"Found {n_prompts} prompts")
    else:
        logging.error(f"No n_prompts.")
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

      
        for idx, prompt in enumerate(tqdm(prompts, desc="Estimating dis", leave=True, disable=False)):
            # Read input image
            rgb_path = input_img_path
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
            png_save_path = os.path.join(output_dir, f"{rgb_name_base}_{idx}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(dis_to_save).save(png_save_path, mode="L")
