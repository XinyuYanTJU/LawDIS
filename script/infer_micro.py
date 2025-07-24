import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import numpy as np
from glob import glob
import torch
from PIL import Image
from tqdm.auto import tqdm
from lawdis.lawdis_micro_pipeline import LawDISMicroPipeline
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def get_bounding_boxes(image):
    # Detect edges
    edges = cv2.Canny(image, 50, 150)
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    
    # Remove background
    stats = stats[1:]  # Remove background statistics
    centroids = centroids[1:]  # Remove background centroids
    
    # Calculate side lengths
    side_lengths = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    
    return edges, stats, centroids, side_lengths

def calculate_white_black_ratio(image, x1, y1, x2, y2):
    region = image[y1:y2, x1:x2]
    total_pixels = region.size
    white_pixels = np.sum(region > 250)
    black_pixels = np.sum(region < 5)
    
    white_ratio = white_pixels / total_pixels
    black_ratio = black_pixels / total_pixels
    gray_ratio = 1 - white_ratio - black_ratio
    
    return white_ratio, black_ratio, gray_ratio

def create_bounding_box(image, centroid, size=1024):
    x, y = centroid
    x1 = max(int(x - size // 2), 0)
    y1 = max(int(y - size // 2), 0)
    x2 = min(int(x + size // 2), image.shape[1])
    y2 = min(int(y + size // 2), image.shape[0])
    return x1, y1, x2, y2

def get_coordinates_and_side_lengths(image, patch_size=1024, max_iterations=50):
    edges, stats, centroids, side_lengths = get_bounding_boxes(image)
    coordinates_with_sizes = []
    iteration_count = 0
    
    while len(side_lengths) > 0 and iteration_count < max_iterations:
        iteration_count += 1
        # Find the largest connected component by side length
        max_index = np.argmax(side_lengths)
        centroid = centroids[max_index]
        x1, y1, x2, y2 = create_bounding_box(image, centroid, patch_size)
        
        # Calculate the ratio of white and black pixels in the bounding box
        white_ratio, black_ratio, gray_ratio = calculate_white_black_ratio(image, x1, y1, x2, y2)
        
        # Skip the box if white or black pixel ratio exceeds threshold
        if white_ratio > 0.9 or black_ratio > 0.95:
            # Update mask
            mask = np.zeros_like(edges, dtype=np.uint8)
            mask[y1:y2, x1:x2] = edges[y1:y2, x1:x2]
            used_edges = np.zeros_like(edges, dtype=np.uint8)
            used_edges[mask > 0] = 255
            
            # Get remaining edges
            remaining_edges = cv2.bitwise_and(edges, cv2.bitwise_not(used_edges))
            
            # Update edges and component info
            edges = remaining_edges
            stats, centroids, side_lengths = get_bounding_boxes(edges)[1:]
            continue  
        
        # Record coordinates and patch size
        width = x2 - x1
        height = y2 - y1
        coordinates_with_sizes.append(((x1, y1), (width, height)))
        
        # Update mask
        mask = np.zeros_like(edges, dtype=np.uint8)
        mask[y1:y2, x1:x2] = edges[y1:y2, x1:x2]
        used_edges = np.zeros_like(edges, dtype=np.uint8)
        used_edges[mask > 0] = 255
        
        # Get remaining edges
        remaining_edges = cv2.bitwise_and(edges, cv2.bitwise_not(used_edges))
        
        # Update edges and component info
        edges = remaining_edges
        stats, centroids, side_lengths = get_bounding_boxes(edges)[1:]
    return coordinates_with_sizes


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
        "--init_seg_dir",
        type=str,
        default='output/output-macro/',
        help="Path to the initial segmentation maps folder."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/output-micro", help="Output directory."
    )
    parser.add_argument(
    "--window_mode",
    type=str,
    choices=["auto", "semi-auto", "manual"],
    default="auto",
    help=(
        "Window selection mode."
        "'auto': Automatically select windows based on object edges in the initial segmentation map."
        "'semi-auto': Simulate user-guided window selection using object edges from the ground-truth (GT) segmentation."
        "'manual': User manually specifies window regions (e.g., via UI or coordinate input)."
    )
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
        default=1,
        help="Inference batch size.",
    )

    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    subset_name = args.subset_name
    init_seg_dir = args.init_seg_dir
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    half_precision = args.half_precision
    processing_res = args.processing_res
    match_input_res = True
    resample_method = args.resample_method
    seed = args.seed
    batch_size = args.batch_size
    window_mode = args.window_mode


    # -------------------- Preparation --------------------
    # Output directories
    output_dir = os.path.join(output_dir, window_mode, subset_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")


    input_rgb_dir = os.path.join(input_rgb_dir, subset_name,'im')
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    # Remove already processed images
    rgb_filename_list = [
        f for f in rgb_filename_list 
        if not os.path.exists(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(f))[0]}.png"))
    ] 
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)
        
    init_seg_dir = os.path.join(init_seg_dir, subset_name)

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

    pipe: LawDISMicroPipeline = LawDISMicroPipeline.from_pretrained(
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

        for rgb_path in tqdm(rgb_filename_list, desc="Estimating dis", leave=True, disable=True):
            # Read input image
            input_image = Image.open(rgb_path)
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            init_seg_path = os.path.join(init_seg_dir, f"{rgb_name_base}.png")
            init_seg = Image.open(init_seg_path)
            
            if window_mode == "auto":
                init_seg_np = np.array(init_seg)
                patch_coordinates_and_sizes = get_coordinates_and_side_lengths(init_seg_np)  
                
            elif window_mode == "semi-auto":
                gt_path = rgb_path.replace("/im/", "/gt/").replace(".jpg", ".png")
                gt = Image.open(gt_path)
                gt_np = np.array(gt)
                patch_coordinates_and_sizes = get_coordinates_and_side_lengths(gt_np)  
                
            else:
                init_seg_np = np.array(init_seg)
                h, w = init_seg_np.shape[:2]
                dpi = 100
                fig, ax = plt.subplots(1, figsize=(w / dpi, h / dpi))
                ax.imshow(init_seg_np, cmap='gray')
                plt.title("Drag to select multiple regions. Press 'q' to exit.")
                rectangles = []
                
                def onselect(eclick, erelease):
                    x1, y1 = int(eclick.xdata), int(eclick.ydata)
                    x2, y2 = int(erelease.xdata), int(erelease.ydata)
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    rectangles.append(((x, y), (w, h)))
                    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    fig.canvas.draw()
                
                toggle_selector = RectangleSelector(
                    ax, onselect,
                    useblit=True,
                    button=[1],  
                    minspanx=5, minspany=5,
                    spancoords='pixels', interactive=True
                )
                
                def on_key(event):
                    if event.key == 'q':
                        plt.close()
                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()
            
                filtered_rectangles = []
                for (xy, wh) in rectangles:
                    (x, y), (w, h) = xy, wh
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    white_ratio, black_ratio, gray_ratio = calculate_white_black_ratio(init_seg_np, x1, y1, x2, y2)
                    if white_ratio <= 0.9 and black_ratio <= 0.95:
                        filtered_rectangles.append(((x, y), (w, h)))
                patch_coordinates_and_sizes = filtered_rectangles
            
            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Predict dis
            dis_pred = pipe(
                input_image,
                init_seg,
                patch_coordinates_and_sizes,
                denoising_steps=denoise_steps,
                processing_res=processing_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator
            )

            # Save
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            png_save_path = os.path.join(output_dir, f"{rgb_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(dis_pred).save(png_save_path, mode="L")
