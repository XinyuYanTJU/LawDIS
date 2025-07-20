#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

python infer_macro.py \
--checkpoint "stable-diffusion-2" \
--input_rgb_dir "/path/to/your/DIS5K/" \
--subset_name "DIS-TE1" \
--prompt_dir 'json' \
--output_dir "output/output-macro" \
--denoise_steps 1 \
--processing_res 1024 
