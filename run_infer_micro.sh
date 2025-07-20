#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

python infer_micro.py \
--checkpoint "stable-diffusion-2" \
--input_rgb_dir "/path/to/your/DIS5K/" \
--subset_name "DIS-TE1" \
--init_seg_dir 'output/output-macro/' \
--output_dir "output/output-micro" \
--window_mode "auto" \
--denoise_steps 1 \
--processing_res 1024 
