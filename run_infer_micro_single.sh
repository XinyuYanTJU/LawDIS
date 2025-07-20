#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

python infer_micro_single.py \
--checkpoint "stable-diffusion-2" \
--input_img_path "imgs/2#Aircraft#7#UAV#16522310810_468dfa447a_o.jpg" \
--init_seg_dir 'output/output-macro/' \
--output_dir "output/output-micro" \
--window_mode "auto" \
--denoise_steps 1 \
--processing_res 1024 
