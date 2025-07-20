#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

python infer_micro_single.py \
--checkpoint "stable-diffusion-2" \
--input_img_path "imgs/2#Aircraft#7#UAV#16522310810_468dfa447a_o.jpg" \
--prompts ["Black professional camera drone with a high-definition camera mounted on a gimbal.","Three men beside a UAV"] \
--output_dir 'output/output-macro-single' \
--denoise_steps 1 \
--processing_res 1024 
