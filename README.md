# LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation (ICCV 2025)

<div align='center'>
    <a href='https://scholar.google.com/citations?user=bYLzUgYAAAAJ&hl' target='_blank'><strong>Xinyu Yan</strong></a><sup> 1,2,6</sup>,&thinsp;
    <a href='https://sunmeijun.github.io/' target='_blank'><strong>Meijun Sun</strong></a><sup> 1,2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=oaxKYKUAAAAJ' target='_blank'><strong>Ge-Peng Ji</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=zvaeYnUAAAAJ&hl=zh-CN&oi=ao' target='_blank'><strong>Fahad Shahbaz Khan</strong></a><sup> 6</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=zh-CN&oi=ao' target='_blank'><strong>Salman Khan</strong></a><sup> 6</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ&hl=zh-CN&oi=ao' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 4,5*</sup>
</div>

<div align='center'>
    <sup>1 </sup>Tianjin University&ensp;  <sup>2 </sup>Tianjin Key Laboratory of Machine Learning&ensp;  <sup>3 </sup>Australian National University&ensp; 
    <br />
    <sup>4 </sup>Nankai Institute of Advanced Research (SHENZHEN FUTIAN)&ensp;  <sup>5 </sup>Nankai University&ensp;  <sup>6 </sup>MBZUAI&ensp; 
</div>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://iccv.thecvf.com/virtual/2025/poster/2103'><img src='https://img.shields.io/badge/Conference-Paper-red'></a>&ensp; 
  <a href='https://arxiv.org/abs/2508.01152'><img src='https://img.shields.io/badge/arXiv-Paper-red'></a>&ensp; 
  <a href='https://github.com/XinyuYanTJU/LawDIS/blob/main/LawDIS_Chinese_version.pdf'><img src='https://img.shields.io/badge/中文版-Paper-red'></a>&ensp; 
  <a href='https://github.com/XinyuYanTJU/LawDIS'><img src='https://img.shields.io/badge/Page-Project-red'></a>&ensp; 
  <a href='https://drive.google.com/drive/folders/1cLmMm2PIrZ00lWuz2EvudNH-7zg6Cn9w?usp=sharing'><img src='https://img.shields.io/badge/GDrive-Stuff-green'></a>&ensp; 
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-yellow'></a>&ensp; 
</div>

<br>

⭐ Let's have a quick recap of our main idea. **Full-screen viewing is recommended for better visual details.**

https://github.com/user-attachments/assets/7d4c33a5-eff9-4474-a454-bff2b6ec78fa



We present LawDIS, a language-window-based controllable dichotomous image segmentation (DIS) framework. It supports two forms of user control: generating an initial mask based on user-provided language prompts, and enabling flexible refinement of user-defined regions (i.e., size-adjustable windows) within initial masks. 



## 🚀 1. Features

The following is an introductory video of our work:

https://github.com/user-attachments/assets/b2327310-0259-4b7a-95dd-fd4e881900fb



- **Framework innovation.** We recast the DIS task as an image-conditioned mask generation problem within a latent diffusion model. This enables LawDIS to seamlessly integrate both macro and micro user controls under a unified model and a shared set of parameters.

- **Dual control modes.** LawDIS employs a mode switcher to coordinate two distinct control modes. In macro mode, a language-controlled segmentation strategy (LS) generates an initial mask guided by user prompts. In micro mode, a window-controlled refinement strategy (WR) supports unlimited refinements on user-specified regions via size-adjustable local windows, enabling precise delineation of fine structures.

- **Flexible adaptation.** LS and WR can function independently or in collaboration. Their joint use meets high-accuracy personalized demands, while the micro mode (WR) alone can serve as a general-purpose post-refinement tool to enhance outputs from any segmentation model.

- **Superior performance.** Extensive evaluations on the DIS5K benchmark demonstrate that LawDIS consistently outperforms 11 state-of-the-art methods. Compared to the second-best model MVANet, LawDIS achieves a 3.6% $F_\beta^\omega$ improvement using LS alone, and a 4.6% gain when combining both LS and WR on DIS-TE.

## 📢 2. News

> [!note]
> The future directions of this project include more precise prompt control and improved efficiency. We warmly invite all potential collaborators to contribute to making LawDIS more accessible and practical. If you are interested in collaboration or have any questions about our paper, feel free to contact us via email (xinyuyan@tju.edu.cn & gepengai.ji@gmail.com). If you are using our code for your research, please cite this paper ([BibTeX](#-7-citations)).

- **2025.07** We have open-sourced the **core code** of LawDIS!
- **2025.06** 🎉 Our paper has been accepted by **[ICCV 2025, Honolulu, Hawai'i](https://iccv.thecvf.com/)**!  


## 🛠️ 3. Setup

### 3.1. Repository
Clone the repository (requires git):
```bash
git clone https://github.com/XinyuYanTJU/LawDIS.git
cd LawDIS
```

### 3.2. Dependencies
#### ✅ Step 1. Install the dependencies:
```bash
conda create -n lawdis python=3.8
conda activate lawdis
pip install -r requirements.txt
```

#### ✅ Step 2. Integrate Custom VAE into `diffusers`

This project uses a custom VAE class `AutoencoderKlLawDIS` that needs to be manually added into the `diffusers` library.

```bash
bash install_lawdis_diffusers.sh
```

### 3.3. Dataset Preparation

Download the **DIS5K dataset** from this [Google Drive link](https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1y6CQJYledfYyEO0C_Gejpw?pwd=rtgw) with the fetch code: `rtgw`. Unzip the dataset and move the DIS5K folder into the LawDIS/data directory.

The language prompts we annotated for DIS5K can be found in `LawDIS/data/json/`.

### 3.4. Inference
#### ✅ Step 1. Download the Checkpoints

Download the pre-trained checkpoints from this [Google Drive link](https://drive.google.com/drive/folders/1RDBTj5-Z9Ek9wqnYoQHkFz4_zCFwiKK_?usp=drive_link) or [Baidu Pan link](https://pan.baidu.com/s/1DGqK_Nl3ccv_pi4mIOMndw) with the fetch code: `2025`.
Place the checkpoint files under:

```bash
.stable-diffusion-2/
```

---

#### ✅ Step 2. Inference in **Macro** Mode

We provide scripts for:
- **Batch testing** a dataset
- Testing **a single image** with multiple language **prompts**
- **Fully automatic** testing of **a single image** without requiring prompt input

**Batch testing**

```bash
python script/infer_macro_batch_imgs.py \
    --checkpoint "stable-diffusion-2" \
    --input_rgb_dir "data/DIS5K" \
    --subset_name "DIS-TE4" \
    --prompt_dir 'data/json' \
    --output_dir "output/output-macro" \
    --denoise_steps 1 \
    --processing_res 1024 
```

**Testing a single image with prompts**

```bash
python script/infer_macro_single_img.py \
    --checkpoint "stable-diffusion-2" \
    --input_img_path "data/imgs/2#Aircraft#7#UAV#16522310810_468dfa447a_o.jpg" \
    --prompts "Black professional camera drone with a high-definition camera mounted on a gimbal." "Three men beside a UAV." \
    --output_dir 'output/output-macro-single' \
    --denoise_steps 1 \
    --processing_res 1024 
```

**Fully automatic testing of a single image without a prompt**

```bash
python script/infer_macro_single_img.py \
    --checkpoint "stable-diffusion-2" \
    --input_img_path "data/imgs/2#Aircraft#7#UAV#16522310810_468dfa447a_o.jpg" \
    --prompts "" \
    --output_dir 'output/output-macro-single' \
    --denoise_steps 1 \
    --processing_res 1024 
```

---

#### ✅ Step 3. Inference in **Micro** Mode

We provide scripts for:
- **Batch testing** a dataset
- Testing **a single image**

You can choose how to generate the refinement windows using `--window_mode`:
- `"auto"`: Automatically select windows based on object edges in the initial segmentation map.
- `"semi-auto"`: Simulate user-guided selection using GT segmentation.
- `"manual"`: User manually selects windows (⚠️ Only works on **local servers**).

**Batch testing**

```bash
python script/infer_micro_batch_imgs.py \
    --checkpoint "stable-diffusion-2" \
    --input_rgb_dir "data/DIS5K" \
    --subset_name "DIS-TE4" \
    --init_seg_dir 'output/output-macro/' \
    --output_dir "output/output-micro/" \
    --window_mode "semi-auto" \
    --denoise_steps 1 \
    --processing_res 1024 
```

**Single image testing**

```bash
python script/infer_micro_single_img.py \
    --checkpoint "stable-diffusion-2" \
    --input_img_path "data/imgs/2#Aircraft#7#UAV#16522310810_468dfa447a_o.jpg" \
    --init_seg_dir 'output/output-macro-single/2#Aircraft#7#UAV#16522310810_468dfa447a_o_0.png' \
    --output_dir "output/output-micro-single" \
    --window_mode "auto" \
    --denoise_steps 1 \
    --processing_res 1024 
```
## 🏋️ 4. SOTA Results

The predicted segmentation maps can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/16WlZEq4NQso3gP3AdYcbdNf9Q0_KjZHN?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1em-6dEh2Qr2si17zi-CCxg) with the fetch code: `lawd`. 

`LawDIS-S` refers to the initial segmentation results obtained under the macro mode with language prompt control. `LawDIS-R` refers to the refined results obtained under the micro mode, where window-based refinement are applied to LawDIS-S. 

Notably, the initial results (LawDIS-S) already achieve SOTA performance, and LawDIS-R further improves the metrics.

<p align="center">
    <img src="assets/4.png"/> <br />
    <em> 
    Fig. 2: Quantitative comparison of DIS5K with 11 representative methods.
    </em>
</p>

<p align="center">
    <img src="assets/5.jpg"/> <br />
    <em> 
    Fig. 3: Qualitative comparison of our model with four leading models. Local masks are evaluated with MAE score for clarity.
    </em>
</p>

## 🎮 5. Applications
Due to its capability of achieving high-precision segmentation of foreground objects at high resolutions, our LawDIS enables extensive application across a variety of scenarios. Fig. 6 shows application cases of background removal. As can be seen, compared with the original image, the background-removed image shows higher aesthetic values and good usability, which can even be directly used as: 3D modeling, augmented reality (AR), and still image animation.
<div align="center">
  <table>
    <tr>
      <td><img src="assets/1.gif" height="200px"/></td>
      <td><img src="assets/2.gif" height="200px"/></td>
      <td><img src="assets/3.gif" height="200px"/></td>
    </tr>
  </table>
  <em>
    Fig. 4: Application cases of background-removed results in various scenarios.
  </em>
</div>

<p align="center">
    <img src="assets/app1-3D.gif" width="850px" /> <br />
    <em> 
    Fig. 5: Application cases of 3D modeling.
    </em>
</p>

<p align="center">
    <img src="assets/app2-AR.gif" width="850px"/> <br />
    <em> 
    Fig. 6: Application cases of AR.
    </em>
</p>

<p align="center">
    <img src="assets/app3-Still-Image-Animation.gif" width="850px"/> <br />
    <em> 
    Fig. 7: Application cases of still image animation.
    </em>
</p>

## 📦 6. Acknowledgement

Our code is based on [Marigold](https://github.com/prs-eth/marigold) and [Diffusers](https://github.com/huggingface/diffusers). Latest DIS studies can refer to this [awesome paper list](https://github.com/Tennine2077/Awesome-Dichotomous-Image-Segmentation) organised by [Xianjie Liu (SCU)](https://github.com/Tennine2077). We are grateful to the authors of these projects for their pioneering work and contributions!


## 🎓 7. Citations

 If you find this code useful, we kindly ask you to cite our paper in your work.
 
```
@article{yan2025lawdis,
  title={LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation},
  author={Xinyu Yan and Meijun Sun and Ge-Peng Ji and Fahad Shahbaz Khan and Salman Khan and Deng-Ping Fan},
  journal={arXiv preprint arXiv:2508.01152},
  year={2025}
}
```
