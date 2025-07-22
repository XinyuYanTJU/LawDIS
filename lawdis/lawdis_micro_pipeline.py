from typing import Dict, Optional, Union
import numpy as np
import torch
from diffusers import (
    AutoencoderKlLawDIS,
    DDIMScheduler,
    DiffusionPipeline,
    TCDScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

from .util.batchsize import find_batch_size
from .util.image_util import (
    get_tv_resample_method
)


class LawDISMicroPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    dis_latent_scale_factor = 0.18215
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKlLawDIS,
        scheduler: Union[DDIMScheduler, TCDScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        result_image: Union[Image.Image, torch.Tensor],
        patch_coordinates_and_sizes: list,
        denoising_steps: Optional[int] = None,
        processing_res: Optional[int] = None,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        
        assert processing_res >= 0
        resample_method: InterpolationMode = get_tv_resample_method(resample_method) 

        # ----------------- Image Preprocess -----------------
        if len(patch_coordinates_and_sizes)>0:
            if isinstance(input_image, Image.Image):
                input_image = input_image.convert("RGB")
                rgb = pil_to_tensor(input_image) #[3, 1280, 960]
                rgb = rgb.unsqueeze(0)  # [1, rgb, H, W] #[1， 3, 1280, 960]
                result = pil_to_tensor(result_image)
                result = result.unsqueeze(0) 
            elif isinstance(input_image, torch.Tensor):
                rgb = input_image
            else:
                raise TypeError(f"Unknown input type: {type(input_image) = }")
            input_size = rgb.shape
            assert (
                4 == rgb.dim() and 3 == input_size[-3]
            ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

            # Normalize rgb values
            rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            rgb_norm = rgb_norm.to(self.dtype)
            result_norm: torch.Tensor = result / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            result_norm = result_norm.to(self.dtype)
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            assert result_norm.min() >= -1.0 and result_norm.max() <= 1.0

            # ----------------- Predicting dis -----------------
            # Batch repeated input image
            duplicated_rgb = rgb_norm.expand(1, -1, -1, -1) 
            duplicated_result = result_norm.expand(1, -1, -1, -1)
            single_rgb_dataset = TensorDataset(duplicated_rgb,duplicated_result)
            _bs = batch_size
            max_patch_num = find_batch_size(
                    input_res=processing_res,
                    dtype=self.dtype,
                )

            single_rgb_loader = DataLoader(
                single_rgb_dataset, batch_size=_bs, shuffle=False
            )

            # Predict dis maps (batched)
            if show_progress_bar:
                iterable = tqdm(
                    single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
                )
            else:
                iterable = single_rgb_loader
            for batch in iterable:
                batched_img, batched_result  = batch   #[5, 3, 768, 576]
                dis_pred = self.single_infer(
                    result_image_np = np.array(result_image),
                    patch_coordinates_and_sizes = patch_coordinates_and_sizes,
                    processing_res = processing_res,
                    max_patch_num = max_patch_num,
                    rgb_in=batched_img,
                    result_in=batched_result, 
                    num_inference_steps=denoising_steps,
                    show_pbar=show_progress_bar,
                    generator=generator,
                ) #[5, 1, 768, 576]
            torch.cuda.empty_cache()  # clear vram cache for ensembling
        else: 
            dis_pred = np.array(result_image)

  
        return dis_pred


    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer( 
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) 
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) 
    
    def get_patches_from_tensor(self,tensor, coordinates_and_sizes, patch_size=1024):

        patches = []
        for (x1, y1), (width, height) in coordinates_and_sizes:

            patch = tensor[:, :, y1:y1+height, x1:x1+width]
            if patch.shape[2] != patch_size or patch.shape[3] != patch_size:
                patch = F.interpolate(patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            patches.append(patch)
        
        return patches
    

    def apply_patches_to_image(self, image, patches, coordinates_with_sizes):

        image_np = np.array(image, dtype=np.float32)
        
        # Initialize the final output image and an overlap area map
        final_image = np.zeros_like(image_np, dtype=np.float32)
        overlap_area_map = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
        
        for patch, ((x1, y1), (width, height)) in zip(patches, coordinates_with_sizes):
            # Resize the 1024x1024 patch back to its original size
            patch_resized = Image.fromarray(patch).resize((width, height), Image.BILINEAR)
            patch_np = np.array(patch_resized, dtype=np.float32)  
            
            # Calculate the proportion of gray pixels
            gray_ratio = np.sum((patch_np > 5) & (patch_np < 250)) / patch_np.size
            
            # Skip the patch if more than 30% of pixels are gray
            if gray_ratio > 0.3:
                continue
            
            # Calculate the black and white pixel ratios in the corresponding original region
            original_region = image_np[y1:y1+height, x1:x1+width]
            original_black_ratio = np.sum(original_region < 5) / original_region.size
            original_white_ratio = np.sum(original_region > 250) / original_region.size
            patch_black_ratio = np.sum(patch_np < 5) / patch_np.size
            patch_white_ratio = np.sum(patch_np > 250) / patch_np.size
            
            # If a mostly black region in the original becomes white in the patch (or vice versa), skip it
            if (original_black_ratio > 0.5 and patch_white_ratio > 0.5) or (original_white_ratio > 0.5 and patch_black_ratio > 0.5):
                continue
            
            # Compute the end coordinates of the patch region
            x_end = x1 + width
            y_end = y1 + height
            
       
            overlap_area = np.ones((height, width), dtype=np.float32)
            final_image[y1:y_end, x1:x_end] = np.where(
                overlap_area > overlap_area_map[y1:y_end, x1:x_end],
                patch_np,
                final_image[y1:y_end, x1:x_end]
            )
            overlap_area_map[y1:y_end, x1:x_end] = np.maximum(
                overlap_area_map[y1:y_end, x1:x_end],
                overlap_area
            )
        
        # For regions not covered by any patch, retain the original pixel values
        uncovered_mask = overlap_area_map == 0
        final_image[uncovered_mask] = image_np[uncovered_mask]
        

        final_image_uint8 = final_image.astype(np.uint8)
        
        return final_image_uint8


    @torch.no_grad()
    def single_infer(
        self,
        result_image_np,
        patch_coordinates_and_sizes,
        processing_res,
        max_patch_num,
        rgb_in: torch.Tensor,
        result_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:

        device = self.device
        result_in = result_in.repeat(1, 3, 1, 1)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T] [999]
        switch_class = torch.tensor([[1., 0.]], device=device, dtype=torch.float32)
        class_embedding = torch.cat([torch.sin(switch_class), torch.cos(switch_class)], dim=-1)

        # Encode image
        rgb_in_patches = self.get_patches_from_tensor(rgb_in, patch_coordinates_and_sizes, processing_res)
        result_in_patches = self.get_patches_from_tensor(result_in, patch_coordinates_and_sizes, processing_res)

        rgb_in_batches = [torch.cat(rgb_in_patches[i:i + max_patch_num], dim=0) for i in range(0, len(rgb_in_patches), max_patch_num)]
        result_in_batches = [torch.cat(result_in_patches[i:i + max_patch_num], dim=0) for i in range(0, len(result_in_patches), max_patch_num)]
        dis_patches = []
        for rgb_in_batch, result_in_batch in zip(rgb_in_batches, result_in_batches):
            rgb_in_batch = rgb_in_batch.to(device)
            result_in_batch = result_in_batch.to(device)
            rgb_in_batch_fea = self.encode_rgb(rgb_in_batch)[0]
            features_savers = self.encode_rgb(rgb_in_batch)[1]
            result_in_batch_fea = self.encode_rgb(result_in_batch)[0]
            

            self.encode_empty_text() #[1, 2, 1024]
            batch_empty_text_embed = self.empty_text_embed.repeat(
                (rgb_in_batch_fea.shape[0], 1, 1)
            ).to(device)  # [5, 2, 1024]

            # Denoising loop
            if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising",
                )
            else:
                iterable = enumerate(timesteps)

            for i, t in iterable:
               
                    unet_input = torch.cat([rgb_in_batch_fea, result_in_batch_fea], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(
                        unet_input, t, encoder_hidden_states=batch_empty_text_embed,class_labels=class_embedding
                    ).sample  # [B, 4, h, w]

                    # compute the previous noisy sample x_t -> x_t-1
                    result_in_batch_fea = self.scheduler.step(
                        noise_pred, t, result_in_batch_fea, generator=generator
                    ).prev_sample

            dis_decoded = self.decode_dis(result_in_batch_fea,features_savers).sigmoid()

            clamped_tensor = np.clip((dis_decoded.squeeze().cpu().numpy())* 255.0,0, 255)


            if clamped_tensor.ndim == 2:
      
                clamped_tensor = clamped_tensor[np.newaxis, :, :]  

            for i in range(clamped_tensor.shape[0]):
                dis_patches.append(clamped_tensor[i, :, :])
            del rgb_in_batch_fea, result_in_batch_fea, noise_pred
            torch.cuda.empty_cache()

        final_image_array = self.apply_patches_to_image(result_image_np, dis_patches, patch_coordinates_and_sizes)

        return final_image_array

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h,encoder_features = self.vae.encoder(rgb_in) 
        moments = self.vae.quant_conv(h) 
        mean, logvar = torch.chunk(moments, 2, dim=1) 
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor # *0.18215
        return rgb_latent,encoder_features

    def decode_dis(self, dis_latent: torch.Tensor,feature_saver: list) -> torch.Tensor: 
        """
        Decode dis latent into dis map.

        Args:
            dis_latent (`torch.Tensor`):
                dis latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded dis map.
        """
        # scale latent
        dis_latent = dis_latent / self.dis_latent_scale_factor  # /0.18215
        # decode
        z = self.vae.post_quant_conv(dis_latent.cuda()) 
        pred = self.vae.decoder(z,feature_saver)
        return pred