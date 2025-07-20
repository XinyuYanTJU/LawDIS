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
from .util.batchsize import find_batch_size
from .util.image_util import (
    get_tv_resample_method
)

class LawDISMacroPipeline(DiffusionPipeline):
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

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
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
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image) #[3, 1280, 960]
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W] #[1， 3, 1280, 960]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0: #[1, 3, 768, 576]
            rgb = resize(rgb, (processing_res, processing_res), resample_method, antialias=True)
            

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting dis -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(1, -1, -1, -1) 
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                    input_res=max(rgb_norm.shape[1:]),
                    dtype=self.dtype,
            )
        # print("batchsize:"+str(_bs))

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict dis maps (batched)
        dis_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch   
            dis_pred_raw = self.single_infer(
                rgb_in=batched_img,
                prompt=prompt,
                num_inference_steps=denoising_steps,
                generator=generator,
            ) 
            dis_pred_ls.append(dis_pred_raw.detach())
        dis_pred = torch.concat(dis_pred_ls, dim=0) 
        torch.cuda.empty_cache()  # clear vram cache for ensembling


        # Resize back to original resolution
        dis_pred = resize(
                dis_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            ) #[1, 1, 4160, 5547]

        # Convert to numpy
        dis_pred = dis_pred.squeeze() #[4160, 5547]
        dis_pred = dis_pred.cpu().numpy() #(4160, 5547)


        # Clip output range
        # dis_pred = dis_pred.clip(0, 1) 

        return dis_pred



    def encode_text(self,prompt):
        """
        Encode text embedding for empty prompt
        """
        text_inputs = self.tokenizer( 
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) 
        text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) 
        return text_embed

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        prompt: str,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
    ) -> torch.Tensor:

        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  

        # Encode image
        rgb_latent,features_saver = self.encode_rgb(rgb_in) 


        # Initial dis map (noise)
        dis_latent = torch.randn( 
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  


        batch_text_embed = self.encode_text(prompt).repeat((rgb_latent.shape[0], 1, 1)).to(device)
        
 
        iterable = enumerate(timesteps)
        switch_class = torch.tensor([[0., 1.]], device=device, dtype=torch.float32)
        class_embedding = torch.cat([torch.sin(switch_class), torch.cos(switch_class)], dim=-1)

        for i, t in iterable:
            
            unet_input = torch.cat(
                [rgb_latent, dis_latent], dim=1
            )  
            

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_text_embed,class_labels=class_embedding
            ).sample  # [B, 4, h, w]
            

            # compute the previous noisy sample x_t -> x_t-1
            dis_latent = self.scheduler.step(
                noise_pred, t, dis_latent, generator=generator
            ).prev_sample 

        
        dis = self.decode_dis(dis_latent,features_saver).sigmoid() #yxy
        
        # dis = torch.clamp((dis + 1.0) / 2.0, min=0.0, max=1.0)
        

        return dis

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
