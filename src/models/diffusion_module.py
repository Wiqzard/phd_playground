import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
import numpy as np
from lightning import LightningModule
import torch.utils.checkpoint
#from diffusers import (  # UNetSpatioTemporalConditionModel,
#    AutoencoderKLTemporalDecoder,
#    StableVideoDiffusionPipeline,
#)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from huggingface_hub import create_repo, upload_folder
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .components.utils import _resize_with_antialiasing, rand_log_normal
from src.utils.torch_utils import default, repeat_as_img_seq, append_dims


from inspect import isfunction
from einops import repeat



#class EDMSampling:
#    def __init__(self, p_mean=-1.2, p_std=1.2, num_frames=25):
#        self.p_mean = p_mean
#        self.p_std = p_std
#        self.num_frames = num_frames
#
#    def __call__(self, n_samples, rand=None):
#        bs = n_samples // self.num_frames
#        rand_init = torch.randn((bs,))[..., None]
#        rand_init = repeat(rand_init, "b 1 -> (b t)", t=self.num_frames)
#        rand = default(rand, rand_init)
#        log_sigma = self.p_mean + self.p_std * rand
#        return log_sigma.exp()
#
#class Denoiser(nn.Module):
#    def __init__(self, scaling_config: Dict, num_frames: int = 25):
#        super().__init__()
#        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)
#        self.num_frames = num_frames
#
#    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
#        return sigma
#
#    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
#        return c_noise
#
#    def forward(
#            self,
#            network: nn.Module,
#            noised_input: torch.Tensor,
#            sigma: torch.Tensor,
#            cond: Dict,
#            cond_mask: torch.Tensor
#    ):
#        sigma = self.possibly_quantize_sigma(sigma)
#        sigma_shape = sigma.shape
#        sigma = append_dims(sigma, noised_input.ndim)
#        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
#        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
#        return (network(noised_input * c_in, c_noise, cond, cond_mask, self.num_frames) * c_out + noised_input * c_skip)
#
#

class SVDLightningModule(LightningModule):
    def __init__(
        self,
        autoencoder: Any,
        unet: Any,
        conditioner: Any,
        optimizer: Any,
        scheduler: Any,
        sigma_sampler: Any,
        denoiser: Any,
        num_frames: int = 25,
        params_to_select: Optional[List[str]] = None,
    ):
        super().__init__()  # autoencoder: Any,

        self.save_hyperparameters(ignore=["autoencoder", "unet", "conditioner", "optimizer", "scheduler"])

        self.autoencoder = autoencoder
        self.unet = unet
        self.conditioner = conditioner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sigma_sampler = sigma_sampler
        self.denoiser = denoiser
        print(unet)


        ## Freeze vae and image_encoder
        # self.vae.requires_grad_(False)
        # self.image_encoder.requires_grad_(False)
        # self.unet.requires_grad_(False)
        # self.unet.train()

        # if cfg.use_ema:
        #    self.ema_unet = EMAModel(
        #        self.unet.parameters(),
        #        model_cls=UNetSpatioTemporalConditionModel,
        #        model_config=self.unet.config,
        #    )

        # self.weight_dtype = torch.float32
        # if cfg.mixed_precision == "fp16":
        #    self.weight_dtype = torch.float16
        # elif cfg.mixed_precision == "bf16":
        #    self.weight_dtype = torch.bfloat16

        ## self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### Move models to correct dtype
        ## self.image_encoder.to(self.device_, dtype=self.weight_dtype)
        ## self.vae.to(self.device_, dtype=self.weight_dtype)
        ## self.cond_generator.to(self.device_)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()


        # self.configure_optimizers()

        ## for validation -- need to  be changed later
        # cfg.base_folder = env.ROOT_SCRATCH / cfg.base_folder
        # cfg.output_dir = env.ROOT_SCRATCH / cfg.output_dir
        # self.output_dir = cfg.output_dir

        ## base_val_path = env.ROOT_SCRATCH / "2_ds_preprocess/1_Frames/val/"
        # base_val_path = self.cfg.val_path  # env.VAL_PATH
        # folders = []
        # for root, dirs, files in os.walk(base_val_path):
        #    for dir_name in dirs:
        #        dir_path = os.path.join(root, dir_name)
        #        folders.append(dir_path)

        # self.all_val_paths = []
        # for folder in folders:
        #    folder_path = os.path.join(base_val_path, folder)
        #    files = os.listdir(folder_path)
        #    for file in files:
        #        if file.endswith(".png") or file.endswith(".jpg"):
        #            self.all_val_paths.append(os.path.join(folder_path, file))

        # print("Total validation images: ", len(self.all_val_paths))

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(
            add_time_ids
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids


    def configure_optimizers(self):
        #parameters_list = list(self.cond_generator.parameters())
        parameters_list = []
        if (
            self.hparams.params_to_select is None
            or len(self.hparams.params_to_select) == 0
        ):
            for param in self.unet.parameters():
                parameters_list.append(param)
                param.requires_grad = True
        else:
            for name, param in self.unet.named_parameters():
                for to_select in self.hparams.params_to_select:
                    if to_select in name:
                        parameters_list.append(param)
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
    
        optimizer = self.optimizer(
            params=parameters_list,
            #lr=self.cfg.learning_rate,
            #betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            #eps=self.cfg.adam_epsilon,
        )
        if self.scheduler is None:
            return optimizer
    
        return [optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    
    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = list()

        video_length = x.shape[1]
        x = rearrange(x, "b f c h w -> (b f) c h w")

        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.autoencoder.encode(
                    x[n * n_samples: (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = rearrange(z, "(b f) c h w -> b f c h w", f=video_length)
        z = z * self.scale_factor
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)  # go to StandardDiffusionLoss
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def get_noised_input(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        cond_mask = torch.zeros_like(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:  # the entire channel is shifted together
            offset_shape = (input.shape[0], input.shape[1])
            # offset_shape = (input.shape[0] // self.num_frames, 1, input.shape[1])
            rand_init = torch.randn(offset_shape, device=input.device)
            # rand_init = repeat(rand_init, "b 1 c -> (b t) c", t=self.num_frames)
            noise = noise + self.offset_noise_level * append_dims(rand_init, input.ndim)
        #if self.replace_cond_frames:
        #    sigmas_bc = append_dims((1 - cond_mask) * sigmas, input.ndim)
        #else:
        sigmas_bc = append_dims(sigmas, input.ndim)

        noised_input = input + noise * sigmas_bc
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, noised_input.ndim)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = 0.25 * sigma.log()
        

        time_steps = c_noise
        return None , time_steps
    





#

        model_output = self(
            noised_input,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
        )

        if self.replace_cond_frames:  # ignore mask predictions
            predict = model_output * append_dims(
                1 - cond_mask, input.ndim
            ) + input * append_dims(cond_mask, input.ndim)
        else:
            predict = model_output

    def criterion(self, predict, target):
        return torch.mean(((predict - target) ** 2).reshape(target.shape[0], -1), 1)

    def training_step(self, batch, batch_idx):
        pixel_values = (
            batch["pixel_values"].to(self.weight_dtype)
            # .to(self.device, non_blocking=True)
        )
        batch_size = pixel_values.shape[0]

        # sample conditioning
        ######### conditioning code start

        # encoder_hidden_states = self.encode_image(pixel_values[:, 0, :, :, :])
        encoder_hidden_states = self.conditioner(pixel_values[:, 0, :, :, :])
        # cond_feats, pixel_values = self.cond_generator(
        #    pixel_values,
        #    self.generator,
        #    enable_pixels_dropout=np.random.rand()
        #    < self.cond_generator.conditioning_provider.image_dropout_prob,
        # )
        # cond_mask = self.cond_generator.conditioning_provider.create_random_mask(
        #    cond_feats
        # )

        # enable_cond = (
        #    torch.rand(batch_size, device=pixel_values.device, generator=self.generator)
        #    > self.cond_generator.conditioning_provider.no_condition_prob
        # )
        # for b in range(batch_size):
        #    enable_condition = enable_cond[b,]
        #    cond_mask[b,] *= enable_condition

        # cond_feats = cond_mask * cond_feats + (
        #    1 - cond_mask
        # ) * self.cond_generator.cond_masked_tokens.unsqueeze(0)

        ########## end

        conditional_pixel_values = pixel_values[:, [0], :, :, :]

        # Get latents to start training the model
        latents = self.tensor_to_vae_latent(pixel_values, self.vae)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        cond_sigmas = rand_log_normal(
            shape=[batch_size],
            loc=-3.0,
            scale=0.5,
        ).to(latents)
        noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1
        cond_sigmas = cond_sigmas[:, None, None, None, None]
        conditional_pixel_values = (
            torch.randn_like(conditional_pixel_values) * cond_sigmas
            + conditional_pixel_values
        )
        conditional_latents = self.tensor_to_vae_latent(
            conditional_pixel_values, self.vae
        )[:, 0, :, :, :]
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        # Sample a random timestep for each image
        sigmas = rand_log_normal(
            shape=[batch_size],
            loc=0.7,
            scale=1.6,
        ).to(latents.device)
        sigmas = sigmas[:, None, None, None, None]
        noisy_latents = latents + noise * sigmas
        timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
            noisy_latents.device,
        )

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        added_time_ids = self._get_add_time_ids(
            7,  # fixed
            127,  # motion_bucket_id = 127, fixed
            noise_aug_strength,  # noise_aug_strength == cond_sigmas
            latents.dtype,
            batch_size,
        )
        added_time_ids = added_time_ids.to(latents.device)

        # Conditioning dropout to support classifier-free guidance during inference
        if self.cfg.conditioning_dropout_prob is not None:
            random_p = torch.rand(batch_size, device=latents.device)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * self.cfg.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
            # Final text conditioning.
            null_conditioning = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.where(
                prompt_mask,
                null_conditioning.unsqueeze(1),
                encoder_hidden_states.unsqueeze(1),
            )
            # Sample masks for the original images.
            image_mask_dtype = conditional_latents.dtype
            image_mask = 1 - (
                (random_p >= self.cfg.conditioning_dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * self.cfg.conditioning_dropout_prob).to(
                    image_mask_dtype
                )
            )
            image_mask = image_mask.reshape(batch_size, 1, 1, 1)
            # Final image conditioning.
            conditional_latents = image_mask * conditional_latents

            # random_p = torch.rand(batch_size, device=latents.device)
            # image_mask_dtype = conditional_latents.dtype
            # image_mask = 1 - (
            #    (random_p >= self.cfg.conditioning_dropout_prob).to(image_mask_dtype)
            #    * (random_p < 3 * self.cfg.conditioning_dropout_prob).to(
            #        image_mask_dtype
            #    )
            # )
            # image_mask = image_mask.reshape(batch_size, 1, 1, 1)
            # conditional_latents = image_mask * conditional_latents

        conditional_latents = conditional_latents.unsqueeze(1).repeat(
            1, noisy_latents.shape[1], 1, 1, 1
        )
        inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

        target = latents
        model_pred = self(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
        )

        c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = model_pred * c_out + c_skip * noisy_latents
        weighing = (1 + sigmas**2) * (sigmas**-2.0)

        loss = torch.mean(
            (
                weighing.float() * (denoised_latents.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            dim=1,
        )
        loss = loss.mean()

        self.log_dict(
            {
                "train_loss": loss,
                "train_noise_aug_strength": noise_aug_strength,
                "train_cond_sigmas": cond_sigmas.mean(),
                "train_sigmas": sigmas.mean(),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def val_dataloader(self):
        # Use the RandomImageDataset to create the validation dataset
        val_dataset = RandomImageDataset(base_val_path=self.cfg.val_path)
        return DataLoader(
            val_dataset,
            batch_size=1,  # Use batch size 1 for random image selection per step
            num_workers=self.cfg.num_workers,
            shuffle=False,  # Shuffle is not needed as we select randomly within the dataset
        )

    def validation_step(self, batch, batch_idx):
        # if main rank:
        if self.trainer.is_global_zero:
            if self.cfg.use_ema and self.ema_unet is not None:
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())

            # Unwrap models for compatibility in distributed training mode.
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                unet=self.unet.module if hasattr(self.unet, "module") else self.unet,
                image_encoder=self.image_encoder,
                vae=self.vae,
                revision=self.cfg.revision,
                torch_dtype=self.weight_dtype,
                # cond_generator=self.cond_generator,
            )
            device = torch.device(f"cuda:{self.local_rank}")
            # pipeline = pipeline.to(device)
            pipeline.set_progress_bar_config(disable=True)

            val_save_dir = os.path.join(self.cfg.output_dir, "validation_images")
            os.makedirs(val_save_dir, exist_ok=True)

            # with torch.autocast(
            #    #device_type=str(self.device_).replace(":0", ""),
            #    enabled=self.cfg.mixed_precision == "fp16",
            # ):
            if True:
                # `batch` contains one image due to batch_size=1
                num_frames = self.cfg.num_frames

                video_frames = pipeline(
                    load_image(batch[0]).resize((self.cfg.width, self.cfg.height)),
                    height=self.cfg.height,
                    width=self.cfg.width,
                    num_frames=num_frames,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                ).frames[0]

                # Convert video frames to numpy arrays and transpose for correct format
                video_frames = np.asarray(
                    [np.array(img) for img in video_frames]
                )  # (time, h, w, c)
                video_frames = video_frames.transpose(0, 3, 1, 2)  # (time, c, h, w)

                # Log video using PyTorch Lightning's logging system
                video = wandb.Video(video_frames, fps=7)
                wandb.log({"val_img": video})
                # self.log("val_img", wandb.Video(video_frames, fps=7))

                output_dir = self.output_dir
                gif_filename = f"val_img_epoch{self.current_epoch}_step{self.global_step}_{batch_idx}.gif"
                gif_path = os.path.join(output_dir, gif_filename)

                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)
                video_frames = video_frames.transpose(0, 2, 3, 1)  # (time, h, w, c)
                # Convert frames to PIL Images and save as GIF
                pil_images = [Image.fromarray(frame) for frame in video_frames]
                pil_images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=1000 / 7,  # Duration per frame in milliseconds
                    loop=0,
                )

                print(f"Video saved as GIF to {gif_path}")

            if self.cfg.use_ema and self.ema_unet is not None:
                # Switch back to the original UNet parameters.
                self.ema_unet.restore(self.unet.parameters())

            # Clean up to free memory
            del pipeline
            torch.cuda.empty_cache()

    def train_dataloader(self):
        train_dataset = self.train_dataset
        train_dataset = SVDVideoDataset(
            self.cfg.base_folder,
            width=self.cfg.width,
            height=self.cfg.height,
            sample_frames=self.cfg.num_frames,
            max_steps=self.cfg.max_train_steps,
        )
        # sampler = RandomSampler(train_dataset)
        return torch.utils.data.DataLoader(
            train_dataset,
            # sampler=sampler,
            shuffle=True,
            batch_size=self.cfg.per_gpu_batch_size,
            num_workers=self.cfg.num_workers,
        )

    def load_from_checkpoint(self, path):
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.cfg.gradient_accumulation_steps
        )

        # chpt_cond_generator = torch.load(
        #    os.path.join(self.cfg.output_dir, "cond_generator.pth")
        # )
        # self.cond_generator.load_state_dict(chpt_cond_generator)
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * self.cfg.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * self.cfg.gradient_accumulation_steps
        )
