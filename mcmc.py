def _load_vae(self) -> None:
    """
    Load the pretrained VAE model.
    """
    vae_cls = VideoVAE if self.is_latent_video_vae else ImageVAE
    self.vae = vae_cls.from_pretrained(
        path=self.cfg.vae.pretrained_path,
        torch_dtype=(
            torch.float16 if self.cfg.vae.use_fp16 else torch.float32
        ),  # only for Diffuser's ImageVAE
        **self.cfg.vae.pretrained_kwargs,
    ).to(self.device)
    freeze_model(self.vae)


@torch.no_grad()
def _run_vae(
    self,
    x: Tensor,
    shape: str,
    vae_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Helper function to run the VAE, either for encoding or decoding.
    - Requires shape to be a permutation of b, t, c, h, w.
    - Reshapes the input tensor to the required shape for the VAE, and reshapes the output back.
        - x: `shape` shape.
        - VideoVAE requires (b, c, t, h, w) shape.
        - ImageVAE requires (b, c, h, w) shape.
    - Split the input tensor into chunks of size cfg.vae.batch_size, to avoid memory errors.
    """
    x = rearrange(x, f"{shape} -> b c t h w")
    batch_size = x.shape[0]
    vae_batch_size = self.cfg.vae.batch_size
    # chunk the input tensor by vae_batch_size
    chunks = torch.chunk(x, (batch_size + vae_batch_size - 1) // vae_batch_size, 0)
    outputs = []
    for chunk in chunks:
        b = chunk.shape[0]
        if not self.is_latent_video_vae:
            chunk = rearrange(chunk, "b c t h w -> (b t) c h w")
        output = vae_fn(chunk)
        if not self.is_latent_video_vae:
            output = rearrange(output, "(b t) c h w -> b c t h w", b=b)
        outputs.append(output)
    return rearrange(torch.cat(outputs, 0), f"b c t h w -> {shape}")


def _encode(self, x: Tensor, shape: str = "b t c h w") -> Tensor:
    return self._run_vae(x, shape, lambda y: self.vae.encode(2.0 * y - 1.0).sample())


def _decode(self, latents: Tensor, shape: str = "b t c h w") -> Tensor:
    return self._run_vae(
        latents,
        shape,
        lambda y: (
            self.vae.decode(y, self._n_tokens_to_n_frames(latents.shape[1]))
            if self.is_latent_video_vae
            else self.vae.decode(y)
        )
        * 0.5
        + 0.5,
    )
