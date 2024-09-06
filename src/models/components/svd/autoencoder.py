from diffusers import AutoencoderKLTemporalDecoder


class AutoencoderKLTemporalDecoderWrapper(AutoencoderKLTemporalDecoder):
    def __init__(
        self,
        pretrained: bool = False,
        pretrained_model_name_or_path: str = None,
        subfolder: str = "unet",
        low_cpu_mem_usage: bool = False,
        variant: str = "fp16",
        *args,
        **kwargs,
    ):
        wrapper_specific_kwargs = {
            "pretrained": pretrained,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "variant": variant,
        }
        parent_kwargs = {
            k: v for k, v in kwargs.items() if k not in wrapper_specific_kwargs
        }
        super().__init__(*args, **parent_kwargs)

        if pretrained:
            pretrained_model = self.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=subfolder,
                low_cpu_mem_usage=low_cpu_mem_usage,
                variant=variant,
            )
            self.__dict__.update(pretrained_model.__dict__)
