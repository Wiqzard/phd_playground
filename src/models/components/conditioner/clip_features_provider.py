import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..utils import _resize_with_antialiasing


class CLIPFeaturesProvider(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="feature_extractor",
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_encoder",
        )

    def encode_image(self, pixel_values):
        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0
        device = pixel_values.device

        # Normalize the image with for CLIP input
        pixel_values = self.feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(device, dtype=self.weight_dtype)
        image_embeddings = self.image_encoder(pixel_values).image_embeds
        return image_embeddings

    def forward(self, pixel_values):
        return self.encode_image(pixel_values)
