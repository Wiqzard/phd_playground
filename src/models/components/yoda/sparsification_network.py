import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SparsificationNetwork(nn.Module):
    def __init__(self, num_samples: int, tau: float, threshold: float):
        super().__init__()

        self.num_samples = num_samples
        self.tau = tau
        self.threshold = threshold

    def forward(self, flows: torch.Tensor, num_vectors: int = None):
        """

        :param flows: [bs, 2, h, w]
        :param num_vectors: number of vectors to select
        :return: [bs, 3, h, w]
        """

        height = flows.size(2)
        width = flows.size(3)

        # Construct distribution to sample from
        dist = torch.norm(flows, p=2, dim=1) / self.tau
        dist = rearrange(dist, "b h w -> b (h w)")

        # Sample random locations and filter outliers
        indices = torch.multinomial(
            dist, num_samples=self.num_samples if num_vectors is None else num_vectors
        ).to(
            flows.device
        )  # [b, k]

        # Compute masks
        flat_masks = F.one_hot(indices, num_classes=height * width)  # [b, k, h * w]
        flat_masks = torch.max(flat_masks, dim=1).values  # [b, h * w]
        masks = rearrange(flat_masks, "b (h w) -> b h w", h=height).unsqueeze(
            1
        )  # [b, 1, h, w]
        masks = masks.to(flows.dtype)

        # Calculate sparse features
        sparse_output = masks * flows
        sparse_output = torch.cat([sparse_output, masks], dim=1)

        return sparse_output, indices, masks
