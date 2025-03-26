import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiLoRALinear(nn.Module):
    """
    Wraps a single nn.Linear with multiple LoRA 'slots'.
    Each slot is a separate (A,B) low-rank decomposition.
    During forward, for each sample i in the batch, we gather
    the correct slot from A and B (using lora_indices[i]) and
    add alpha * (x @ A_i @ B_i) to the base linear output.

    Args:
        original_linear: the nn.Linear being wrapped
        num_lora_slots: how many separate LoRA sets to store
        rank: low-rank dimension for each LoRA
        alpha: scale factor for LoRA (for all slots)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        num_lora_slots: int,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        # Save hyperparams
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.num_lora_slots = num_lora_slots

        # We store the "current" alpha in self.alpha, but also keep original for "re-enable"
        self.alpha = alpha
        self._original_alpha = alpha  # for toggling on/off easily

        # Store the original (base) weight and bias
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

        # Create the stacked LoRA A and B
        # A: [num_lora_slots, in_features, rank]
        # B: [num_lora_slots, rank, out_features]
        self.lora_A = nn.Parameter(torch.zeros(num_lora_slots, self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_lora_slots, rank, self.out_features))

        # Simple initialization (often small or near-zero)
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialize the LoRA A/B matrices."""
        nn.init.normal_(self.lora_A, std=1e-4)
        nn.init.normal_(self.lora_B, std=1e-4)

    def forward(
        self, x: torch.Tensor, lora_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [batch_size, ..., in_features]
        lora_indices: [batch_size], specifying which LoRA slot to use per sample.
                      If None, we just do a normal forward with the base weights.
        """
        # Base forward
        out = x.matmul(self.weight.t())
        if self.bias is not None:
            out = out + self.bias

        if (lora_indices is None) or (self.alpha == 0.0):
            # No LoRA injection if alpha=0 or if no indices provided
            return out

        # shape info
        B = x.shape[0]
        orig_shape = x.shape

        # Flatten extra dims
        x_2d = x.reshape(B, -1, self.in_features)  # [B, any, in_features]

        # Gather correct A and B for each sample
        chosen_A = self.lora_A[lora_indices]  # [B, in_features, rank]
        chosen_B = self.lora_B[lora_indices]  # [B, rank, out_features]

        # (B, N, in_features) x (B, in_features, rank) -> (B, N, rank)
        inter = torch.bmm(x_2d, chosen_A)
        # (B, N, rank) x (B, rank, out_features) -> (B, N, out_features)
        out_lora = torch.bmm(inter, chosen_B)

        # Reshape to match original
        out_lora = out_lora.reshape(*orig_shape[:-1], self.out_features)

        # Scale by alpha
        out = out + self.alpha * out_lora
        return out

    # ---------- Convenience methods for toggling alpha & trainability ----------
    def disable_adapter(self):
        """Sets alpha=0 so that LoRA injection is effectively disabled."""
        self.alpha = 0.0

    def enable_adapter(self):
        """Restores alpha to its original value."""
        self.alpha = self._original_alpha

    def set_trainable(self, trainable: bool = True):
        """Sets requires_grad for the LoRA A/B matrices."""
        self.lora_A.requires_grad_(trainable)
        self.lora_B.requires_grad_(trainable)
        # If you also want to freeze/unfreeze the base weights, do it here:
        # self.weight.requires_grad_(trainable)
        # if self.bias is not None:
        #     self.bias.requires_grad_(trainable)


###############################################################################
#                  Monkey-Patching Attention and Mlp                          #
###############################################################################


def patch_attention_forward(attn: Attention):
    """
    Modify an existing timm Attention module so that it:
      - expects a 'lora_indices' argument
      - calls its qkv and proj as multi-lora lines with that lora_indices
    """
    import types

    def forward_with_lora(self, x, lora_indices=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x, lora_indices=lora_indices)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x, lora_indices=lora_indices)
        x = self.proj_drop(x)
        return x

    attn.forward = types.MethodType(forward_with_lora, attn)


def patch_mlp_forward(mlp: Mlp):
    import types

    def forward_with_lora(self, x, lora_indices=None):
        x = self.fc1(x, lora_indices=lora_indices)
        x = self.act(x)
        x = self.fc2(x, lora_indices=lora_indices)
        return x

    mlp.forward = types.MethodType(forward_with_lora, mlp)


def patch_ditblock_forward(block):
    import types

    def forward_with_lora(self, x, c, lora_indices=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        # MSA
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), lora_indices=lora_indices
        )
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp), lora_indices=lora_indices
        )
        return x

    block.forward = types.MethodType(forward_with_lora, block)


def patch_finallayer_forward(final_layer):
    import types

    def forward_with_lora(self, x, c, lora_indices=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x, lora_indices=lora_indices)
        return x

    final_layer.forward = types.MethodType(forward_with_lora, final_layer)


def replace_linear_with_lora(
    module: nn.Module, num_lora_slots: int, rank: int, alpha: float
):
    """
    Recursively traverse `module`, and whenever we find an nn.Linear,
    replace it with MultiLoRALinear wrapping the original linear weights.
    """
    for name, child in list(module.named_children()):
        # Recurse into submodules
        replace_linear_with_lora(child, num_lora_slots, rank, alpha)

        if isinstance(child, nn.Linear):
            wrapped = MultiLoRALinear(child, num_lora_slots, rank=rank, alpha=alpha)
            setattr(module, name, wrapped)


def inject_lora(
    model: nn.Module, num_lora_slots: int, rank: int = 4, alpha: float = 1.0
):
    """
    Modifies a DiT model *in-place* so that:
      1) All nn.Linear layers in the attention, Mlp, and final layer become MultiLoRALinear.
      2) The forward passes of Attention, Mlp, DiTBlock, and FinalLayer accept a `lora_indices` arg.
      3) The DiT forward pass also accepts `lora_indices` and passes it down.
    """
    # 1) Replace nn.Linear with MultiLoRALinear
    for block in model.blocks:
        replace_linear_with_lora(block.attn, num_lora_slots, rank, alpha)
        replace_linear_with_lora(block.mlp, num_lora_slots, rank, alpha)
    replace_linear_with_lora(model.final_layer, num_lora_slots, rank, alpha)

    # 2) Patch forward methods
    for block in model.blocks:
        patch_attention_forward(block.attn)
        patch_mlp_forward(block.mlp)
        patch_ditblock_forward(block)
    patch_finallayer_forward(model.final_layer)

    # 3) Patch the DiT forward to accept lora_indices
    import types

    original_dit_forward = model.forward

    def forward_with_lora(self, x, t, lora_indices=None, **kwargs):
        y = kwargs.get("y", None)
        # Standard DiT logic
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)

        if self.num_classes > 0 and y is not None:
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for blk in self.blocks:
            x = blk(x, c, lora_indices=lora_indices)
        x = self.final_layer(x, c, lora_indices=lora_indices)
        x = self.unpatchify(x)
        return x

    model.forward = types.MethodType(forward_with_lora, model)
    return model


###############################################################################
#                         Utility: Replace nn.Linear                           #
###############################################################################


def replace_linear_with_lora(
    module: nn.Module, num_lora_slots: int, rank: int, alpha: float
):
    """
    Recursively traverse `module`, and whenever we find an nn.Linear,
    replace it with MultiLoRALinear wrapping the original linear weights.
    """
    for name, child in list(module.named_children()):
        # Recurse into submodules
        replace_linear_with_lora(child, num_lora_slots, rank, alpha)

        if isinstance(child, nn.Linear):
            wrapped = MultiLoRALinear(child, num_lora_slots, rank=rank, alpha=alpha)
            setattr(module, name, wrapped)


###############################################################################
#                     The main "inject_lora" function                         #
###############################################################################


def inject_lora(
    model: nn.Module, num_lora_slots: int, rank: int = 4, alpha: float = 1.0
):
    """
    Modifies a DiT model *in-place* so that:
      1) All nn.Linear layers in the attention, Mlp, and final layer become MultiLoRALinear.
      2) The forward passes of Attention, Mlp, DiTBlock, and FinalLayer accept a `lora_indices` arg.
      3) The DiT forward pass also accepts `lora_indices` and passes it down.
    """
    # 1) Replace nn.Linear with MultiLoRALinear
    for block in model.blocks:
        replace_linear_with_lora(block.attn, num_lora_slots, rank, alpha)
        replace_linear_with_lora(block.mlp, num_lora_slots, rank, alpha)
    replace_linear_with_lora(model.final_layer, num_lora_slots, rank, alpha)

    # 2) Patch forward methods
    for block in model.blocks:
        patch_attention_forward(block.attn)
        patch_mlp_forward(block.mlp)
        patch_ditblock_forward(block)
    patch_finallayer_forward(model.final_layer)

    # 3) Patch the DiT forward to accept lora_indices
    import types

    original_dit_forward = model.forward

    def forward_with_lora(self, x, t, lora_indices=None, **kwargs):
        y = kwargs.get("y", None)
        # Standard DiT logic
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)

        if self.num_classes > 0 and y is not None:
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for blk in self.blocks:
            x = blk(x, c, lora_indices=lora_indices)
        x = self.final_layer(x, c, lora_indices=lora_indices)
        x = self.unpatchify(x)
        return x

    model.forward = types.MethodType(forward_with_lora, model)
    return model


###############################################################################
#                         Utility: Disable/Enable Adapters                     #
###############################################################################
def disable_all_adapters(model: nn.Module):
    """
    Recursively set alpha=0 for every MultiLoRALinear in model.
    This effectively disables LoRA injection.
    """
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.disable_adapter()


def enable_all_adapters(model: nn.Module):
    """
    Recursively restore alpha to the original alpha for every MultiLoRALinear in model.
    """
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.enable_adapter()


def reset_all_lora_parameters(model: nn.Module):
    """
    Re-initialize lora_A and lora_B for every MultiLoRALinear in model.
    """
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.reset_parameters()


def set_lora_trainability(model: nn.Module, trainable: bool):
    """
    Toggle requires_grad for the LoRA A/B parameters in all MultiLoRALinear submodules.
    """
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.set_trainable(trainable)


def save_lora_adapters(model: nn.Module, save_path: str):
    """
    Saves only the LoRA parameters (A and B) from each MultiLoRALinear to a file.
    The base weights are not saved.
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.clone()
    torch.save(lora_state, save_path)


def load_lora_adapters(model: nn.Module, load_path: str, strict: bool = True):
    """
    Loads LoRA A/B parameters from a file and copies them into the model.
    If strict=True, will raise an error if keys don't match.
    """
    lora_state = torch.load(load_path, map_location="cpu")
    missing_keys = []

    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            keyA = f"{name}.lora_A"
            keyB = f"{name}.lora_B"
            if keyA in lora_state and keyB in lora_state:
                # copy in
                module.lora_A.data.copy_(lora_state[keyA])
                module.lora_B.data.copy_(lora_state[keyB])
            else:
                missing_keys.append((keyA, keyB))

    if strict and len(missing_keys) > 0:
        msg = "Missing LoRA keys in loaded state:\n"
        msg += "\n".join([str(k) for k in missing_keys])
        raise RuntimeError(msg)


###############################################################################
#                 The modulate function from your original code               #
###############################################################################


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


if __name__ == "__main__":
    from src.models.components.dit import DiT_S_2  # or your local DiT factory

    # 1) Create a base DiT as usual
    model = DiT_S_2(
        in_channels=4,
        num_classes=1000,
    )

    # 2) Inject multi-LoRA
    model = inject_lora(model, num_lora_slots=4, rank=4, alpha=0.1)

    # 3) Example input
    x = torch.randn(8, 4, 32, 32)
    t = torch.randint(0, 1000, (8,))
    y = torch.randint(0, 1000, (8,))
    lora_indices = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

    # 4) Test forward
    out = model(x, t, y=y, lora_indices=lora_indices)
    print("Output shape:", out.shape)

    # 5) Disable all adapters
    disable_all_adapters(model)
    out_disabled = model(x, t, y=y, lora_indices=lora_indices)
    print("Adapters disabled. Output shape:", out_disabled.shape)

    # 6) Re-enable all adapters
    enable_all_adapters(model)
    out_enabled_again = model(x, t, y=y, lora_indices=lora_indices)
    print("Adapters re-enabled. Output shape:", out_enabled_again.shape)

    # 7) Freeze LoRA layers
    set_lora_trainability(model, trainable=False)

    # 8) Reset LoRA parameters
    reset_all_lora_parameters(model)

    # 9) Save & load adapters
    save_lora_adapters(model, "my_lora_adapters.pth")
    # (modify or re-init model as needed)
    load_lora_adapters(model, "my_lora_adapters.pth", strict=True)
    print(0)
