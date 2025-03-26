import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange

# functions


def l2norm(t):
    return F.normalize(t, dim=-1)


# norms


class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = rearrange(gamma, "b d -> b 1 d")

        return self.ln(x) * (gamma + 1.0)


# norm + residual wrapper, as used in original TTT paper
# but could be removed


class ResidualNorm(Module):
    def __init__(self, dim, model: Module):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model

    def forward(self, x, *args):

        out = self.model(x, *args)

        return self.norm(out) + x


# memory mlp proposed in TTT


class MemoryMLP(Module):
    def __init__(self, dim, depth, expansion_factor=2.0):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        self.weights = ParameterList(
            [
                Parameter(torch.randn(dim_in, dim_out))
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight

        return x


# memory mlp, but with gated residual + final projection


class GatedResidualMemoryMLP(Module):
    def __init__(self, dim, depth, expansion_factor=4.0):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList(
            [
                ParameterList(
                    [
                        Parameter(torch.randn(dim, dim_hidden)),
                        Parameter(torch.randn(dim_hidden, dim)),
                        Parameter(torch.randn(dim * 2, dim)),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.final_proj = Parameter(torch.randn(dim, dim))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x):

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim=-1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj


# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes


class FactorizedMemoryMLP(Module):
    def __init__(self, dim, depth, k=32):
        super().__init__()
        self.weights = ParameterList(
            [
                ParameterList(
                    [
                        Parameter(torch.randn(dim, k)),
                        Parameter(torch.randn(k, dim)),
                    ]
                )
                for _ in range(depth)
            ]
        )

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(self, x):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x


# an MLP modelled after the popular swiglu ff in modern transformers


class MemorySwiGluMLP(Module):
    def __init__(
        self,
        dim,
        depth=1,  # default to 2 layer MLP from TTT, depth of 2 would be 4 layer MLP, but done as 2 feedforwards with residual
        expansion_factor=4.0,
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            weights.append(
                ParameterList(
                    [
                        Parameter(torch.randn(dim, dim_inner * 2)),
                        Parameter(torch.randn(dim_inner, dim)),
                    ]
                )
            )

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x

            x, gates = (x @ w1).chunk(2, dim=-1)

            x = x * F.gelu(gates)

            x = x @ w2

            x = x + residual

        return self.norm(x)


# improvised attention as memory module


class MemoryAttention(Module):
    def __init__(self, dim, scale=8.0, expansion_factor=2.0):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = ParameterList(
            [
                Parameter(torch.randn(dim, dim)),  # queries
                Parameter(torch.randn(dim, dim)),  # keys
                Parameter(torch.randn(dim, dim)),  # values
                Parameter(torch.randn(dim, dim_ff_hidden)),  # ff w1
                Parameter(torch.randn(dim_ff_hidden, dim)),  # ff w2
            ]
        )

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return attn_out + ff_out


import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryMoE2(nn.Module):
    def __init__(self, dim, num_experts, depth, expansion_factor=2.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.depth = depth

        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        # For each expert, we store a ParameterList of [W0, W1, ..., W_{depth-1}],
        # each W_i is shape (batch_size, in_features, out_features) *if* you truly
        # want a different weight "row" per sample. Typically, though, you'd just do
        # (in_features, out_features).
        #
        # But since the question code has "temp = temp @ weight[j]" and the user says
        # "each batch gets processed by a different weight," it implies W_i might have
        # shape (batch_size, in_dim, out_dim). Adjust as needed!
        #
        # Below we assume that each W_i is shape (batch_size, in_features, out_features)
        # so that weight_i[j] is the j-th slice of that matrix. If that's *not* what
        # you want, remove the extra dimension and skip the j-based indexing.

        # For demonstration, let's pretend we do:
        #   W_i has shape (batch_size, dims[l], dims[l+1])
        #   so that "weight[j]" is shape (dims[l], dims[l+1]).
        #
        # Often you'd do normal (dims[l], dims[l+1]) with no batch dimension.

        # If you do NOT actually have a per-batch dimension in your weights,
        # you can simplify this even further (just normal weight shapes).

        # We'll omit the actual "batch_size" dimension here, since in your snippet
        # you did not explicitly show it being allocated. Just keep in mind that if
        # you do have it, the shape below has to match that.

        weights = []
        for _ in range(num_experts):
            weight_ = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(dim_in, dim_out))
                    for dim_in, dim_out in zip(dims[:-1], dims[1:])
                ]
            )
            for w in weight_:
                nn.init.xavier_uniform_(w)
            weights.append(weight_)
        self.weights = nn.ParameterList(weights)

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,  # shape (num_seq, num_experts)
        routing_indices: torch.Tensor,  # shape (num_seq, top_k)
    ) -> torch.Tensor:
        """
        x:               (n_seq, dim)
        routing_weights: (n_seq, num_experts)
        routing_indices: (n_seq, top_k) with expert ids selected per sample
        """

        N, D = x.shape
        counts = torch.bincount(routing_indices.flatten(), minlength=self.num_experts)
        for i in range(self.num_experts):
            if counts[i] == 0:
                continue

            # expert = self.experts[i]
            # idx, top = torch.where(routing_indices == i)
            idx, top = torch.where(routing_indices == i)
            ## x[idx] should be bs, n_tokens, dim
            # y[idx] += expert(x[idx]) * routing_weights[idx, top]

            expert_weights = self.weights[i]
            for ind, weight in enumerate(expert_weights):
                is_first = ind == 0
                if not is_first:
                    x[idx] = F.gelu(x[idx])
                x[idx] = x[idx] @ weight[idx]  # here is the problem !!!!!
            y[idx] += x[idx] * routing_weights[idx, top, None]
        return y.view(shape)

        y = torch.zeros_like(x)  # accumulate mixture results here

        # If top_k=1, then routing_indices[b] is just a single expert for sample b.
        # If top_k>1, you may need to handle multiple experts per sample, etc.

        # ---------------------------------------------------------------------
        # 1) Find for each expert i the batch-indices that route to it.
        #    We’ll build a list idx_expert[i], containing all batch positions
        #    b s.t. routing_indices[b, *] == i for some “top_k” slot.
        # ---------------------------------------------------------------------
        # Example for top_k=1:
        #     idx_expert[i] = torch.where(routing_indices.squeeze(-1) == i)[0]
        #
        # Example for general top_k:
        #     idx_expert[i] = (routing_indices == i).nonzero(as_tuple=True)[0]
        # ---------------------------------------------------------------------

        # We can do it the general way (works for top_k=1 or bigger):
        # Flatten out the batch x top_k into one dimension, then gather unique.
        # The snippet below is a straightforward (though not always memory-optimal)
        # method:
        all_expert_ids = routing_indices.view(-1)  # shape = (B * top_k,)
        all_batch_ids = torch.arange(B).unsqueeze(-1).expand_as(routing_indices)
        all_batch_ids = all_batch_ids.reshape(-1)  # shape = (B * top_k,)

        # For each expert i, collect the b’s that route there.
        idx_expert = [[] for _ in range(self.num_experts)]
        for b_i, e_i in zip(all_batch_ids.tolist(), all_expert_ids.tolist()):
            # e_i is the expert ID, b_i is the sample ID
            idx_expert[e_i].append(b_i)

        # At this point, idx_expert[i] is the list of sample-IDs that route to
        # expert i. Many could be empty. We can convert them to tensors:
        idx_expert = [torch.tensor(v, device=x.device) for v in idx_expert]

        # ---------------------------------------------------------------------
        # 2) For each expert i: gather the relevant x's in a batch, pass them
        #    through that expert's stacked-layers, multiply by gating weight,
        #    and scatter back into y.
        # ---------------------------------------------------------------------
        for i, expert_weight_list in enumerate(self.weights):
            # Which samples go to this expert?
            idx_i = idx_expert[i]
            if idx_i.numel() == 0:
                # None of the samples use this expert
                continue

            # X_i: shape (n_i, D) where n_i = number of samples that route to expert i
            X_i = x[idx_i, :]

            for layer_idx, W in enumerate(expert_weight_list):
                if layer_idx > 0:
                    X_i = F.gelu(X_i)
                # shape of W: (D_in, D_out)
                # shape of X_i: (n_i, D_in)
                X_i = X_i @ W  # result: (n_i, D_out)

            # Multiply by gating
            # routing_weights has shape (B, num_experts).
            # We want routing_weights[idx_i, i], shape (n_i,)
            gate_i = routing_weights[idx_i, i]  # shape (n_i,)
            # Broadcast multiply across last dim
            X_i = X_i * gate_i.unsqueeze(-1)  # shape (n_i, D_out)

            # Finally, accumulate into y
            # y has shape (B, D). We scatter-add for these samples:
            y[idx_i] += X_i

        return y


class MemoryMoE(Module):
    def __init__(
        self,
        dim,
        num_experts,
        depth,
        expansion_factor=2.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.depth = depth
        self.expansion_factor = expansion_factor

        # self.expert_gate = nn.Linear(dim, num_experts)
        # self.experts = nn.ModuleList([MemoryMLP(dim, depth, expansion_factor) for _ in range(num_experts)])

        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        weights = []
        for _ in range(num_experts):
            weight_ = ParameterList(
                [
                    Parameter(torch.randn(dim_in, dim_out))
                    for dim_in, dim_out in zip(dims[:-1], dims[1:])
                ]
            )
            for weight in weight_:
                nn.init.xavier_uniform_(weight)
            weights.append(weight_)
        self.weights = ParameterList(weights)

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        routing_indices: torch.Tensor,
        aux_loss=False,
    ) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): shape (batch_size, n_seq, dim). The input to be routed.
            routing_weights (torch.Tensor): shape (batch_size, n_seq, num_experts).
                Pre-computed gating weights for each expert.

        Returns:
            torch.Tensor: shape (batch_size, n_seq, dim), mixture of expert outputs.
        """
        shape = x.size()
        # x = x.view(-1, self.dim)
        # weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(
            routing_indices.flatten(), minlength=self.num_experts
        )  # .tolist()

        for j in range(shape[0]):
            counts = torch.bincount(
                routing_indices[j].flatten(), minlength=self.num_experts
            )  # .tolist()
            for i in range(self.num_experts):
                if counts[i] == 0:
                    continue
                top = torch.where(routing_indices[j].squeeze(-1) == i)
                expert_weights = self.weights[i]
                temp = x[j]
                for ind, weight in enumerate(expert_weights):
                    is_first = ind == 0
                    if not is_first:
                        temp = F.gelu(temp)
                    temp = temp @ weight[j]

                y[j] += temp * routing_weights[j, top[0]]

        return y.view(shape)
