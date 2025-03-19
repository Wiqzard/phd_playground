import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange, repeat, reduce, pack, unpack

from tensordict import TensorDict

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))


class MemoryMLP(nn.Module):
    def __init__(self, dim, depth, expansion_factor=2.0):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(dim_in, dim_out))
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


class TTTMoE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = 64

        num_experts = 8
        self.memory_model = MemoryMLP(self.dim, depth=2)

        
        mem_model_params = dict(self.memory_model.named_parameters())
        memory_model_parameters = [*mem_model_params.values()]
        memory_model_parameter_names = [*mem_model_params.keys()]
        memory_model_parameters = nn.ParameterList(memory_model_parameters)
        self.memory_model_parameter_dict = TensorDict(dict(zip(memory_model_parameter_names, memory_model_parameters)))


        self.to_keys = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
        )
        #self.W = nn.Parameter(torch.normal(0, 0.02, size=(num_experts, self.dim, self.dim)))
        #self.b = nn.Parameter(torch.zeros(num_experts, 1, self.dim))

        self.store_memory_loss_fn = nn.MSELoss()

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = torch.func.functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target)
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        self.forward_and_loss = forward_and_loss
        grad_fn = torch.func.grad(self.forward_and_loss, has_aux=True)
        self.per_sample_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, 0))

    def init_weights(
        self,
        batch,
    ):
        weights = repeat_dict_values(self.memory_model_parameter_dict, '... -> bh ...', bh = batch)
        return weights

    def forward(self, X):
        # X has shape (batch, seq_len, dim)

        weights = self.init_weights(X.shape[0])
        keys = self.to_keys(X)

        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(
            dict(weights), keys, keys, keys 
        )
        return grads, unweighted_mem_model_loss

if __name__ == "__main__":
    model = TTTMoE()
    model(torch.rand(1, 10, 64))
    print(0)
