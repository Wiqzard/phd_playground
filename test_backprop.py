import torch
import torch.nn as nn
import torch.optim as optim

from insert_memory import DiT3DTTT

torch.autograd.set_detect_anomaly(True)
# Assuming DiT3DTTT and scheduler are correctly imported and initialized
model = DiT3DTTT(
    depth=12,
    input_size=64,
    in_channels=6,
    hidden_size=768,
    patch_size=8,
    num_heads=12,
    max_frames=16,
    out_channels=3,
).cuda()
model.train()

# Optimizer and criterion
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()


# Dummy scheduler (replace with your scheduler)
class DummyScheduler:
    def __init__(self, num_timesteps=1000):
        self.config = type("", (), {"num_train_timesteps": num_timesteps})()
        self.alphas_cumprod = torch.linspace(1.0, 0.0, num_timesteps).cuda()

    def add_noise(self, x, noise, timesteps):
        return x + noise

    def get_velocity(self, sample, noise, timesteps):
        return sample - noise


scheduler = DummyScheduler()

# Dummy input
batch_size, frames, channels, height, width = 2, 2, 3, 64, 64
x = torch.randn(batch_size, frames, channels, height, width).cuda()

# Training step
optimizer.zero_grad()

# Forward pass (using simplified version of provided code)
b, t, c, h, w = x.shape
x_input = x.permute(0, 2, 1, 3, 4)
noise = torch.randn_like(x_input)
timesteps = torch.randint(
    0, scheduler.config.num_train_timesteps, (batch_size,), device=x.device
).long()
x_noisy = scheduler.add_noise(x_input, noise, timesteps)
y = torch.zeros((batch_size,), device=x.device, dtype=torch.long)

video = []
memory_states = None
for i in range(2):
    x_noisy_i = x_noisy[:, :, i].unsqueeze(2)
    padding = torch.zeros_like(x_noisy_i)
    frame_padding = torch.zeros_like(x_noisy_i)
    frame_padding[:, :, 0] = x_input[:, :, 0]
    x_noisy_i = torch.cat([x_noisy_i, frame_padding], dim=1)
    x_noisy_i = x_noisy_i.permute(0, 2, 1, 3, 4)

    if True:
        noise_pred, memory_states = model.forward(
            x_noisy_i,
            timesteps,
            cond=y,
            cache_params=memory_states,
            use_cache=True,
            run=i,
        )
    else:
        noise_pred = model.forward(
            x_noisy_i, timesteps, cond=y, cache_params=memory_states, use_cache=False
        )

    x_pred = scheduler.get_velocity(
        sample=noise_pred, noise=x_noisy_i[:, :, :3], timesteps=timesteps
    )
    video.append(x_pred)

x_pred = torch.cat(video, dim=1)
x_pred = x_pred.permute(0, 2, 1, 3, 4)

alphas_cumprod = scheduler.alphas_cumprod[timesteps]
weights = 1 / (1 - alphas_cumprod)
while len(weights.shape) < len(x_pred.shape):
    weights = weights.unsqueeze(-1)

diffusion_loss = criterion(x_pred * weights, x_input * weights)

# Backward pass
diffusion_loss.backward()
optimizer.step()

print("Loss:", diffusion_loss.item())
