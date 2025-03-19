import torch
from insert_memory import DiT3D


model = DiT3D(
    in_channels= 6,
    out_channels= 3,
    input_size= 64,
    num_classes= 1,
    depth= 12,
    hidden_size= 192,
    num_heads= 6,
    patch_size= 4,
    batch_size= 256, 
    chunk_size= 256,
)

from peft import LoraConfig, TaskType, get_peft_model
lora_config = LoraConfig(
    r=8,                  # rank
    lora_alpha=32,        # alpha scaling
    lora_dropout=0.05,    # dropout, can be 0.0 as well
    bias="none",
    target_modules=["qkv", "proj"], 
    #task_type=TaskType.FEATURE_EXTRACTION  
)
model = get_peft_model(model, lora_config)
x = torch.randn(4, 2, 6, 64, 64)
timesteps = torch.randint(0, 64, (4,))
output = model(x, timesteps)
loss = (output[0].sum() -  2)**2 
loss.backward()

# Checking if LoRA layers receive gradients:
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        print(name, param.grad.abs().sum())
print(0)


