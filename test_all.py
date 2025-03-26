import torch
from omegaconf import OmegaConf
import hydra


if __name__ == "__main__":

    data_cfg_path = "/home/ss24m050/Documents/phd_playground/configs/data/memory.yaml"
    data_cfg = OmegaConf.load(data_cfg_path)
    dataset = hydra.utils.instantiate(data_cfg)
    dataset.setup(stage="fit")
    sample = dataset.dataset_train[0]
    print(0)

    cfg_path = "/home/ss24m050/Documents/phd_playground/configs/debug.yaml"
    cfg = OmegaConf.load(cfg_path)
    model = hydra.utils.instantiate(cfg)
    in_tensor = torch.randn(2, 13, 4, 32, 32)
    timestep = torch.randint(0, 100, (2,))
    cond = torch.randint(0, 100, (2,))
    out = model(in_tensor, timestep, cond)

    print(0)
