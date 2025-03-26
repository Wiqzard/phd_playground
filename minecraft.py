import hydra
from omegaconf import OmegaConf

if __name__ == "__main__":

    # Load configuration
    path = "/home/ss24m050/Documents/phd_playground/configs/data/memory.yaml"
    cfg = OmegaConf.load(path)
    dataset = hydra.utils.instantiate(cfg)
    sample = dataset[0]
    print(0)
