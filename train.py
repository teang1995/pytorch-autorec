import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

HYDRA_FULL_ERROR=1 

# TODO: get hyperparameter using hydra
@hydra.main(config_path="config", config_name="train_config")
def main(cfg: DictConfig):
    hp_config = cfg['train']
    data_config = cfg['dataset']
    
    epoch = hp_config['epoch']
    batch_size = hp_config['batch_size']
    optimizer = hp_config['optimizer']

    train_path = data_config['train_path']
    valid_path = data_config['valid_path']

    print(epoch)
    print(batch_size)
    print(optimizer)
    print(train_path)
    print(valid_path)

if __name__ == "__main__":
    main()