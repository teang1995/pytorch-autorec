import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from autorec.model.autorec_module import AutoRecModule
from autorec.datasets.datamodule import MovieLensDataModule


HYDRA_FULL_ERROR=1 
@hydra.main(config_path="config", config_name="train_config")
def main(cfg: DictConfig):
    hp_config = cfg['train']
    data_config = cfg['dataset']
    
    epoch = hp_config['epoch']
    batch_size = hp_config['batch_size']
    optimizer = hp_config['optimizer']

    train_path = data_config['train_path']
    valid_path = data_config['valid_path']

    
    logger = TensorBoardLogger(
                            save_dir="autorec",  
                            name="movielens-autorec", 
                            default_hp_metric=False,
                            )

    checkpoint_callback = ModelCheckpoint(
                                        monitor='Accuracy',  # 어떤 지표를 기준으로 삼을 것인지
                                        dirpath='classification_ckpt',  # 체크포인트 파일들이 저장될 경로
                                        filename='cats&dogs_clssification',  # 체크포인트 파일의 이름(확장자 불요)
                                        mode='max'  # 위 지표가 최대일 때의 모델을 저장(min으로 설정 가능)
                                        )


    early_stop_callback = EarlyStopping(
                                        monitor='Accuracy',  # 어떤 지표를 기준으로 삼을 것인지
                                        min_delta=1e-4,  # 위 지표가 얼마나 향상이 되어야 하는지
                                        patience=20,  # 몇 에포크동안 지켜볼 것인지
                                        mode='max'
                                        )
    # TODO: add data_type, model_type to config
    data_type = '1M'
    model_type = 'item'
    

if __name__ == "__main__":
    main()