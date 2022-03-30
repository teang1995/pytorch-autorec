import hydra
import logging
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from autorec.datasets.datamodule import MovieLensDataModule
from autorec.model.autorec_module import AutoRecModule
from autorec.path import CHECKPOINT_DIR, LOGGING_DIR, DATA_DIR


@hydra.main(config_path="config", config_name="ml-1m-item")
def main(cfg: DictConfig):
    # load paths from path.py
    data_root = DATA_DIR
    checkpoint_dir = CHECKPOINT_DIR
    logging_dir = LOGGING_DIR

    # load configs from hydra config file
    datamodule_config = cfg['data_module']
    trainmodule_config = cfg['train_module']
    train_config = cfg['train']

    # datamodule_config
    batch_size = datamodule_config['batch_size']
    data_size = datamodule_config['data_size']
    model_type = datamodule_config['model_type']
    valid_ratio = datamodule_config['valid_ratio']

    # trainmodule_config
    init_lr = trainmodule_config['init_lr']
    hidden_size = trainmodule_config['hidden_size']
    optimizer = trainmodule_config['optimizer']

    # train_config
    num_epochs = train_config['num_epochs']
    device = train_config['device']

    # data_config 
    data_config = cfg[f'movielens_{data_size}']
    num_users = data_config['num_users']
    num_items = data_config['num_items']

    # set model_size
    input_size = num_users if model_type == 'item' else num_items

    logger = TensorBoardLogger(
                            save_dir=logging_dir,
                            name=f'ml-{data_size}-{model_type}-logs'
                            )

    # TODO: how to use modelcheckpoint by rmse (custom loss)?
    # First, I will save weight of the last epoch. 
    checkpoint_callback = ModelCheckpoint(
                                        dirpath=checkpoint_dir,
                                        filename=f'ml-{data_size}-{model_type}',
                                        mode='max'
                                        )

    early_stop_callback = EarlyStopping(
                                    monitor='validation_loss',  
                                    min_delta=1e-4,  
                                    patience=20,  
                                    mode='min'
                                    )
    # declare data module
    data_module = MovieLensDataModule(batch_size=batch_size,
                                      data_root=data_root,
                                      data_size=data_size,
                                      device=device,
                                      model_type=model_type,
                                      valid_ratio=valid_ratio) 

    # declare train module
    train_module = AutoRecModule(hidden_size=hidden_size,
                                 init_lr=init_lr,
                                 input_size=input_size,
                                 optimizer=optimizer)
    # declare pl trainer
    trainer = pl.Trainer(gpus=device,
                         max_epochs=num_epochs,
                         callbacks=[early_stop_callback,
                                    checkpoint_callback],
                         logger=logger,
                         log_every_n_steps=10)
    
    # trainer.fit
    trainer.fit(model=train_module,
                datamodule=data_module)
                
if __name__ == "__main__":
    main()