import os
from timeseriesPredictor.config.configuration import PrepareCallbacksConfig
import keras
import time

class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    #Enable visualizations for TensorBoard.
    @property
    def _create_tb_callbacks(self): 
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
            )   
        return keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
        
    #ModelCheckpoint callback is used in conjunction with training using model. fit() to save a model or weights (in a checkpoint file) //
    # at some interval, so the model or weights can be loaded later to continue the training from the state saved.    
    @property
    def _create_ckpt_callbacks(self):
        return keras.callbacks.ModelCheckpoint(
                filepath = self.config.ckeckpoint_model_filepath,
                save_best_only = True)
    
    @property
    def _create_es_callbacks(self):
        return keras.callbacks.EarlyStopping(
                monitor = 'val_loss',
                patience = self.config.patience)


    def get_tb_ckpt_es_callbacks(self):
        return [self._create_tb_callbacks, self._create_ckpt_callbacks, self._create_es_callbacks]