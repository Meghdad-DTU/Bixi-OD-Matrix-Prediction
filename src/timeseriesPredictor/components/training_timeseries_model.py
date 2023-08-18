from pathlib import Path
import keras
import pickle
from timeseriesPredictor.logger import logging
from timeseriesPredictor.utils import videos_frame_format, model_loss
from timeseriesPredictor.config.configuration import TrainingTimeseriesConfig




class TrainingTimeseries:
    def __init__(self, config: TrainingTimeseriesConfig):
        self.config= config

    @staticmethod
    def get_model(path:Path):
        model = keras.models.load_model(path)
        logging.info(f'Model is loaded from {path}!')
        return model  

    @staticmethod
    def load_pickle_file(path: Path):
        file= open(path, 'rb')
        obj = pickle.load(file)
        logging.info(f'Pickle file is loaded from {path}!')
        return obj
    
    @staticmethod
    def save_model(path:Path, model:keras.Model):        
        model.save(path)

    def data_preparation(self, data_path: Path, autoencoder_path: Path):
        data = self.load_pickle_file(data_path)
        autoencoder = self.get_model(autoencoder_path)
        encoderLayer = autoencoder.get_layer("encoder")
        encoder = keras.Model(
            autoencoder.input,
            encoderLayer.output
            )   
        encoded_data = encoder.predict(data)
        videos, next_frame = videos_frame_format(encoded_data, data, self.config.params_time_lag)        
        return  videos, next_frame


    def train(self, callback_list: list): 
        
        training_data_paths = [self.config.training_od_data, 
                               self.config.training_tensor_data]
        
        trained_autoencoder_model_paths = [self.config.trained_od_autoencoder_model_path,
                                            self.config.trained_tensor_autoencoder_model_path]
        
        base_timeseries_model_paths = [self.config.base_od_timeseries_model_path, 
                                       self.config.base_tensor_timeseries_model_path] 
        
        trained_timeseries_model_paths = [self.config.trained_od_timeseries_model_path, 
                                          self.config.trained_tensor_timeseries_model_path]       
        
        for data_path, autoencoder_path, base_model_path, trained_model_path in zip(training_data_paths, 
                                                                                    trained_autoencoder_model_paths, 
                                                                                    base_timeseries_model_paths, 
                                                                                    trained_timeseries_model_paths):
            
            videos, next_frame = self.data_preparation(data_path=data_path, autoencoder_path=autoencoder_path)

            model = self.get_model(base_model_path) 

            history = model.fit(videos, next_frame,
                validation_split=self.config.validation_ratio,           
                epochs= self.config.params_epochs,
                batch_size = self.config.params_batch_size,            
                callbacks = callback_list
                )
        
            model_loss(history)
            self.save_model(path=trained_model_path, model= model)
            logging.info(f'Trained timeseries model is saved at {trained_model_path}!')