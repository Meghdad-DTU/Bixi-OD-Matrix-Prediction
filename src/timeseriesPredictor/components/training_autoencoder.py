import keras
from pathlib import Path
import pickle
from timeseriesPredictor.utils import model_loss
from timeseriesPredictor.entity.config_entity import TrainingAutoencoderConfig

class TrainingAutoencoder:
    def __init__(self, config: TrainingAutoencoderConfig):
        self.config= config

    @staticmethod
    def get_base_model(path:Path):
        model = keras.models.load_model(path)
        return model  

    @staticmethod
    def load_pickle_file(path: Path):
        file= open(path, 'rb')
        obj = pickle.load(file)
        return obj
    
    @staticmethod
    def save_model(path:Path, model:keras.Model):
        model.save(path)

    def train(self, callback_list: list): 
        X_trains = [self.load_pickle_file(self.config.training_od_data) , self.load_pickle_file(self.config.training_tensor_data)]
        base_model_paths = [self.config.base_od_model_path, self.config.base_tensor_model_path] 
        trained_model_paths = [self.config.trained_od_model_path, self.config.trained_tensor_model_path]       
        
        for base_model_path, X_train, trained_model_path in zip(base_model_paths, X_trains, trained_model_paths):
            model = self.get_base_model(base_model_path) 
            history = model.fit(X_train, X_train,
                validation_split=self.config.validation_ratio,           
                epochs= self.config.params_epochs,
                batch_size = self.config.params_batch_size,            
                callbacks = callback_list
                )
        
            model_loss(history)
            self.save_model(path=trained_model_path, model= model)