import keras
import pickle
from pathlib import Path
from timeseriesPredictor.entity.config_entity import TimeseriesModelEvaluationConfig
from timeseriesPredictor.logger import logging
from timeseriesPredictor.utils import videos_frame_format, save_json, evaluate_forecasts




class TimeseriesEvaluation:
    def __init__(self, config: TimeseriesModelEvaluationConfig):
        self.config = config

    
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
    
    def save_score(self):
        scores = {'Average RMSE': self.best_rmse, 'Average MAE': self.best_mae}
        save_json(path='timeseries_scores.json', data=scores)

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
    
    def evaluation(self):
        scalers = [self.load_pickle_file(self.config.scaler_od),
                   self.load_pickle_file(self.config.scaler_tensor)]
        
        X_test_paths = [self.config.test_od_data,
                        self.config.test_tensor_data]
        
        trained_autoencoder_model_paths = [self.config.trained_od_autoencoder_model_path,
                                           self.config.trained_tensor_autoencoder_model_path]
        
        trained_timeseries_model_paths = [self.config.trained_od_timeseries_model_path,
                                          self.config.trained_tensor_timeseries_model_path] 
        
        best_rmse = 100000
        for scaler, X_test_path, trained_autoencoder_path, trained_timeseries_path in zip(scalers,
                                                                                                X_test_paths,
                                                                                                trained_autoencoder_model_paths,
                                                                                                trained_timeseries_model_paths):
            videos, next_frame = self.data_preparation(X_test_path, trained_autoencoder_path)
            model = self.get_model(trained_timeseries_path)
            prediction = model.predict(videos)
            l,m,n,_ = prediction.shape
            pred_test = prediction.reshape(l,m*n)
            true_test = next_frame.reshape(l,m*n)
            avg_mae, _ , avg_rmse, _ = evaluate_forecasts(scaler.inverse_transform(true_test), scaler.inverse_transform(pred_test), "Test")
            
            if best_rmse> avg_rmse:
               self.best_rmse = avg_rmse
               self.best_mae = avg_mae
               best_timeseries_model_path = trained_timeseries_path
        
        logging.info(f'Best timeseries model based on the least average RMSE is stored at: {best_timeseries_model_path}') 
        