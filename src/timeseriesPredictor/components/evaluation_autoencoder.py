import keras
import pickle
from pathlib import Path
from timeseriesPredictor.utils import evaluate_forecasts, save_json
from timeseriesPredictor.logger import logging
from timeseriesPredictor.entity.config_entity import AutoencoderEvaluationConfig

class AutoencoderEvaluation:
    def __init__(self, config: AutoencoderEvaluationConfig):
        self.config = config

    
    @staticmethod
    def get_trained_model(path:Path):
        model = keras.models.load_model(path)
        return model  

    @staticmethod
    def load_pickle_file(path: Path):
        file= open(path, 'rb')
        obj = pickle.load(file)
        return obj
    
    def save_score(self):
        scores = {'Average RMSE': self.best_rmse, 'Average MAE': self.best_mae}
        save_json(path='autoencoder_scores.json', data=scores)
    
    def evaluation(self):
        X_tests = [self.load_pickle_file(self.config.test_od_data) , self.load_pickle_file(self.config.test_tensor_data)]
        scalers = [self.load_pickle_file(self.config.scaler_od) , self.load_pickle_file(self.config.scaler_tensor)]
        trained_model_paths = [self.config.trained_od_model_path, self.config.trained_tensor_model_path] 
        
        best_rmse = 100000
        for trained_model_path, X_test , scaler in zip(trained_model_paths, X_tests, scalers):
            model = self.get_trained_model(trained_model_path)
            prediction = model.predict(X_test)
            l,m,n,_ = prediction.shape
            pred_test = prediction.reshape(l,m*n)
            true_test = X_test.reshape(l,m*n)
            avg_mae, _ , avg_rmse, _ = evaluate_forecasts(scaler.inverse_transform(true_test), scaler.inverse_transform(pred_test), "Test")
            
            if best_rmse> avg_rmse:
               self.best_rmse = avg_rmse
               self.best_mae = avg_mae
               best_autoencoder_path = trained_model_path
        
        logging.info(f'Best autoencoder model based on the least average RMSE is stored at: {best_autoencoder_path}') 