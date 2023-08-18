import sys
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import configurationManeger
from timeseriesPredictor.components.prepare_callbacks import PrepareCallbacks
from timeseriesPredictor.components.training_timeseries_model import TrainingTimeseries


STAGE_NAME = "Training Timeseries Model Stage"

class TimeseriesModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_es_callbacks()
        
        training_config = config.get_timeseries_training_config()
        training = TrainingTimeseries(config= training_config)   
        training.train(callback_list= callback_list)



if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = TimeseriesModelTrainingPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)