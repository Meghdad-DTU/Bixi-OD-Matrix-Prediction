import sys
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import configurationManeger
from timeseriesPredictor.components.prepare_timeseries_base_model import PrepareTimeseriesBaseModel


STAGE_NAME = "Prepare Timeseries Base Model Stage"

class PrepareTimeseriesBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        prepare_timeseries_base_model_config = config.get_prepare_timeseries_base_model_config()
        prepare_timeseries_base_model = PrepareTimeseriesBaseModel(config=prepare_timeseries_base_model_config)    
        prepare_timeseries_base_model.update_model()


if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = PrepareTimeseriesBaseModelTrainingPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)