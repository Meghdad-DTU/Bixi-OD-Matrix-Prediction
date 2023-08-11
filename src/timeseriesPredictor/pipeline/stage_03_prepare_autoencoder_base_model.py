
import sys
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import configurationManeger
from timeseriesPredictor.components.prepare_autoencoder_base_model import PrepareAutoencoderBaseModel


STAGE_NAME = "Prepare Autoencoder Base Model Stage"

class PrepareAutoencoderBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        prepare_auto_base_model_config = config.get_prepare_autoencoder_base_model_config()
        prepare_auto_base_model = PrepareAutoencoderBaseModel(config=prepare_auto_base_model_config)    
        prepare_auto_base_model.update_model()


if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = PrepareAutoencoderBaseModelTrainingPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)