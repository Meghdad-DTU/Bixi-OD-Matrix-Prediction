import sys
from timeseriesPredictor.logger import logging
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from timeseriesPredictor.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from timeseriesPredictor.pipeline.stage_03_prepare_autoencoder_base_model import PrepareAutoencoderBaseModelTrainingPipeline


"""STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME = "Data Transformation Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)"""


STAGE_NAME = "Prepare Autoencoder Base Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = PrepareAutoencoderBaseModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)