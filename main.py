import sys
from timeseriesPredictor.logger import logging
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from timeseriesPredictor.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from timeseriesPredictor.pipeline.stage_03_prepare_cnn_autoencoder_base_model import PrepareCNNAutoencoderBaseModelTrainingPipeline
from timeseriesPredictor.pipeline.stage_04_training_cnn_autoencoder import CNNAutoencoderModelTrainingPipeline
from timeseriesPredictor.pipeline.stage_05_evaluating_cnn_autoencoder import CNNAutoencoderEvaluationPipeline
from timeseriesPredictor.pipeline.stage_06_prepare_timeseries_base_model import PrepareTimeseriesBaseModelTrainingPipeline
from timeseriesPredictor.pipeline.stage_07_training_timeseries_model import TimeseriesModelTrainingPipeline
from timeseriesPredictor.pipeline.stage_08_evaluation_timeseries_model import TimeseriesModelEvaluationPipeline


'''STAGE_NAME = "Data Ingestion Stage"

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
    raise CustomException(e, sys)


STAGE_NAME = "Prepare CNN Autoencoder Base Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = PrepareCNNAutoencoderBaseModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

STAGE_NAME = "CNN Autoencoder Training Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = CNNAutoencoderModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

STAGE_NAME = "CNN Autoencoder Evaluation Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = CNNAutoencoderEvaluationPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

STAGE_NAME = "Prepare Timeseries Base Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = PrepareTimeseriesBaseModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

STAGE_NAME = "Training Timeseries Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = TimeseriesModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)'''


STAGE_NAME = "Timeseries Evaluation Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = TimeseriesModelEvaluationPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)