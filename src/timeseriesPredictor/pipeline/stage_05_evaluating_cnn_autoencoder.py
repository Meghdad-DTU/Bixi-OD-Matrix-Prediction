import sys
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import configurationManeger
from timeseriesPredictor.components.evaluation_cnn_autoencoder import CNNAutoencoderEvaluation


STAGE_NAME = "CNN Autoencoder Evaluatiing Model Performance Stage"

class CNNAutoencoderEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        evaluation_config = config.get_autoencoder_evaluation_config()
        evaluation = CNNAutoencoderEvaluation(evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()


if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = CNNAutoencoderEvaluationPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)