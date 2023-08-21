import sys
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import configurationManeger
from timeseriesPredictor.components.evaluation_timeseries_model import TimeseriesEvaluation


STAGE_NAME = "Timeseries Evaluation Model Stage"

class TimeseriesModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        evaluation_config = config.get_timeseries_evaluation_config()
        evaluation = TimeseriesEvaluation(evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = TimeseriesModelEvaluationPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)