
import sys
import os
import pandas as pd
from box import ConfigBox

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from keras.preprocessing.image import save_img

from timeseriesPredictor.config.configuration import DataTransformationConfig
from timeseriesPredictor.utils import OD_tensor_matrix, train_validation_test
from timeseriesPredictor.exception import CustomException
from timeseriesPredictor.logger import logging


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        try:            
            OD_matrix_pipeline = Pipeline(
                steps=[
                    ('matrix_creation', 
                    FunctionTransformer(OD_tensor_matrix)),
                    ('train_val_test_split', 
                      FunctionTransformer(train_validation_test, kw_args={'train_test_ratio':0.75, 'train_val_ratio':0.15}))
                ]
            )

            tensor_matrix_pipeline = Pipeline(
                steps=[
                    ('matrix_creation', 
                    FunctionTransformer(OD_tensor_matrix, kw_args={'tensor': True})),
                    ('train_val_test_split', 
                      FunctionTransformer(train_validation_test, kw_args={'train_test_ratio':0.75, 'train_val_ratio':0.15}))
                ]
            )
            return  ConfigBox({'OD_matrix_pipeline': OD_matrix_pipeline, 
                               'tensor_matrix_pipeline': tensor_matrix_pipeline})           
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self):
        if not os.path.exists(self.config.local_data_file):
            logging.info(f"WARNING: {self.config.local_data_file} does not exist!")             
        
        else:  
            
            df = pd.read_csv(self.config.local_data_file)
            logging.info('Read data is completed!')
            df['start_date'] = pd.to_datetime(df['start_date'], format='ISO8601')
            logging.info('Changing object to datetime format is completed!')
           
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()  

            logging.info(f"Applying OD preprocessing object on dataframe")
            matrix_train, matrix_val, matrix_test = preprocessing_obj.OD_matrix_pipeline.fit_transform(df)

            
            for i, matrix in enumerate(matrix_train):                
                path_file = os.path.join(self.config.local_train_OD_dir, str(i)+'.png')
                save_img(path_file, matrix)            
            logging.info(f"Train OD set is saved as .png")   

            leght= len(matrix_train)
            for i, matrix in enumerate(matrix_val):                
                path_file = os.path.join(self.config.local_val_OD_dir, str(leght+i)+'.png')
                save_img(path_file, matrix)            
            logging.info(f"Validation OD set is saved as .png")  

            leght= len(matrix_train) + len(matrix_val)
            for i, matrix in enumerate(matrix_test):                
                path_file = os.path.join(self.config.local_test_OD_dir, str(leght+i)+'.png')
                save_img(path_file, matrix)            
            logging.info(f"Test OD set is saved as .png") 
  

            logging.info(f"Applying tensor preprocessing object on dataframe")
            matrix_train, matrix_val, matrix_test = preprocessing_obj.tensor_matrix_pipeline.fit_transform(df)
            
            for i, matrix in enumerate(matrix_train):                
                path_file = os.path.join(self.config.local_train_tensor_dir, str(i)+'.png')
                save_img(path_file, matrix)            
            logging.info(f"Train tensor set is saved as .png")   

            leght= len(matrix_train)
            for i, matrix in enumerate(matrix_val):                
                path_file = os.path.join(self.config.local_val_tensor_dir, str(leght+i)+'.png')
                save_img(path_file, matrix)            
            logging.info(f"Validation tensor set is saved as .png")  

            leght= len(matrix_train) + len(matrix_val)
            for i, matrix in enumerate(matrix_test):                
                path_file = os.path.join(self.config.local_test_tensor_dir, str(leght+i)+'.png')
                save_img(path_file, matrix)            
            logging.info(f"Test tensor set is saved as .png") 