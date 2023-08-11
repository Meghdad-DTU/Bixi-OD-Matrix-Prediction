
import sys
import os
import pandas as pd
from box import ConfigBox

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from timeseriesPredictor.config.configuration import DataTransformationConfig
from timeseriesPredictor.utils import OD_tensor_matrix, OD_tensor_matrix, train_test_split, save_pickle
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
                      FunctionTransformer(train_test_split, kw_args={'train_test_ratio':0.75}))
                ]
            )

            tensor_matrix_pipeline = Pipeline(
                steps=[
                    ('matrix_creation', 
                    FunctionTransformer(OD_tensor_matrix, kw_args={'tensor': True})),
                    ('train_val_test_split', 
                      FunctionTransformer(train_test_split, kw_args={'train_test_ratio':0.75}))
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
            matrix_train, matrix_test = preprocessing_obj.OD_matrix_pipeline.fit_transform(df)             
            scaler = MinMaxScaler()
            l, m, n , c = matrix_train.shape
            scaled_matrix_train = scaler.fit_transform(matrix_train.reshape(l, m*n)).reshape(l, m, n , c) 
            l, m, n , c = matrix_test.shape
            scaled_matrix_test = scaler.transform(matrix_test.reshape(l, m*n)).reshape(l, m, n , c)
            
            save_pickle(path= self.config.local_train_od_dir, obj= scaled_matrix_train)
            save_pickle(path = self.config.local_test_od_dir, obj= scaled_matrix_test)
            save_pickle(path= self.config.local_scaler_od_dir, obj= scaler)               
  

            logging.info(f"Applying tensor preprocessing object on dataframe")
            matrix_train, matrix_test = preprocessing_obj.tensor_matrix_pipeline.fit_transform(df)
            scaler = MinMaxScaler()
            l, m, n , c = matrix_train.shape
            scaled_matrix_train = scaler.fit_transform(matrix_train.reshape(l, m*n)).reshape(l, m, n , c) 
            l, m, n , c = matrix_test.shape
            scaled_matrix_test = scaler.transform(matrix_test.reshape(l, m*n)).reshape(l, m, n , c) 
              
            save_pickle(path= self.config.local_train_tensor_dir , obj= scaled_matrix_train)
            save_pickle(path= self.config.local_test_tensor_dir , obj= scaled_matrix_test)
            save_pickle(path= self.config.local_scaler_tensor_dir, obj= scaler)
             
