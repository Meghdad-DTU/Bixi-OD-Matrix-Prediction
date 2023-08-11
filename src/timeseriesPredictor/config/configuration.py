import os
from timeseriesPredictor.constants import *
from timeseriesPredictor.utils import read_yaml, create_directories
from timeseriesPredictor.entity.config_entity import (DataIngestionConfig,
                                                      DataTransformationConfig,
                                                      PrepareAutoencoderBaseModelConfig,
                                                      PrepareCallbacksConfig)
                                                      

class configurationManeger:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 secret_filepath = SECRET_FILE_PATH,                 
                 params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath) 
        self.secret = read_yaml(secret_filepath)        
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion   
        secret = self.secret.aws_credential

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            s3_bucket = secret.s3_bucket,
            s3_key = secret.s3_key,
            s3_secret_key = secret.s3_secret_key,
            object_key = secret.object_key,
            local_data_file = config.local_data_file
            

        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation             

        create_directories([config.root_dir])

        data_trnsformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            local_data_file=self.config.data_ingestion.local_data_file,           
            local_train_od_dir= config.local_train_od_dir,            
            local_test_od_dir= config.local_test_od_dir,
            local_scaler_od_dir=config.local_scaler_od_dir,
            local_train_tensor_dir= config.local_train_tensor_dir,            
            local_test_tensor_dir= config.local_test_tensor_dir,
            local_scaler_tensor_dir= config.local_scaler_tensor_dir         

        )

        return data_trnsformation_config
    
    def get_prepare_autoencoder_base_model_config(self) -> PrepareAutoencoderBaseModelConfig:
        config = self.config.prepare_autoencoder_base_model
        
        create_directories([config.root_dir])

        prepare_autoencoder_base_model_config = PrepareAutoencoderBaseModelConfig(
            root_dir = config.root_dir,           
            base_od_model_path = config.base_od_model_path,    
            base_tensor_model_path = config.base_tensor_model_path,                   
            params_od_size = self.params.OD_SIZE,
            params_tensor_size= self.params.TENSOR_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,          

        )

        return prepare_autoencoder_base_model_config
    

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.ckeckpoint_model_filepath)

        create_directories([config.tensorboard_root_log_dir, model_ckpt_dir ])

        prepare_callbacks_config = PrepareCallbacksConfig(
           root_dir= config.root_dir,
           tensorboard_root_log_dir= config.tensorboard_root_log_dir,
           ckeckpoint_model_filepath=  config.ckeckpoint_model_filepath,
           patience = self.params.PATIENCE

        )

        return prepare_callbacks_config