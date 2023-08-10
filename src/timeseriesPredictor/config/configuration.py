from timeseriesPredictor.constants import *
from timeseriesPredictor.utils import read_yaml, create_directories
from timeseriesPredictor.entity.config_entity import (DataIngestionConfig,
                                                      DataTransformationConfig)
                                                      

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

        create_directories([config.root_dir,                            
                            config.local_train_OD_dir,
                            config.local_train_OD_dir,
                            config.local_val_OD_dir,
                            config.local_test_OD_dir,
                            config.local_train_tensor_dir,
                            config.local_val_tensor_dir,
                            config.local_test_tensor_dir]
                            )

        data_trnsformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            local_data_file=self.config.data_ingestion.local_data_file,           
            local_train_OD_dir= config.local_train_OD_dir,
            local_val_OD_dir= config.local_val_OD_dir,
            local_test_OD_dir= config.local_test_OD_dir,
            local_train_tensor_dir= config.local_train_tensor_dir,
            local_val_tensor_dir= config.local_val_tensor_dir,
            local_test_tensor_dir= config.local_test_tensor_dir          

        )

        return data_trnsformation_config