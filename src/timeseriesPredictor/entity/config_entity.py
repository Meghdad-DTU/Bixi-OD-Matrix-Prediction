from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    s3_bucket: str
    s3_key: str
    s3_secret_key: str
    object_key: Path
    local_data_file: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    local_data_file: Path 
    local_train_od_dir: Path       
    local_test_od_dir: Path
    local_scaler_od_dir: Path 
    local_train_tensor_dir: Path   
    local_test_tensor_dir: Path
    local_scaler_tensor_dir: Path

@dataclass
class PrepareCNNAutoencoderBaseModelConfig:
    root_dir: Path    
    base_od_model_path: Path
    base_tensor_model_path: Path      
    params_od_size: list
    params_tensor_size: list
    params_learning_rate: float

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    ckeckpoint_model_filepath: Path
    patience: int

@dataclass(frozen=True)
class TrainingCNNAutoencoderConfig:    
    root_dir: Path
    trained_od_model_path: Path    
    base_od_model_path: Path
    trained_tensor_model_path: Path    
    base_tensor_model_path: Path     
    training_od_data: Path    
    training_tensor_data: Path    
    params_od_size: list
    params_tensor_size: list
    params_epochs: int
    params_batch_size: int   
    learning_rate: float
    validation_ratio: float

@dataclass(frozen=True)
class CNNAutoencoderEvaluationConfig:
    trained_od_model_path: Path
    trained_tensor_model_path: Path       
    test_od_data: Path
    test_tensor_data: Path 
    scaler_od: Path
    scaler_tensor: Path   
    params_od_size: list
    params_tensor_size: list
    
@dataclass(frozen=True)
class PrepareTimeseriesBaseModelConfig:
    root_dir: Path
    base_od_timeseries_model_path: Path
    base_tensor_timeseries_model_path: Path
    trained_od_autoencoder_path: Path
    trained_tensor_autoencoder_path: Path
    params_learning_rate: float
    params_time_lag : int

@dataclass(frozen=True)
class TrainingTimeseriesConfig:    
    root_dir: Path
    trained_od_timeseries_model_path: Path    
    base_od_timeseries_model_path: Path
    trained_tensor_timeseries_model_path: Path    
    base_tensor_timeseries_model_path: Path  
    trained_od_autoencoder_model_path: Path
    trained_tensor_autoencoder_model_path: Path   
    training_od_data: Path    
    training_tensor_data: Path    
    params_epochs: int
    params_batch_size: int   
    learning_rate: float
    validation_ratio: float
    params_time_lag: int

@dataclass(frozen=True)
class TimeseriesModelEvaluationConfig:
    trained_od_timeseries_model_path: Path
    trained_tensor_timeseries_model_path: Path 
    trained_od_autoencoder_model_path: Path
    trained_tensor_autoencoder_model_path: Path      
    test_od_data: Path
    test_tensor_data: Path 
    scaler_od: Path
    scaler_tensor: Path
    params_time_lag: int