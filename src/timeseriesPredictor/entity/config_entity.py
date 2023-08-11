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
    local_train_OD_dir: Path
    local_val_OD_dir: Path
    local_test_OD_dir: Path
    local_train_tensor_dir: Path
    local_val_tensor_dir: Path
    local_test_tensor_dir: Path

@dataclass
class PrepareAutoencoderBaseModelConfig:
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