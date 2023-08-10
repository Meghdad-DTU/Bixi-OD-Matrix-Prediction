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