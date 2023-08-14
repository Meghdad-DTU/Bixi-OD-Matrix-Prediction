import os
import sys
import yaml
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeseriesPredictor.logger import logging
from timeseriesPredictor.exception import CustomException
from sklearn.metrics import mean_absolute_error, mean_squared_error

from typing import Any
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns
    Args:
        path to yaml (str): path like input
    Raises:
        valueError
    Returns:
        configBox: configBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f'yaml file {yaml_file.name} loaded successfully')
            return ConfigBox(content)
            
    except Exception as e:
        raise CustomException(e, sys)    

@ensure_annotations
def create_directories(path_to_directories: list, verbos=True):
    """
    create list of directories
    Args:
    path_to_directories(list): list of path of directories
    ignore_log(bool, optional): ignore if multiple dirs is to be created. Defaults to be False
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbos:
            logging.info(f'created directory at {path}')

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

#@ensure_annotations
def save_pickle(path: Path, obj:Any):
    """
    save object as pickel 

    Args:
        path (Path): path to .pkl 
        obj : object to be saved in the file             
    """
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"pickel file saved at {path}")
    
    except Exception as e:
            raise CustomException(e, sys)
    
#@ensure_annotations
def save_json(path:Path, data:dict):
    """
    save json data

    Args:
        path(Path): path to json file
        data (dict): data to be saved in json file        
    """

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at {path}")



def resampling(dat, tensor, window="1D"):
    """
    Resampling O-D daily, hourly or 15 mins
    Args:
        dat (DataFrame): Bixi dataframe including origin and distnation
        tensor (Boolean): if true it relates to tensor matrix else OD matrix
        window (str): time windows (1D, 60Min & 15Min)

    """ 
    if tensor:
        dat = dat.loc[:,["start_date","ki","kj"]].set_index("start_date")    
        dat["freq"] = np.repeat(1, len(dat))
        resampled_dat = dat.groupby([pd.Grouper(freq= window),"ki","kj"]).aggregate(np.sum).reset_index()

    else:
        dat = dat.loc[:,["start_date","zone_O","zone_D"]].set_index("start_date")    
        dat["freq"] = np.repeat(1, len(dat))
        resampled_dat = dat.groupby([pd.Grouper(freq= window),"zone_O","zone_D"]).aggregate(np.sum).reset_index()
    
    return resampled_dat  


def time_window_matrix(dat, n_origin, n_dist=None):
    """
    Creating O-D Matrix or tensor for each time window
    Args:
        dat (DataFrame: sub dataset for each timestamp
        n_origin (int): total number of origins
        n_dist (int): if not None, total number of distinations
    """
    if n_dist is not None:
        arr_org = np.arange(n_origin)
        arr_dis = np.arange(n_dist)
        origin, destination = arr_org.repeat(n_dist), np.tile(arr_dis,n_origin)
        columns = ["ki","kj"]
        o_size, d_size = n_origin, n_dist
    else:
        arr = np.arange(n_origin)
        origin, destination = arr.repeat(n_origin), np.tile(arr,n_origin)
        columns = ["zone_O","zone_D"]
        o_size, d_size = n_origin, n_origin        

    df1 = pd.DataFrame(np.vstack([origin, destination]).T)
    df1.columns = columns
    df2 = dat.loc[:,columns]
    # Find Rows in df1 Which Are Not Available in df2
    merge_df = df1.merge(df2, how = 'outer' ,indicator=True)
    
    freq = dat['freq'].values
    
    i=0
    trips = []
    for x in merge_df._merge.values:
        if x!= "both":
            trips.append(0)
        else:
            trips.append(freq[i])
            i+=1
    channel=1

    matrix=np.array(trips).reshape(o_size, d_size, channel) 
    return matrix


def OD_tensor_matrix(dat, tensor=False):  
    """
    Creating O-D Matrix or tensor for whole datset
    Args:
        dat (DataFrame): Bixi dataframe including origin and distnation
        tensor (Boolean): if true it creates tensor matrix        
    """
    if tensor:
        n_origin =  len(np.unique(dat.ki.values)) 
        n_dist = len(np.unique(dat.kj.values))
        o_size, d_size = n_origin, n_dist         

    else:
        n_origin =  len(np.unique(dat.zone_O.values))
        n_dist = None 
        o_size, d_size = n_origin, n_origin
            
    res_dat = resampling(dat, tensor)
    timestamp = np.unique(res_dat.start_date)        
    channel=1   
    OD_tensor_matrix = np.empty((len(timestamp), o_size, d_size, channel))
    for i, ts in enumerate(timestamp):
        df = res_dat[res_dat.start_date==ts]
        mat = time_window_matrix(df, n_origin, n_dist)
        OD_tensor_matrix[i] = mat
    return OD_tensor_matrix

def train_test_split(matrix, train_test_ratio):
    lenght = len(matrix)
    test_inital = int(train_test_ratio*lenght)        

    matrix_train = matrix[:test_inital]    
    matrix_test = matrix[test_inital:]

    return matrix_train, matrix_test

def model_loss(history):
    plt.figure(figsize=(10,5))

    ColNames = {'model loss':['loss', 'val_loss'],
                'model_accuracy':['MSE', 'val_MSE']}
    
    i=1
    for ColName, _ in ColNames.items():   
        plt.subplot(1,2 , i)     
        plt.plot(history.history[ColNames[ColName][0]], label='Train')
        plt.plot(history.history[ColNames[ColName][1]], label='Validation')
        plt.title(ColName)
        plt.ylabel(ColNames[ColName][0])

        plt.legend(loc='upper right')
        plt.grid(linestyle="--")
        plt.tight_layout()
        i+=1
    plt.show();

def evaluate_forecasts(actual, predicted, text = "Test", plot=True):
    """
    Evaluate prediction performance based on RMSE and MAE
    """
    RMSEs = list()
    MAEs = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        RMSEs.append(rmse)

        # calculate mae
        mae = mean_absolute_error(actual[:,i], predicted[:,i])
        # store
        MAEs.append(mae)

    # calculate overall RMSE and MAE
    y_true = actual.flatten()
    y_hat = predicted.flatten()

    overal_mae = mean_absolute_error(y_true, y_hat)
    overal_rmse = np.sqrt(mean_squared_error(y_true, y_hat))

    print("#### Evaluating performance metrics ####")
    print("\n===="+ text+" SET ====")
    print("MAE: {0:.3f}".format(overal_mae))
    print("RMSE: {0:.3f}".format(overal_rmse))
    print("MAEs: ", np.round(MAEs,3))
    print("RMSEs: ", np.round(RMSEs,3))

    if plot:
        plt.plot(np.arange(len(RMSEs)), RMSEs, label=True)
        plt.plot(np.arange(len(MAEs)), MAEs, label=True)
        plt.grid(linestyle="--")
        plt.xlabel("Matrix cells")
        plt.legend(["RMSE", "MAE"])
        plt.title("Performance metrics for "+ text +" dataset")
        plt.show()

    return overal_mae, MAEs, overal_rmse, RMSEs

def split_sequence(sequence, lag):
        '''
        This function splits a given univariate sequence into
        multiple samples where each sample has a specified number
        of time steps and the output is a single time step.
        param new_input: If True it is used for predicting new input
        '''
        try:
                
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + lag
                # check if we are beyond the sequence               
                if end_ix > len(sequence)-1:
                    break
            # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)
    
        except Exception as e:
            raise CustomException(e, sys)
    
def convert_to_supervised(dat, lag):
        '''
        This function takes a 2D sequennce, scales the array and splits
        a given multivariate sequence into multiple samples where each sample has a specified number
        of time steps. It returns multiple time steps as well as the scaler.
        param df (DataFrame): Bike sharing demand for each community over time
        param lag (int): History length or time lag
        '''
        
        try:
            if isinstance(dat, np.ndarray):
                pass
            else:
                dat = dat.values
            
            m, n = dat.shape
            # e.g., if lag = 7, BIXI demand of past 7*15 minutes
            X = np.zeros((m-lag,lag, n))
            Y = np.zeros((m-lag,n))

            for i in range(0,n):
                x, y = split_sequence(dat[:,i], lag)
                X[:,:,i] = x
                Y[:,i] = y
            return X, Y
    
        except Exception as e:
            raise CustomException(e, sys)   