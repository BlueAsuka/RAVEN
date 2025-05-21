import re
import os
import ast
import sys
import time
import json
import loguru
import torch
import argparse

import tsai.learner
sys.path.append('../')

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from transform import *
from tsai.all import *

from torch.utils.data import DataLoader, TensorDataset

cfg = json.load(open("config/config.json"))
ts_transform = TimeSeriesTransform(cfg)

INSTANCES_DIR = 'data/linear_acuator/instances'
INFERENCE_DIR = 'data/linear_acuator/inference'
STATES = ['normal', 
          'backlash1', 'backlash2',
          'lackLubrication1', 'lackLubrication2',
          'spalling1', 'spalling2', 'spalling3', 'spalling4', 'spalling5', 'spalling6', 'spalling7', 'spalling8']
MODELS = ['LSTM', 'LSTM_FCN', 'InceptionTime', 'TST', 'mWDN']
OUTPUT_DIR = 'outputs/'


def construct_dataset(filenames: List[str], load:str, train: bool=True):
    X, y = [], []
    data_dir = INSTANCES_DIR if train else INFERENCE_DIR
    for filename in filenames:
        load_num = load[:2]
        state = re.match(fr'(.*)_{load_num}', filename).group(1)
        df = pd.read_csv(os.path.join(data_dir, load, state, filename))
        tmp_cur = ts_transform.smoothing(ts_df=df, field='current')
        # tmp_pos = ts_transform.smoothing(ts_df=df, field='position_error')
        X.append(tmp_cur)
        y.append(state)
    return np.array(X), np.array(y)


def train_model(dls, model_type: str):
    learner = ts_learner(dls, model_type, metrics=accuracy, verbose=False)
    learner.fit_one_cycle(cfg["EPOCHS"])
    return learner
    

def test_model(X_t, y_t, learn):
    _, _, preds = learn.get_X_preds(X_t, y_t, with_decoded=True)
    # Convert the string to a list
    preditions = ast.literal_eval(preds)
    return sum(preditions == y_t) / len(y_t)*100


def get_model_parameters_size(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Control the load, model type, number of runs, and more.")
    
    parser.add_argument('load', type=str, help="The load to be used (e.g., 20kg).")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = json.load(open("config/config.json"))

    # Set the load and other params from the command-line arguments
    load = args.load
    verbose = args.verbose

    # Verbose logging setup
    if verbose:
        loguru.logger.enable("__main__")
    else:
        loguru.logger.disable("__main__")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filenames = [os.listdir(os.path.join(INSTANCES_DIR, load, state)) for state in STATES]
    filenames = [filename for sublist in filenames for filename in sublist]
    loguru.logger.info(f'Read {len(filenames)} files for learning')

    test_filenames = [os.listdir(os.path.join(INFERENCE_DIR, load, state)) for state in STATES]
    test_filenames = [filename for sublist in test_filenames for filename in sublist]
    loguru.logger.info(f'Read {len(test_filenames)} files for inference')
    
    X, y = construct_dataset(filenames, load, train=True)
    X_t, y_t = construct_dataset(test_filenames, load, train=False)
    loguru.logger.info(f'Construct dataset for training and inference')
    
    X, y, splits = combine_split_data([X], [y]) 
    X_t, y_t, splits_t = combine_split_data([X_t], [y_t])
    loguru.logger.info(f'Split dataset')

    tfms = [None, [Categorize()]]
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, bs=[64, 64], device=device)
    
    accuracy_dict = {}
    for model in MODELS:
        loguru.logger.info(f'Running {model}')
        
        accuracy_dict[model] = {"accuracy": [], 
                                "mean": None, 
                                "std": None, 
                                "train_average_time": None,
                                "test_average_time": None,
                                "trainable_params": None,
                                "total_params": None,
                                "model_size(Mb)": None}
        
        
        trained_models = []
        train_times, test_time = [], []
        for _ in tqdm(range(cfg["NUM_RUNS"])):
            start_time = time.perf_counter()
            learner = train_model(dls, model)
            end_time = time.perf_counter()
            train_times.append(end_time - start_time)
            
            start_time = time.perf_counter()
            accuracy =  test_model(X_t, y_t, learner)
            end_time = time.perf_counter()
            accuracy_dict[model]["accuracy"].append(accuracy)
            test_time.append(end_time - start_time)
            
        t_p, total_p = get_model_parameters_size(learner.model)    
        model_size = get_model_size(learner.model)
            
        accuracy_dict[model]["mean"] = np.mean(accuracy_dict[model]["accuracy"])
        accuracy_dict[model]["std"] = np.std(accuracy_dict[model]["accuracy"])
        accuracy_dict[model]["train_average_time"] = np.mean(train_times)
        accuracy_dict[model]["test_average_time"] = np.mean(test_time)
        assert t_p is not None and total_p is not None
        accuracy_dict[model]["trainable_params"] = t_p
        accuracy_dict[model]["total_params"] = total_p
        accuracy_dict[model]["model_size(Mb)"] = model_size

    filename = f'{load}_baseline.json'
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        json.dump(accuracy_dict, f, indent=4)
    loguru.logger.info(f'Saved to {filename}')
