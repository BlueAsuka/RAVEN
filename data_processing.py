"""
This is the file for time series segmentation and instance set construction
After segmentation, the instance will be stored and organized into following structure:

instances
    |_-40kg
    |    |_normal
    |    |_spalling1
    |    |_spalling2
    |    |_...
    |_ 20kg
    |    |_normal
    |    |_spalling1
    |    |_spalling2
    |    |_...
    |_ 40kg
    |    |_normal
    |    |_spalling1
    |    |_spalling2
    |    |_...
"""

import os
import re
import sys
import json
import shutil
import loguru
import random
import pandas as pd
import numpy as np
from colorama import Fore, Style
from typing import List, Tuple
from tqdm.auto import tqdm

sys.path.append('..')
from transform import TimeSeriesTransform

RAW_DATA_DIR = 'data/linear_acuator/raw/'
INSTANCES_DIR = 'data/linear_acuator/instances/'
INFERENCE_DIR = 'data/linear_acuator/inference/'
STATES = ['normal', 
          'backlash1', 'backlash2',
          'lackLubrication1', 'lackLubrication2',
          'spalling1', 'spalling2', 'spalling3', 'spalling4',
          'spalling5', 'spalling6', 'spalling7', 'spalling8']
RE_STATES = ['no_obvious_fault', 'light_spalling', 'medium_spalling', 
             'heavy_spalling', 'backlash', 'lack_lubrication']
LOADS= ['20kg', '40kg', '-40kg']
INFERENCE_RATE = 0.3 # the percentage of inference instances
REPEAT = 5 # repeat 5 times in each test
VERBO = True


with open('../config.configs.json') as f:
    cfg = json.load(f)
time_series_transform = TimeSeriesTransform(cfg)


def init_dirs():
    """Initialize directories"""
    
    # Create INSTANCES_DIR and INFERENCE_DIR
    if not os.path.exists(INSTANCES_DIR):
        os.mkdir(INSTANCES_DIR)
    if not os.path.exists(INFERENCE_DIR):
        os.mkdir(INFERENCE_DIR)
    
    # Create subfolders under instances and inference folder if not existed, else clean all contents for initialization
    for load in LOADS:
        for state in STATES:
            cur_instances_dir = os.path.join(INSTANCES_DIR, f'{load}/{state}')
            cur_inference_dir = os.path.join(INFERENCE_DIR, f'{load}/{state}')
            # If a folder already existed, delete firstly then make a new dir
            if os.path.exists(cur_instances_dir):
                shutil.rmtree(cur_instances_dir)
            os.makedirs(cur_instances_dir) 
            if os.path.exists(cur_inference_dir):
                shutil.rmtree(cur_inference_dir)
            os.makedirs(cur_inference_dir)

            # Show creation processes
            if VERBO:
                loguru.logger.info(f'Directory: {cur_instances_dir} INITIALIZED.')
                loguru.logger.info(f'Directory: {cur_inference_dir} INITIALIZED.')

def construct_state_load_instances(state: str, load: str, dataframe: pd.DataFrame):
    """Construct instances for one state under a specific load"""

    # Extract all groups label for segementation
    group = dataframe.groupby('group')
    group_idx = list(group.groups.keys())

    for i in group_idx:
        # Get set_point, position_error and current from original dataframe
        set_point = group.get_group(i)['set_point']
        position_error = group.get_group(i)['position_error']
        current = group.get_group(i)['current']

        # Group the data into a new dataframe
        tmp_data = {
            'set_point': set_point,
            'position_error': position_error,
            'current': current
        }
        tmp_df = pd.DataFrame(tmp_data)

        # Construct the filename for the saving 
        # load_run_repeat.csv
        load = group.get_group(i)['load'].tolist()[0]
        run = group.get_group(i)['run'].tolist()[0]
        repeat = i % REPEAT if i % REPEAT > 0 else REPEAT
        filename = f"{state}_{load}_{run}_{repeat}.csv"

        # Save the file according to different load
        tmp_df.to_csv(os.path.join(INSTANCES_DIR, f"{load}kg/{state}", filename), index=False)

def construct_instances_set():
    """Construct instances"""

    files = [os.path.join(RAW_DATA_DIR, f"{state}.csv") for state in STATES]
    state_files_dict = {state: file for state, file in zip(STATES, files)}

    for load in LOADS:
        for state in STATES:
            # Load the state file
            state_file = state_files_dict[state]
            try:
                df = pd.read_csv(state_file)
                if VERBO:
                    loguru.logger.info(f"File {state_file} loaded SUCCESSFULLY.")
            except FileNotFoundError:
                loguru.logger.error(f"File {state_file} not found.")
            except Exception as e:
                loguru.logger.exception(f"An error occurred while reading file {state_file}: {e}")

            # Construct instances set for one state under a load
            construct_state_load_instances(state, load, df)
            # Check saving process
            save_path = os.path.join(INSTANCES_DIR, f"{load}/{state}")
            if VERBO:
                loguru.logger.info(f"Constructed {len(os.listdir(save_path))} instance samples for {state} under {load}.")

def construct_inference_set():
    """Move ramdonly selected samples in instances set to inference set to construct inference set"""
    
    random.seed(42) # For reproduciable

    for load in LOADS:
        for state in STATES:
            # Source and destination dir
            source_dir = os.path.join(INSTANCES_DIR, f"{load}/{state}")
            destination_dir = os.path.join(INFERENCE_DIR, f"{load}/{state}")

            # Randomly choose some files in instances to inference
            source_files = os.listdir(source_dir)
            inference_samples = random.sample(source_files, int(len(source_files) * INFERENCE_RATE))
            
            # Move the selected samples from source dir to destination dir
            for file in inference_samples:
                shutil.move(os.path.join(source_dir, file), os.path.join(destination_dir, file))
            if VERBO:
                loguru.logger.info(f"Constructed {len(inference_samples)} inference samples for {state} under {load}.")

def main():
    """Main function"""

    print(Fore.YELLOW + "Initialized..." + Fore.RESET)
    init_dirs()
    print(Fore.YELLOW + "Constructing instances set..." + Fore.RESET)
    construct_instances_set()
    print(Fore.YELLOW + "Constructing inference set..." + Fore.RESET)
    construct_inference_set()
    print(Fore.YELLOW + "Done!" + Fore.RESET)


if __name__ == "__main__":
    main()
