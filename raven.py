import os
import json
import pywt
import loguru
import numpy as np

from data_loader import get_X_y
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


cfg = json.load(open("config/config.json"))

random_state = cfg['RANDOM_STATE']
lower_bound = cfg['LOWER_BOUND']
upper_bound = cfg['UPPER_BOUND'] 
n_kernels = cfg['N_KERNELS']
wavelet = cfg['WAVELET']

np.random.seed(random_state)  # for reproducibility
random_scales = np.random.uniform(lower_bound, upper_bound, n_kernels)

def minmax_scale(X):
    minmax_result = 2 * (X - np.min(X, axis=1, keepdims=True)) / (np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)) - 1
    return minmax_result

def raven(sample, random_scales, wavelet):
    coefficients, _ = pywt.cwt(sample, random_scales, wavelet, method='conv')
    integral = np.array([np.trapz(coeff) for coeff in coefficients])
    return integral

def raven_parallel(X, random_scales, wavelet, max_workers=8):
    raven_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(raven, X[i], random_scales, wavelet) for i in range(len(X))]
        for future in tqdm(futures):
            raven_results.append(future.result())
    return raven_results

if __name__ == "__main__":
    INSTANCES_DIR = 'data/linear_acuator/instances/'
    INFERENCE_DIR = 'data/linear_acuator/inference/'
    STATES = ['normal', 
              'backlash1', 'backlash2',
              'lackLubrication1', 'lackLubrication2',
              'spalling1', 'spalling2', 'spalling3', 'spalling4', 'spalling5', 'spalling6', 'spalling7', 'spalling8']
    LOADS= ['20kg', '40kg', '-40kg']
    
    load = '20kg'
    filenames_20kg = [os.listdir(os.path.join(INSTANCES_DIR, load, state)) for state in STATES]
    filenames_20kg = [filename for sublist in filenames_20kg for filename in sublist]
    filenames_20kg_t = [os.listdir(os.path.join(INFERENCE_DIR, load, state)) for state in STATES]
    filenames_20kg_t = [filename for sublist in filenames_20kg_t for filename in sublist]
    
    loguru.logger.debug(f'Loading instance samples...')
    X_20kg, y_20kg = get_X_y(INSTANCES_DIR, filenames_20kg, load='20kg')
    loguru.logger.debug(f'Loading inference samples...')
    X_20kg_t, y_20kg_t = get_X_y(INFERENCE_DIR, filenames_20kg_t, load='20kg')

    loguru.logger.debug(f'Min-Max scaling instance samples...')
    minmax_X_20kg = minmax_scale(X_20kg)
    loguru.logger.debug(f'Min-Max scaling inference samples...')
    minmax_X_20kg_t = minmax_scale(X_20kg_t)
    
    loguru.logger.debug(f'Transforming instance samples...')
    raven_results_20kg = np.array(raven_parallel(minmax_X_20kg, random_scales, wavelet))
    loguru.logger.debug(f'Transforming inference samples...')
    raven_results_20kg_t = np.array(raven_parallel(minmax_X_20kg_t, random_scales, wavelet))
    
    ridge = RidgeClassifierCV()
    lda = LDA()

    ridge.fit(raven_results_20kg, y_20kg)
    lda.fit(raven_results_20kg, y_20kg)

    print(f'Raven + Ridge: {ridge.score(raven_results_20kg_t, y_20kg_t)}')
    print(f'Raven + LDA: {lda.score(raven_results_20kg_t, y_20kg_t)}')
