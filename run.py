import numpy as np
import logging
from scipy.spatial import distance_matrix
from module.kernel_functions import *
from keras.datasets import mnist
import pickle
import json
import sys
import os

# capture target
try:
    target = sys.argv[1].lower()
except:
    target = None

# clean produced files
if target == 'clean':
    created_files = [
        'test/out/test_results.json',
        'out/results.json',
        'logs/script_progress.log',
        'logs/distance_matrices/D.pickle',
        'logs/distance_matrices/D_test.pickle'
    ]
    
    for file in created_files:
        if os.path.exists(file):
            os.remove(file)
    
# setup logging
logging.basicConfig(filename='logs/script_progress.log', level=logging.DEBUG, format = '%(message)s %(asctime)s', datefmt = '%I:%M:%S %p')

# Read Data
if target == 'test':
    fp = 'test/testdata'
    with open(os.path.join(fp, 'train_X.pickle'), 'rb') as f:
        train_X = pickle.load(f)
    with open(os.path.join(fp, 'train_y.pickle'), 'rb') as f:
        train_y = pickle.load(f)
    with open(os.path.join(fp, 'test_X.pickle'), 'rb') as f:
        test_X = pickle.load(f)
    with open(os.path.join(fp, 'test_y.pickle'), 'rb') as f:
        test_y = pickle.load(f)
else:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

# Read script parameters
with open('config/script-params.json', 'r') as f:
    script_params = json.load(f)

if target == 'test':
    num_train = len(train_y)
    num_test = len(test_y)
    num_classes = 2
elif target != 'clean':
    num_train = script_params['num_train']
    num_test = script_params['num_test']
    num_classes = script_params['num_classes']

# prepare data
if target != 'clean':
    X_train, y_train, X_test, y_test = prepare_data(train_X, train_y, test_X, test_y, num_train, num_test, num_classes)
    results = {}

# only setup distance matrices
if target == 'build':
    fp = 'logs/distance_matrices'
    
    if not os.path.exists(fp + '/D.pickle'):
        D = distance_matrix(X_train, X_train)
        with open(os.path.join(fp, 'D.pickle'), 'wb') as f:
            pickle.dump(D, f)
        
    if not os.path.exists(fp + '/D_test.pickle'):
        D_test = distance_matrix(X_test, X_train)
        with open(os.path.join(fp, 'D_test.pickle'), 'wb') as f:
            pickle.dump(D_test, f)

# setup distance matrices        
elif target != 'clean':
    fp = 'logs/distance_matrices'
    if os.path.exists(fp + '/D.pickle') and target != 'test':
        with open(os.path.join(fp, 'D.pickle'), 'rb') as f:
            D = pickle.load(f)
    else:
        D = distance_matrix(X_train, X_train)
        with open(os.path.join(fp, 'D.pickle'), 'wb') as f:
            pickle.dump(D, f)
        
    if os.path.exists(fp + '/D_test.pickle') and target != 'test':
        with open(os.path.join(fp, 'D_test.pickle'), 'rb') as f:
            D_test = pickle.load(f)
    else:
        D_test = distance_matrix(X_test, X_train)
        with open(os.path.join(fp, 'D_test.pickle'), 'wb') as f:
            pickle.dump(D_test, f)

if target != 'build' and target != 'clean':
    # iterate through specified kernel types
    for p in script_params['p_kernels']:
        results[p] = {}

        # Iterate through specified c modifiers
        for c in script_params['c_modifiers']:
            results[p][c] = []

            # setup kernel matrices
            K = kernel_func_p(D, c * np.mean(D), p)
            K_inv = np.linalg.inv(K)
            K_test = kernel_func_p(D_test, c * np.mean(D), p)

            # Iterate through noisiness levels
            for noise in np.arange(script_params['min_noise'], script_params['max_noise'] + script_params['noise_step'], script_params['noise_step']):
                # randomly add noise to y
                noisy_y = add_noise(y_train, round(noise, 5), num_classes)

                # create noisy label matrix
                y_matrix = create_y_matrix(noisy_y, num_classes)

                # solve for alpha vector
                alpha = np.matmul(K_inv, y_matrix)

                # Create predictions and find classification error
                predictions = np.matmul(K_test, alpha)
                results[p][c].append(round(classification(predictions, y_test), 5))

                # log results
            logging.debug(f'Finished p = {p} and c = {c} at')

    # Write results
    if target == 'test':
        with open('test/out/test_results.json', 'w') as f:
            json.dump(results, f)
    else:
        with open('out/results.json', 'w') as f:
            json.dump(results, f)
