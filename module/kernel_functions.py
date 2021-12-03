import numpy as np

def kernel_func_p(x, sigma, p = 1):
    return np.exp(-((x**p)/(p * sigma**p)))

def solve_alpha(K, y):
    return np.matmul(np.linalg.inv(K), y)

def create_y_matrix(y_train, num_digits):
    y_mat = np.zeros((len(y_train), num_digits))
    for i in range(len(y_train)):
        y_mat[i][y_train[i]] = 1
    return y_mat

def classification(predictions, y_test):
    preds = np.argmax(predictions, axis = 1)
    correct = 0
    for i, j in zip(preds, y_test):
        if i == j:
            correct += 1
    return 1 - (correct / len(y_test))
    
def prepare_data(train_X, train_y, test_X, test_y, num_train = 1000, num_test = 500, num_digits = 10):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    digits = range(num_digits)
    
    for i in range(len(train_X)):
        if train_y[i] in digits:
            X_train.append(train_X[i].flatten())
            y_train.append(train_y[i])
            
        if len(y_train) == num_train:
            break
            
    for i in range(len(test_X)):
        if test_y[i] in digits:
            X_test.append(test_X[i].flatten())
            y_test.append(test_y[i])
        
        if len(y_test) == num_test:
            break
    
    return X_train, y_train, X_test, y_test

def add_noise(y_train, p, num_digits):
    new_y = y_train.copy()
    to_change = np.random.choice(len(y_train), int(p * len(y_train)), replace = False)
    for index in to_change:
        new_y[index] = np.random.randint(num_digits)
    return new_y
