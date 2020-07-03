import numpy as np
import h5py
import sys
from time import perf_counter

# how many neurons in each layer (layer 0 is the input and the last layer is the output)
LAYERS = [784,800,10,10]
# numpy array of classes (needs to be the same number of classes as there are nodes in the final layer)
CLASSES = np.array([[0,1,2,3,4,5,6,7,8,9]]).T
# default activation type (relu, sigmoid or tanh)
ACTIVATION = 'tanh'

def save_params(params, filename = 'data.h5') :
    # save weights and biases to file
    try :
        hf = h5py.File(filename, 'w')
        for key,value in params.items() :
            hf.create_dataset(key, data=value)
        hf.close()
    except :
        print('Error saving params')

def load_params(layers, filename = 'data.h5') :
    # load weights and biases from file
    params = dict()
    try :
        hf = h5py.File(filename, 'r')
        for key in hf.keys() :
            params[key] = np.array(hf.get(key))
        hf.close()
    except :
        params = init(layers)
    return params

def one_hot(Y) :
    hotY = (Y == CLASSES)
    hotY = hotY.astype(int)
    return hotY

def mnist_X_Y(filename = 'mnist_train.csv') :
    # load training or test data from csv files (mnist)
    try :
        data = np.loadtxt(filename, delimiter=",")
    except :
        return (None, None)
    X = np.array(data[:,1:], dtype='uint8') / 255
    X = X.T
    Y = np.array(data[:,0], dtype='uint8')
    Y = one_hot(Y)
    return (X, Y)

def random_mini_batches(X, Y, mini_batch_size = 128) :
    m = X.shape[1]
    mini_batches = []
    perms = list(np.random.permutation(m))
    shuffled_X = X[:,perms]
    shuffled_Y = Y[:,perms]
    complete = int(np.floor(m/mini_batch_size))
    for i in range(0, complete):
        mini_batch_X = shuffled_X[:, i * mini_batch_size : i * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, i * mini_batch_size : i * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, complete * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, complete * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def relu(Z) :
    # relu activation function
    return Z * (Z > 0)

def drelu(Z) :
    # relu derivative
    return 1 * (Z > 0)

def sigmoid(Z) :
    # sigmoid activation function
    return 1/(1 + np.exp(-Z))

def dsigmoid(Z) :
    # sigmoid derivative
    return sigmoid(Z)*(1-sigmoid(Z))

def tanh(Z) :
    # tanh activation function
    return np.tanh(Z)

def dtanh(Z) :
    # tanh derivative
    return 1/(np.cosh(Z)**2)

def activation(Z, type = ACTIVATION) :
    if type == 'relu' :
        return relu(Z)
    elif type == 'sigmoid' :
        return sigmoid(Z)
    elif type == 'tanh' :
        return tanh(Z)

def dactivation(Z, type = ACTIVATION) :
    if type == 'relu' :
        return drelu(Z)
    elif type == 'sigmoid' :
        return dsigmoid(Z)
    elif type == 'tanh' :
        return dtanh(Z)

def softmax(Z) :
    # softmax activation function
    ez = np.exp(Z - np.max(Z))
    return ez/np.sum(ez, axis=0)

def init(layers) :
    # He initialisation
    params = dict()
    for l in range(1, len(layers)) :
        params['W'+str(l)] = (np.random.randn(layers[l], layers[l-1]))*np.sqrt(2/(layers[l-1] + layers[l]))
        params['b'+str(l)] = (np.random.randn(layers[l], 1))
    return params

def fp(X, params, layers) :
    # forward propagation
    cache = dict()
    H = 0
    cache['A0'] = X
    for l in range(1, len(layers)) :
        cache['Z'+str(l)] = (params['W'+str(l)] @ cache['A'+str(l-1)]) + params['b'+str(l)]
        if l < len(layers) - 1 :
            cache['A'+str(l)] = activation(cache['Z'+str(l)])
        else :
            cache['A'+str(l)] = softmax(cache['Z'+str(l)])
    H = cache.get('A'+str(len(layers)-1), None)
    return (H, cache)

def cost(X, Y, params, layers, lambd = 0, e = 1e-10) :
    # cost function
    (H, cache) = fp(X, params, layers)
    m = X.shape[1]
    L2reg = 0
    if lambd > 0 :
        for l in range(1, len(layers)) :
            L2reg += np.sum(np.square(params['W'+str(l)]))
        L2reg = lambd * L2reg / (2 * m)
    J = np.sum(np.sum(-Y*np.log(H+e)))/m + L2reg
    return (J, cache)

def bp(X, Y, params, layers, cache, lambd = 0, grad_check = False) :
    # simple back propagation with L2 regularisation, no batch norm yet
    L = len(layers)-1
    m = X.shape[1]
    grads = dict()
    cache['dZ'+str(L)] = cache['A'+str(L)] - Y
    grads['dW'+str(L)] = cache['dZ'+str(L)] @ np.transpose(cache['A'+str(L-1)])/m + (lambd * params['W'+str(L)])/m
    grads['db'+str(L)] = np.mean(cache['dZ'+str(L)], axis=1, keepdims = True)
    cache['dA'+str(L-1)] = np.transpose(params['W'+str(L)]) @ cache['dZ'+str(L)]
    for l in range(len(layers)-2, 0, -1) :
        cache['dZ'+str(l)] = dactivation(cache['Z'+str(l)])*cache['dA'+str(l)]
        grads['dW'+str(l)] = cache['dZ'+str(l)] @ np.transpose(cache['A'+str(l-1)])/m + (lambd * params['W'+str(l)])/m
        grads['db'+str(l)] = np.mean(cache['dZ'+str(l)], axis=1, keepdims = True)
        if l > 1 :
            cache['dA'+str(l-1)] = np.transpose(params['W'+str(l)]) @ cache['dZ'+str(l)]
    if grad_check :
        numeric_grads = num_grads(X, Y, params, layers)
        if not compare_grads(grads, numeric_grads) :
            print("GRAD CHECK FAILED!")
    return grads

def num_grads(X, Y, params, layers, e = 1e-8) :
    # calculate gradients numerically to check back propagation function
    grads = dict()
    perturbed_params_1 = params.copy()
    perturbed_params_2 = params.copy()
    for key,value in params.items() :
        grads['d'+key] = np.zeros(params[key].shape)
        perturbed_params_1[key] = np.copy(params[key])
        perturbed_params_2[key] = np.copy(params[key])
        for i in range(len(value[:,0])) :
            for j in range(len(value[0,:])) :
                orig1 = perturbed_params_1[key][i,j]
                orig2 = perturbed_params_2[key][i,j]
                perturbed_params_1[key][i,j] = perturbed_params_1[key][i,j] + e
                perturbed_params_2[key][i,j] = perturbed_params_2[key][i,j] - e
                (loss_1,_) = cost(X, Y, perturbed_params_1, layers)
                (loss_2,_) = cost(X, Y, perturbed_params_2, layers)
                grads['d'+key][i,j] = (loss_1 - loss_2) / (2*e)
                perturbed_params_1[key][i,j] = orig1
                perturbed_params_2[key][i,j] = orig2
    return grads

def compare_grads(grads1, grads2, tolerance = 1e-4) :
    # compare two sets of gradients (i.e. from numerical calculation and back propagation)
    ok = True
    for key,value in grads1.items() :
        for i in range(len(value[:,0])) :
            for j in range(len(value[0,:])) :
                if np.abs(grads1[key][i,j] - grads2[key][i,j]) > tolerance :
                    print(key,i,j)
                    print(grads1[key][i,j], grads2[key][i,j])
                    print('diff:',grads1[key][i,j]-grads2[key][i,j])
                    ok = False
    return ok

def duplicate_params(params) :
    # copy all weights and biases for the numerical gradient calculations
    copy = dict()
    for key,value in params.items() :
        copy[key] = value.copy() # numpy array
    return copy

def momentum_zeros(layers = LAYERS) :
    V = dict()
    for l in range(1, len(layers)) :
        V['dW'+str(l)] = np.zeros((layers[l], layers[l-1]))
        V['db'+str(l)] = np.zeros((layers[l], 1))
    return V

def gradient_decent(X, Y, params, layers, alpha = 0.1, lambd = 0, beta = 0, epochs = 10, mini_batch_size = 64, grad_check = False, save_parameters = False, print_J = False, skip_bad_batch = False) :
    # basic bitch gradent decent, no adam yet
    V = momentum_zeros()
    for i in range(epochs) :
        mini_batches = random_mini_batches(X, Y, mini_batch_size = mini_batch_size)
        for mini_batch in mini_batches :
            (mini_batch_X, mini_batch_Y) = mini_batch
            (J, cache) = cost(mini_batch_X, mini_batch_Y, params, layers, lambd = lambd)
            if skip_bad_batch :
                J_prev = J
                params_prev = duplicate_params(params) # worth the performance hit? probably not
            grads = bp(mini_batch_X, mini_batch_Y, params, layers, cache, lambd = lambd, grad_check = grad_check)
            for j in range(1, len(layers)) :
                # momentum
                V['dW'+str(j)] = beta * V['dW'+str(j)] + (1 - beta) * grads['dW'+str(j)]
                V['db'+str(j)] = beta * V['db'+str(j)] + (1 - beta) * grads['db'+str(j)]
                avdW = alpha * V['dW'+str(j)]
                avdb = alpha * V['db'+str(j)]
                if not np.isnan(avdW).any() and not np.isnan(avdb).any():
                    params['W'+str(j)] = params['W'+str(j)] - avdW
                    params['b'+str(j)] = params['b'+str(j)] - avdb
                else :
                    if skip_bad_batch :
                        params = params_prev
                        break
            if skip_bad_batch :
                (J, cache) = cost(mini_batch_X, mini_batch_Y, params, layers, lambd = lambd)
                if J > J_prev :
                    print('COST GOING UP, skipping mini-batch')
                    params = params_prev
                    continue
                if np.isnan(J) :
                    print('Cost is NaN, skipping mini-batch')
                    continue
            J_prev = J
            if print_J :
                print('Epoch: ', i+1, '/', epochs, ' Cost = ', J, sep='', end='\r')
            if save_parameters :
                save_params(params)
    if print_J :
        print('\n', end='')
    return (params, J)

def train(filename = 'mnist_train.csv') :
    print('Loading Data...')
    params = load_params(LAYERS)
    (X, Y) = mnist_X_Y(filename)
    if X is None or Y is None :
        print('Error loading training data')
        quit()
    print('Training...')
    start = perf_counter()
    (params, J) = gradient_decent(X, Y, params, LAYERS, alpha = 0.02, lambd = 0.025, beta = 0.95, epochs = 50, mini_batch_size = 512, grad_check = False, save_parameters = False, print_J = True, skip_bad_batch = False)
    save_params(params)
    end = perf_counter()
    print('Training completed in', end - start, 'seconds')
    print('Testing...')
    test(title = 'Train Data', filename = 'mnist_train.csv')

def test(title = 'Test Data', filename = 'mnist_test.csv') :
    params = load_params(LAYERS)
    (X, Y) = mnist_X_Y(filename)
    if X is None or Y is None :
        print('Error loading test data')
        quit()
    m = X.shape[1]
    (H, cache) = fp(X, params, LAYERS)
    H = np.argmax(H, axis=0)
    Y = np.argmax(Y, axis=0)
    comp = (H == Y).astype(int)
    accuracy = np.round(np.mean(comp)*100, decimals = 3)
    print(title,': ',str(accuracy)+'% accuracy', sep='')
    return accuracy

if 'loop' in sys.argv :
    i = 0
    while True :
        i += 1
        print('Itteration:',i)
        train()
        test()
else :
    if 'train' in sys.argv :
        train()
    if 'test' in sys.argv :
        test()
    if 'train' not in sys.argv and 'test' not in sys.argv :
        print("Use 'train' and/or 'test' or 'loop' argument")
