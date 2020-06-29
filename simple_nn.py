import numpy as np
import h5py
import sys
from time import perf_counter

# how many neurons in each layer (layer 0 is the input and the last layer is the output)
LAYERS = [2,15,15,15,1]

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
        print('Error loading params')
        params = init(layers)
    return params

def classification_to_learn(X) :
    # in the absence of data, lets make some up
    m = len(X[0,:])
    Y = np.zeros((LAYERS[len(LAYERS)-1],m))
    for i in range(m) :
        # whatever classification fucntion you like so long as Y is either 1 or 0
        if 2*np.sin(X[0,i]) > np.cos(X[1,i]) :
            Y[0,i] = 1
        else :
            Y[0,i] = 0
    return Y

def generate_X_Y(m, layers) :
    # random inputs from -5 to 5
    X = np.random.uniform(-5, 5, [layers[0],m])
    Y = classification_to_learn(X)
    return (X, Y)

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
            cache['A'+str(l)] = relu(cache['Z'+str(l)])
        else :
            cache['A'+str(l)] = sigmoid(cache['Z'+str(l)])
        H = cache.get('A'+str(len(layers)-1), None)
    return (H, cache)

def cost(X, Y, params, layers, e = 1e-10) :
    # cost function
    (H, cache) = fp(X, params, layers)
    m = len(Y[0,:])
    J = np.sum(np.sum(-Y*np.log(H+e)-(1-Y)*np.log(1-H+e)))/m
    return (J, cache)

def bp(X, Y, params, layers, cache, grad_check = False) :
    # simple back propagation, no L2 regularisation or batch norm yet
    L = len(layers)-1
    m = len(Y[0,:])
    grads = dict()
    cache['dA'+str(L)] = - Y / cache['A'+str(L)] + (1 - Y) / (1 - cache['A'+str(L)])
    cache['dZ'+str(L)] = dsigmoid(cache['Z'+str(L)])*cache['dA'+str(L)]
    grads['dW'+str(L)] = cache['dZ'+str(L)] @ np.transpose(cache['A'+str(L-1)]) / m
    grads['db'+str(L)] = np.mean(cache['dZ'+str(L)], axis=1, keepdims = True)
    cache['dA'+str(L-1)] = np.transpose(params['W'+str(L)]) @ cache['dZ'+str(L)]
    for l in range(len(layers)-2, 0, -1) :
        cache['dZ'+str(l)] = drelu(cache['Z'+str(l)])*cache['dA'+str(l)]
        grads['dW'+str(l)] = cache['dZ'+str(l)] @ np.transpose(cache['A'+str(l-1)]) / m
        grads['db'+str(l)] = np.mean(cache['dZ'+str(l)], axis=1, keepdims = True)
        if l > 1 :
            cache['dA'+str(l-1)] = np.transpose(params['W'+str(l)]) @ cache['dZ'+str(l)]
    if grad_check :
        numeric_grads = num_grads(X, Y, params, layers)
        if not compare_grads(grads, numeric_grads) :
            print("GRAD CHECK FAILED!")
    return grads

def num_grads(X, Y, params, layers, e = 1e-4) :
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

def compare_grads(grads1, grads2, tolerance = 1e-3) :
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

def gradient_decent(X, Y, params, layers, alpha = 0.001, itterations = 100, grad_check = False) :
    # basic bitch gradent decent, no momentum or adam yet
    (J, cache) = cost(X, Y, params, layers)
    J_prev = J
    m = len(Y[:,0])
    for i in range(itterations) :
        grads = bp(X, Y, params, layers, cache, grad_check = grad_check)
        params_prev = duplicate_params(params) # worth the performance hit? probably not
        for j in range(1, len(layers)) :
            adW = alpha * grads['dW'+str(j)]
            adb = alpha * grads['db'+str(j)]
            if not np.isnan(adW).any() and not np.isnan(adb).any():
                params['W'+str(j)] = params['W'+str(j)] - adW
                params['b'+str(j)] = params['b'+str(j)] - adb
            else :
                return (params, J)
        (J, cache) = cost(X, Y, params, layers)
        if J_prev is not None and J == J_prev :
            print('Cost stopped reducing, breaking')
            break
        if J_prev is not None and J > J_prev :
            print('COST GOING UP, breaking')
            params = params_prev
            break
        if np.isnan(J) :
            print('Cost is NaN, breaking')
            break
        else :
            J_prev = J
    return (params, J)

training = False
testing = False
if 'train' in sys.argv :
    training = True
if 'test' in sys.argv :
    testing = True

if training :
    J = 1
    for i in range(10000) : # 10000 mini-batches
        #start = perf_counter()
        m = 1024
        (X, Y) = generate_X_Y(m, LAYERS)
        params = load_params(LAYERS)
        (params, J) = gradient_decent(X, Y, params, LAYERS, alpha = 0.003*min(J,1), itterations = 100, grad_check = False)
        print('cost =', J)
        save_params(params)
        #end = perf_counter()
        #print(end - start, 'seconds')
elif testing :
    params = load_params(LAYERS)
    m = 10000
    (X, Y) = generate_X_Y(m, LAYERS)
    (H, cache) = fp(X, params, LAYERS)
    H = np.round(H)
    comp = (H == Y)
    print(str(np.round(np.sum(comp)/len(comp[0,:])*100, decimals = 3))+'% accuracy')
else :
    print("Use 'train' or 'test' argument")
