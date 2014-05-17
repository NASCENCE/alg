from compressed import *
import xnes
import snes
import simple
import cosyne

import numpy as np
import cPickle as pickle
import gzip

np.set_printoptions(suppress=True)

# Loading MNIST data
f = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()

(inputs, classes) = train_set
inputs = inputs.T
targets = np.zeros((10, classes.size))

# inputs = np.random.randn(10,100)
# real_b = np.random.randn(5,10)
# targets = np.dot(real_b, inputs)

# for i, n in enumerate(classes):
# 	targets[n, i] = 1

# Compressor
print "setting up compressor"
weights = [np.zeros((10,784)).astype(np.float32), np.zeros((784,1)).astype(np.float32), 
			np.zeros((10,10)).astype(np.float32), np.zeros((10,1)).astype(np.float32)]
# weights = [np.zeros((5,10)).astype(np.float32), np.zeros((10,1)).astype(np.float32)]

compressor = Compressor(weights, max_index = 100)

def forward(nn_params, inputs):
    nn_weights = nn_params[0::2]
    nn_biasses = nn_params[1::2]
    outputs = inputs
    for i, w in enumerate(nn_weights):
        outputs = np.dot(w, outputs + nn_biasses[i])
        np.tanh(outputs, outputs)
        if i + 1 == len(nn_weights): # last
    	    outputs = outputs * .5 + .5
    	    outputs /= np.sum(outputs, axis=0)

    return outputs

#Define fitness function
def f(values):
    nn_params = compressor.decode(values) #select first (and only) matrix

    # print values
    outputs = forward(nn_params, inputs)
    # print outputs[:,:8]
    # print targets[:,:8]
    fitness = -np.sum(np.abs(outputs - targets)) / 2
    # print outputs
    print fitness
    if np.isnan(fitness):
    	return -99999999
    return fitness

start_x = np.zeros(50)

print "Starting search"
# bestx, bestfitness = snes.SNES(f, start_x, verbose=False, sigma = .1, maxEvals = 25000)
# bestx, bestfitness = xnes.xNES(f, start_x, verbose=True, sigma = .01, maxEvals = 250000, batchSize = 100)
# bestx, bestfitness = simple.Simple(f, start_x, verbose=True, sigma = .01, maxEvals = 250000, batchSize = 100)
bestx, bestfitness = cosyne.Cosyne(f, start_x, verbose=True, sigma = .5, maxEvals = 250000, batchSize = 30)

b = compressor.decode(bestx)[0]

print "bestx and fitness"
print bestfitness, bestx

best_nn_params = compressor.decode(bestx) #select first (and only) matrix
guess = forward(best_nn_params, inputs)

print "b"
print b

print "real b"
print real_b

print "targets"
print targets


print "guess"
print guess
print "error"
print guess - targets
print np.linalg.norm(guess - targets)
print np.linalg.norm(np.dot(real_b, inputs) - targets)
