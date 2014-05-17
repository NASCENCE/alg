__author__ = 'Marijn Stollenga'

from scipy import randn, rand, Inf, argmax, zeros_like, log, zeros, ones, floor
from numpy import array, exp, mean
from numpy.random import permutation, normal, standard_cauchy
import copy
    
def MCMC(f, x0, maxEvals=1e5, verbose=False, targetFitness=-1e-10, sigma = .1, batchSize = None):
    dim = len(x0)
    
    bestFound = None
    bestFitness = -Inf

    last_fitness = f(x0)
    T = 1.

    numEvals = 0

    accept_estimate = .5
    avg_fitness = last_fitness
    SMOOTH = .0001
    while numEvals <= maxEvals and bestFitness < targetFitness:
        x_new = x0 + standard_cauchy(dim) * sigma
        fitness = f(x_new)
        if fitness > bestFitness:
            bestFitness = fitness
            bestFound = x_new.copy()
            print bestFitness, bestFound

        print   fitness / T, last_fitness / T
        avg_fitness = avg_fitness * (1.0 - SMOOTH) + fitness * SMOOTH
        accept = min(1.0, exp((fitness - avg_fitness) / T) / exp((last_fitness - avg_fitness) / T))
        # print accept, fitness, last_fitness
        if accept > rand():
            x0 = x_new
            last_fitness = fitness
            accept_estimate = accept_estimate * (1.0 - SMOOTH) + SMOOTH
        else:
            accept_estimate = accept_estimate * (1.0 - SMOOTH)
        if accept_estimate > .4:
            sigma *= 1.1
            # T /= 1.001
        else:
            sigma /= 1.1
            # T *= 1.001
        if numEvals % 1000 == 0:
            print sigma
        numEvals += 1
    return bestFound, bestFitness


if __name__ == "__main__":
    from scipy import dot, array, power
    
    # Rosenbrock function
    def rosen(x):
        return - sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    
    # example run (30-dimensional Rosenbrock)
    print MCMC(rosen, zeros(30), verbose=True)
