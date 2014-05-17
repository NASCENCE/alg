__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import dot, exp, log, sqrt, floor, ones, randn, zeros_like, Inf, argmax, eye, outer
from scipy.linalg import expm2
from numpy import array
import numpy as np
import cPickle as pickle

def dummy(center):
    pass

def dummy_x(x, center, best=None):
    pass


class XNES(object):
    def __init__(self, center, sigma, f, population = None, center_learning_rate = 1.0):
        self.center = center
        self.dim = len(self.center)
        if sigma == None:
            sigma = 1.0
        if type(sigma) == type(.0):
            self.A = eye(self.dim) * sigma
        else:
            self.A = sigma

        print "center:", self.center
        print "A:", self.A

        if not population:
            self.population = 4 + int(floor(3 * log(self.dim)))
        else:
            self.population = population

        self.population -= self.population % 2 #make even for symmetry trick
        self.learningRate = 0.1 * (3 + log(self.dim)) / self.dim / sqrt(self.dim)
        self.center_learning_rate = center_learning_rate
        self.numEvals = 0
        self.bestFound = None
        self.bestFitness = -Inf
        self.f = f

    def start(self, maxEvals=1e6, verbose=False, target_fitness= -1e-10, sigma = 1.0, pre_iteration_hook=dummy, post_iteration_hook=dummy_x, best_found_hook=dummy_x):
        """ Exponential NES (xNES), as described in
        Glasmachers, Schaul, Sun, Wierstra and Schmidhuber (GECCO'10).
        Maximizes a function f.
        Returns (best solution found, corresponding fitness).
        """

        I = eye(self.dim)
        n_iterations = 0
        while self.numEvals + self.population <= maxEvals and self.bestFitness < target_fitness:
            pre_iteration_hook(self, self.center)
            # produce and evaluate samples
            print "n evals:", self.numEvals, self.bestFitness
            if verbose: print "Current mean:", self.center, "max min cov:", np.max(self.A), np.min(self.A)
            samples = [randn(self.dim) for _ in range(self.population / 2)]
            samples.extend([-s for s in samples])

            realSamples = [dot(self.A, s) + self.center for s in samples]
            # realSamples.extend([-dot(self.A, s) + self.center for s in samples])
            fitnesses = [self.f(s) for s in realSamples]
            if max(fitnesses) > self.bestFitness:
                self.bestFitness = max(fitnesses)
                self.bestFound = realSamples[argmax(fitnesses)].copy()
                best_found_hook(self.center, self.A, self.bestFound)

            self.numEvals += self.population
            if verbose: print "Step", self.numEvals / self.population, ":", max(fitnesses), "best:", self.bestFitness, realSamples[argmax(fitnesses)]
            #print A
            # update self.center and variances
            utilities = computeUtilities(fitnesses)
            self.center += dot(self.A, dot(utilities, samples)) * self.center_learning_rate
            covGradient = sum([u * (outer(s, s) - I) for (s, u) in zip(samples, utilities)])
            self.A = dot(self.A, expm2(0.5 * self.learningRate * covGradient))
            post_iteration_hook(self.center, self.A, n_iterations)
            n_iterations += 1

        print "Step", self.numEvals / self.population, "best:", self.bestFitness
        return self.bestFound, self.bestFitness


def computeUtilities(fitnesses):
    L = len(fitnesses)
    ranks = zeros_like(fitnesses)
    l = zip(fitnesses, range(L))
    l.sort()
    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = array([max(0., x) for x in log(L / 2. + 1.0) - log(L - array(ranks))])
    utilities /= sum(utilities)       # make the utilities sum to 1
    utilities -= 1. / L  # baseline
    return utilities


if __name__ == "__main__":
    from scipy import dot, array, power

    # Rosenbrock function
    def rosen(x):
        return - sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    # example run (30-dimensional Rosenbrock)
    optim = XNES(ones(30), sigma=1, f=rosen)
    print optim.start(verbose=False)
