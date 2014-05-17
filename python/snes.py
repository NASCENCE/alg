__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import dot, exp, log, sqrt, floor, ones, randn, zeros_like, Inf, argmax
from numpy import array, mean


def dummy(x):
    pass

def dummy_x(x, center, best=None):
    pass

MAX_SIGMA = 1.

class SNES(object):
    def __init__(self, center, sigma, f, population = None, center_learning_rate = 1.0, sigma_learning_rate = None):
        """ Separable NES, as described in Schaul, Glasmachers and Schmidhuber (GECCO'11).
        Maximizes a function f.
        Returns (best solution found, corresponding fitness) """
        self.dim = len(center)
        self.center = center.copy()
        if sigma == None:
            sigma = 1.0
        if type(sigma) == type(.0):
            self.sigmas = ones(self.dim) * sigma
        else:
            self.sigmas = sigma

        if not population:
            self.population = 4 + int(floor(3 * log(self.dim)))
        else:
            self.population = population
        self.population -= self.population % 2 #make even for symmetry trick

        self.learning_rate = sigma_learning_rate
        if self.learning_rate is None:
            self.learning_rate = 100 * 0.6 * (3 + log(self.dim)) / self.dim / sqrt(self.dim)
        self.center_learning_rate = center_learning_rate
        self.numEvals = 0
        self.bestFound = None
        self.bestFitness = -Inf
        self.f = f

    def start(self, max_evals=1e6, verbose=False, target_fitness=-1e-10, sigma = 1.0, pre_iteration_hook=dummy, post_iteration_hook=dummy_x, best_found_hook=dummy_x):
        n_iterations = 0
        while self.numEvals + self.population <= max_evals and self.bestFitness < target_fitness:
            pre_iteration_hook(self, self.center)
            # produce and evaluate samples
            samples = [randn(self.dim) for _ in range(self.population/2)]
            samples.extend([-s for s in samples])
            realSamples = [self.sigmas * s + self.center for s in samples]
            # realSamples.extend([-self.sigmas * s + self.center for s in samples]) #symmetry trick
            # print realSamples
            # exit(1)
            fitnesses = [self.f(s) for s in realSamples]
            if max(fitnesses) > self.bestFitness:
                self.bestFitness = max(fitnesses)
                self.bestFound = realSamples[argmax(fitnesses)]
                best_found_hook(self.center, self.sigmas, self.bestFound)
            self.numEvals += self.population
            if verbose: print "Step", self.numEvals/self.population, ":", max(fitnesses), "best:", self.bestFitness, "cov:", mean(self.sigmas)

            # update center and variances
            utilities = computeUtilities(fitnesses)
            self.center += self.sigmas * dot(utilities, samples) * self.center_learning_rate
            covGradient = dot(utilities, [s ** 2 - 1 for s in samples])
            self.sigmas = self.sigmas * exp(0.5 * self.learning_rate * covGradient)
            self.sigmas[self.sigmas > MAX_SIGMA] = MAX_SIGMA
            self.sigmas[self.sigmas < -MAX_SIGMA] = -MAX_SIGMA
            post_iteration_hook(self.center, self.sigmas, n_iterations)
            n_iterations += 1

            print "new center:", self.center[:10]
            print "new sigmas:", self.sigmas[:10]
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

    # 100-dimensional ellipsoid function
    dim = 3
    A = array([power(1000, 2 * i / (dim - 1.)) for i in range(dim)])
    def elli(x):
        return -dot(A * x, x)

    # example run
    optim = SNES(ones(dim), .5, elli)
    print optim.start()
