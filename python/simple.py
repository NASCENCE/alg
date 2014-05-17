__author__ = 'Marijn Stollenga'

from scipy import randn, rand, Inf, argmax, zeros_like, log, zeros, ones, floor
from numpy import array, exp, mean, set_printoptions
from numpy.random import permutation, normal, standard_cauchy
import copy

set_printoptions(suppress=True)

def Simple(f, x0, maxEvals=1e6, verbose=False, targetFitness=-1e-10, sigma = .001, batchSize = None):
    dim = len(x0)
    if not batchSize:
        batchSize = 4 + int(floor(3 * log(dim)))
    numEvals = 0
    bestFound = None
    bestFitness = -Inf
    mutation = sigma

    population = [x0 + randn(dim) * mutation for _ in range(batchSize)]
    sigmas = ones(batchSize) * sigma

    while numEvals + batchSize <= maxEvals and bestFitness < targetFitness:
        # produce and evaluate samples
        fitnesses = [f(s) for s in population]
        # print fitnesses

        if max(fitnesses) > bestFitness:
            bestFitness = max(fitnesses)
            bestFound = population[argmax(fitnesses)]

        numEvals += batchSize 
        if verbose: print "Step", numEvals/batchSize, ":", max(fitnesses), "best:", bestFitness
        
        # update center and variances
        utilities = computeUtilities(fitnesses)

        new_population = []
        new_sigmas = []
        for n in range(batchSize):
            chosen = select_proportional(utilities)

            new_sigmas.append(sigmas[chosen] * normal(1,.001))
            new_population.append(population[chosen].copy() + standard_cauchy(dim) * new_sigmas[n])

            # new_sigmas.append(sigmas[chosen] + normal(0,.01))
            # new_population.append(population[chosen].copy() + randn(dim) * new_sigmas[n])

        population = new_population
        sigmas = new_sigmas
        print mean(sigmas)
        # population = mix(new_population)

    return bestFound, bestFitness

def mix(population):
    dim = len(population[0])
    new_population = copy.copy(population)

    for d in range(dim):
        if rand() < .0:
            for i, ind in enumerate(permutation(len(population))):
                new_population[ind][d] = population[i][d]

    return new_population

def select_proportional(values):
    selection = rand() * values.sum()
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if total > selection: return i
    return len(values) - 1
    
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
    # utilities -= 1. / L  # baseline
    # print utilities
    return utilities

if __name__ == "__main__":
    from scipy import dot, array, power
    
    # Rosenbrock function
    def rosen(x):
        return - sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    
    # example run (30-dimensional Rosenbrock)
    print Simple(rosen, zeros(30), verbose=True)
