__author__ = 'Marijn Stollenga'

from scipy import randn, rand, Inf, argmax, zeros_like, ones_like, log, zeros, ones, floor
from numpy import array, exp, mean, uint32, set_printoptions
from numpy.random import permutation, normal, standard_cauchy, uniform
import numpy as np
import copy
import random
import utils

set_printoptions(suppress=True)

def dummy(center):
    pass

def dummy_x(x, center, best=None):
    pass


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
    # print "sort:", l

    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = array([max(0., x) for x in log(L*2./3. + 1.0) - log(L - array(ranks))])
    # utilities = array([max(0., x) for x in array(ranks)])
    utilities /= sum(utilities)       # make the utilities sum to 1
    # utilities -= 1. / L  # baseline
    # print utilities
    return utilities


class CosyneM:
    def __init__(self, center, sigma, f, population_size = None, center_learning_rate = 1.0, crossover_proportion = .1, load_file = None):
        self.center = center
        self.dim = len(center)
        self.f = f

        self.population_size = population_size
        if not self.population_size:
            self.population_size = 4 + int(floor(3 * log(self.dim)))
        self.numEvals = 0
        self.best_found = center
        self.best_fitness = -Inf

        self.mutation = sigma

        self.population = [center + randn(self.dim) * self.mutation for _ in range(self.population_size)]
        self.momentums = [randn(self.dim) * self.mutation for _ in range(self.population_size)]

        self.sigmas = ones(self.population_size) * sigma
        self.crossover_proportion = crossover_proportion

        if load_file is not None:
            print "loading cosyne file"
            self.load(load_file)


    def start(self, max_evals=1e6, verbose=False, target_fitness=-1e-10, pre_iteration_hook=dummy, post_iteration_hook=dummy_x, best_found_hook=dummy_x):
        while self.numEvals + self.population_size <= max_evals and self.best_fitness < target_fitness:
            pre_iteration_hook(self.center)
            # produce and evaluate samples
            fitnesses = array([self.f(s) for s in self.population])

            # find best

            self.center = self.population[argmax(fitnesses)]
            #print "best now:", self.center
            #print "len:", len(self.center)
            self.save("cosynem_state.gzip")

            if max(fitnesses) > self.best_fitness:
                self.best_fitness = max(fitnesses)
                self.best_found = self.population[argmax(fitnesses)].copy()
                print "best:", self.best_found
                best_found_hook(self.center, self.sigmas, self.best_found)


            #Recombine
            new_population = []
            new_sigmas = []
            new_momentums = []
            
            utilities = computeUtilities(fitnesses)

            #print "fitness & utilities:"
            #print fitnesses
            #print utilities

            #create new population
            crossover_percentage = .1
            noise_chance = .3
            for n in range(self.population_size):
                if rand() > crossover_percentage:
                    chosen = select_proportional(utilities)
                    #new_population.append(self.population[chosen] + randn(self.dim) * self.mutation)
                    m = self.momentums[chosen] + randn(self.dim) * self.sigmas[chosen] * (uniform(0,1,self.dim) < noise_chance)
                    if uniform(0,1) < .05:
                        m = randn(self.dim) * .001 #reset
                        
                    new_population.append(self.population[chosen] + m)
                    new_sigmas.append(self.sigmas[chosen] * (normal(1,.1)**2))
                    new_momentums.append(m)
                else:
                    a, b = select_proportional(utilities), select_proportional(utilities)

                    while a == b:
                        b = select_proportional(utilities)

                    cross_point = random.randint(0, self.dim)

                    ind = np.append(self.population[a][:cross_point], self.population[b][cross_point:])
                    indsig = self.sigmas[a]
                    new_population.append(ind)
                    new_sigmas.append(indsig * (normal(1,.1) ** 2))
                    new_momentums.append(np.concatenate([self.momentums[a][:cross_point], self.momentums[b][cross_point:]]))

            self.population = new_population
            self.sigmas = new_sigmas
            self.momentums = new_momentums
            #print self.momentums
            #permute
            max_fitness, min_fitness = max(fitnesses), min(fitnesses)
            if (max_fitness - min_fitness) > 1e-6:
                # chances = ((fitnesses - min_fitness) / (max_fitness - min_fitness)) ** (1.0 / len(self.population))
                chances = ones_like(fitnesses) * .9

                for d in range(self.dim):
                    from_list = []

                    for ind in range(len(self.population)):
                        if rand() > chances[ind]:
                            from_list.append(ind)
                    to_list = permutation(from_list).astype(uint32)
                    
                    values = zeros(len(from_list))
                    momentums = zeros(len(from_list))
                    
                    for i, v in enumerate(from_list):
                        values[i] = self.population[v][d]
                        momentums[i] = self.momentums[v][d]

                    # print to_list, from_list

                    for i, v in enumerate(to_list):
                        self.population[v][d] = values[i]
                        self.momentums[v][d] = momentums[i]

            #elite
            self.population[0] = self.center

            self.numEvals += self.population_size
            post_iteration_hook(self.center, self.sigmas, self.best_found)

        return self.best_found, self.best_fitness

    def save(self, filename):
        utils.save_pickle_gzip([self.numEvals, self.best_found, self.center, self.best_fitness, self.mutation, self.population, self.momentums, self.sigmas, self.crossover_proportion], filename)

    def load(self, filename):
        print filename
        a = utils.load_pickle_gzip(filename)
        print len(a)
        
        self.numEvals, self.best_found, self.center, self.best_fitness, self.mutation, self.population, self.momentums, self.sigmas, self.crossover_proportion = a
        #  [self.numEvals, self.best_found, self.best_fitness, self.mutation, self.population, self.sigmas, self.crossover_proportion] = utils.load_pickle_gzip(filename)


if __name__ == "__main__":
    from scipy import dot, array, power

    # Rosenbrock function
    def rosen(x):
        return - sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    def bla(x):
        return -sum(x**2)

    optim = CosyneM(zeros(30), .01, rosen)
    # example run (30-dimensional Rosenbrock)
    try:
        optim.load("test.gzip")
    except:
        print "failed loading"
        pass
    print optim.start(verbose=True, max_evals=1000000)
    optim.save("test.gzip")

