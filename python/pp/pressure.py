#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt
plt.ion()


N_SAMPLES = 4
N_PROBLEMS = 5
PROBLEM_GEN = 2 * N_SAMPLES

N_IND = 5
IND_GEN = 2*5 + 5 + 5*1

#mutation_distribution = mutation_distribution
mutation_distribution = np.random.standard_cauchy

class Mixer:
    def __init__(self, noise, shuffle_prob):
        self.noise = noise
        self.shuffle_prob = shuffle_prob

    def mix(self, genomes, fitnesses, reverse=False):
        N, G = genomes.shape[0], genomes.shape[1]
        N_NEW = 0
        N_CLONE = N - N_NEW


        ranking = np.argsort(fitnesses)
        if not reverse: #normal is higher fitness = better, unless reversed
            ranking = ranking[::-1]

        #create selection mat
        select_prob = np.zeros(N_CLONE)
        for i, o in zip(np.arange(N_CLONE), ranking):
            select_prob[i] = max(N_CLONE - i - 1, 0)
            # select_prob[i] = max(N / 2 - i - 1, 0)
        select_prob /= np.sum(select_prob)
        for i in np.arange(1, N_CLONE):
            select_prob[i] += select_prob[i-1]
        #selection
        new_genomes = np.zeros(genomes.shape)
        selection = np.random.uniform(size=N_CLONE)


        for i in np.arange(N_CLONE):
            winner = np.searchsorted(select_prob, selection[i])
            #print(winner, ranking[winner], select_prob, selection[i])
            new_genomes[i,:] = genomes[ranking[winner], :]

        new_genomes[N_CLONE:,:] = mutation_distribution(size=new_genomes[N_CLONE:,:].shape)

        genomes[:] = new_genomes

        #shuffling
        shuffle_mat = np.random.uniform(size=genomes.shape) < self.shuffle_prob

        for g in np.arange(G):
            selected = [i for i in np.arange(N) if shuffle_mat[i,g]]
            move_to = selected.copy()
            np.random.shuffle(move_to)
            genomes[selected, g] = genomes[move_to, g]

        #mutation
        genomes += mutation_distribution(size=genomes.shape) * self.noise


class NN:
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden


    def f(self, ind, x):
        l2 = ind[:self.n_hidden]
        l1 = ind[self.n_hidden:self.n_hidden*(1+2)].reshape(self.n_hidden, 2)
        bias = ind[self.n_hidden*(1+2):]

        return np.tanh(np.dot(l2, bias + np.tanh(np.dot(l1, x))))

    def forward(self, ind, prob):
        err = 0.0

        for n in np.arange(N_SAMPLES):
            x = prob[n*2:(n+1)*2]

            f = self.f(ind, x)
            if n % 2 == 0:
                err += ((1.0 - f) / 2) ** 2
            else:
                err += ((-1.0 - f) / 2) ** 2
        return err / N_SAMPLES

    def picture(self, ind, problem):
        tmp = problem.reshape(N_SAMPLES, 2).T

        width, height = 20, 20
        pic = np.zeros((width, height))
        for i, x in enumerate(np.linspace(np.min(tmp[0,:]), np.max(tmp[0,:]), width)):
            for j, y in enumerate(np.linspace(np.max(tmp[1,:]), np.min(tmp[1,:]), height)):
                pic[j, i] = self.f(ind, [x,y])
        return pic


problems = mutation_distribution(size=(N_PROBLEMS, PROBLEM_GEN))
inds = mutation_distribution(size=(N_IND, IND_GEN))
errs = np.zeros((N_PROBLEMS, N_IND))

problem_fitness = np.zeros(N_PROBLEMS)
ind_fitness = np.zeros(N_IND)

#Problem class
nn = NN(n_hidden = 5)
ind_mixer = Mixer(.03, .01)
problem_mixer = Mixer(.03, .01)

epoch = int(0)

fig, ax = plt.subplots(2)

while True:
    print("Epoch:", epoch)

    #Evaluate all problem - solver pairs
    for ip, p in enumerate(problems):
        for ii, ind in enumerate(inds):
            errs[ip, ii] = nn.forward(ind, p)

    #Calculate problem fitness
    for ip, prob in enumerate(problems):
        #problem_fitness[ip] = np.max(errs[ip, :]) - np.min(errs[ip, :]) - .01 * np.sum(prob ** 2)
        #problem_fitness[ip] = np.std(errs[ip, :]) - .01 * np.sum(prob ** 2)
        problem_fitness[ip] = np.sum(errs[ip, :] ** 2) - .01 * np.sum(prob ** 2)

    #Calculate solver fitness
    for ii, ind in enumerate(inds):
        ind_fitness[ii] = np.mean(errs[:, ii]) + .05 * np.sum(ind ** 2)

    #Who are the best?
    best_prob, best_ind = np.argmax(problem_fitness), np.argmin(ind_fitness)

    #plt.figure(1)
    #ax.clf()

    print("if:", ind_fitness)
    print("pf:", problem_fitness)
    print("i:", inds[best_ind])
    print("p:", problems[best_prob])
    for n in np.arange(N_SAMPLES):
        x = problems[best_prob][n*2:(n+1)*2]

        print(x, nn.f(inds[best_ind], x))

    global landscape
    landscape = ax[0].matshow(nn.picture(inds[best_ind], problems[best_prob]), cmap='gray')
    #plotting
    if True:#epoch % 100 == 0 and epoch > 0:
        landscape.set_data(nn.picture(inds[best_ind], problems[best_prob]))

        ax[1].cla()
        for p in problems:
            plot_problem = p.reshape(N_SAMPLES,2).T
            ax[1].plot(plot_problem[0,::2], plot_problem[1,::2], 'ro', plot_problem[0,1::2], plot_problem[1,1::2], 'bx')
        plt.draw()
        plt.pause(.001)

    #Mix all the genomes!
    if (epoch // 25) % 2 == 0:
        problem_mixer.mix(problems, problem_fitness)
    else:
        ind_mixer.mix(inds, ind_fitness, reverse=True) #reversed because fitness = cost


    #plt.figure(2)
    #plt.clf()
    #pic = nn.picture(inds[best_ind])
    #plt.matshow(pic)

    # print(errs)

    # print(np.min(ind_fitness))
    # print(np.max(problem_fitness))
    # print(inds[best_ind])
    # print(problems[best_prob])
    epoch += 1

