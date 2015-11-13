import numpy as np

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
            selected = np.array([i for i in np.arange(N) if shuffle_mat[i,g]]).astype(np.int)
            move_to = selected.copy()
            np.random.shuffle(move_to)

            genomes[selected, g] = genomes[move_to, g]

        #mutation
        genomes += mutation_distribution(size=genomes.shape) * self.noise
