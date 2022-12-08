from neuralNet import Neural_Net
import pandas as pd
import numpy as np
import functools
import random
from operator import add

class Population:
    def __init__(self, data):
        self.data = data
        self.NN = Neural_Net(data)

    '''
    @param hidden_layers_num
    @return: a pandas data frame that has randomly generated weights with fitness, probability, and location
    '''
    def createPopulation(self, num_hidden, dataset, size = 50):
        target_length = len(self.data.classes) if self.data.classes is not None else 1
        nrows = self.data.df.shape[1] - 1
        hidden_vector = self.data.hidden_vectors[num_hidden - 1] if num_hidden > 0 else []
        weight_series = pd.Series(int(size) * [0]).map(lambda i: self.NN.list_weights(nrows, hidden_vector, target_length))
        chromosome_series = weight_series.map(self.NN.matrix_to_list)
        return self.populationDF(num_hidden, chromosome_series, dataset, weight_series)

    '''
    @param num_hidden: the number of hidden layers
    @param chromosome_series: a pandas series of chromosomes
    @param ws: an optional pandas series of weights
    @return a pandas dataframe of chromosomes with their respective fitness, probability, and bucket range
    '''
    def populationDF(self, num_hidden, chromosome_series, dataset, ws = None):
        weight_series = chromosome_series.map(lambda c: self.NN.list_to_matrix(c, num_hidden)) if ws is None else ws
        fitness_series = weight_series.map(lambda ws: self.fitness(df = dataset)(ws=ws))
        prob_list = self.NN.prob_distribution(fitness_series.to_numpy().reshape(-1, 1)).flatten()
        loc_list = prob_list.cumsum()
        df = pd.DataFrame({"Chromosome": chromosome_series, "Fitness": fitness_series,
                           "Probability": prob_list, "Location": loc_list})
        return df

    '''
    @param n: the size of the data subset we are predicting
    @param df: the data set we are predicting on
    @return: a function that takes a hidden vector and weights and returns the fitness
    '''
    def fitness(self, n=None, df=None):
        pred_set = self.NN.predict_set(n, df)

        def f(ws=None, hidden_vector=None):
            return 1 / pred_set(ws, hidden_vector, error=True)

        return f

    '''
    @param pop_df: a population dataframe
    @return: the fittest chromosome in the population
    '''
    def getBest(self, pop_df):
        pop_fit = pop_df["Fitness"]
        bestfitness = pop_fit.max()
        bestindex = pop_fit.loc[lambda f: f == bestfitness].index[0]
        return tuple(pop_df.loc[bestindex, ["Chromosome", "Fitness"]])

    '''
    @param pop_df: the dataframe that represents a population
    @return: a tuple of indices used for the chosen parents
    '''
    def selection(self, pop_df):
        l = pop_df["Location"]
        (r1, r2) = (random.random(), random.random())
        parent_indices = list(pd.Series([r1, r2]).map(lambda r: l.loc[l.map(lambda l: l > r)].index[0]))
        return tuple(pop_df.loc[parent_indices, "Chromosome"])

    '''
    @param p1: parent 1
    @param p2: parent 2
    '''
    def crossover(self, p1, p2):
        k = random.randint(0, len(p1))
        return (p1[0:k] + p2[k:], p2[0:k] + p1[k:])
    '''
    @param g: gene in a chromosome
    @return: a mutated gene
    '''
    def mutate_gene(self, g):
        return g + random.uniform(-1, 1)
    '''
    @param chr: chromosome in a list
    @param p_m: the mutation rate
    @return: a mutated chromosome
    '''
    def mutation(self, chr, p_m):
        return [self.perchance(p_m, self.mutate_gene)(g) for g in chr]
    '''
    @param prob: probability of returning output of function
    @param f: the function
    @return: a function that sometimes returns the original function's output and sometimes returns the input
    '''
    def perchance(self, prob, f):
        def g(*args):
            return f(*args) if random.random() < prob else args if len(args) > 1 else args[0]
        return g
    '''
    @param num_hidden: the number of hidden layers
    @param p_c: the crossover rate
    @param p_m: the mutation rate
    '''
    def generation(self, num_hidden, dataset, p_c, p_m):
        def f(pop_df):
            chromosome_list = []
            for i in range(pop_df.shape[0] // 2):
                (p1, p2) = self.selection(pop_df)
                (c1, c2) = self.perchance(p_c, self.crossover)(p1, p2)
                (mc1, mc2) = tuple(pd.Series([c1, c2]).map(lambda c: self.mutation(c, p_m)))
                chromosome_list += [mc1, mc2]
            return self.populationDF(num_hidden, pd.Series(chromosome_list), dataset=dataset)
        return f
    '''
    @param num_hidden: the number of hidden layers
    @param p_c: the crossover rate
    @param p_m: the mutation rate
    @param pop_size: the population size, which is size of population dataframe
    @param gens: the number of generations
    @param initial_pop: the initial population to use
    @return: (a pandas series of best fitness per generation, overall best chromosome for all generations)
    '''
    def run(self, num_hidden, dataset, p_c, p_m, pop_size, gens = 30, initial_pop = None):
        nextGen = self.generation(num_hidden, dataset, p_c, p_m)
        initial_pop = self.createPopulation(num_hidden, dataset, pop_size) if initial_pop is None else initial_pop
        def recurse(pop_df, i, curr_best_chr, curr_best_fit):
            (chr, fit) = self.getBest(pop_df)
            (best_c, best_f) = (chr, fit) if fit > curr_best_fit else (curr_best_chr, curr_best_fit)
            return best_c if i == gens else recurse(nextGen(pop_df), i + 1, best_c, best_f)
        return recurse(initial_pop, 0, None, 0)
    '''
    @param chr: chromosome
    @param num_hidden: number of hidden layers
    @param error: whether or not we want the error of prediction
    @return: predicted series of values if error is false or the prediction error if error is true
    '''
    def predict_with_chr(self, dataset, chr, num_hidden, error = False):
        return self.NN.predict_set(df = dataset)(self.NN.list_to_matrix(chr, num_hidden), error = error)


    '''
    @param num_hidden: the number of hidden layers
    @param p_c: the crossover rate
    @param p_m: the mutation rate
    '''
    def generationDE(self, num_hidden, dataset, p_c, p_m):
        def f(pop_df):
            chromosome_list = []
            for i in range(pop_df.shape[0] // 2):
                (p1, p2) = self.selection(pop_df)
                
                (mc1, mc2) = tuple(pd.Series([p1, p2]).map(lambda c: self.mutation(c, p_m)))
                (c1, c2) = self.perchance(p_c, self.crossover)(mc1, mc2)
                chromosome_list += [c1, c2]
            return self.populationDF(num_hidden, pd.Series(chromosome_list), dataset=dataset)
        return f
    '''
    @param num_hidden: the number of hidden layers
    @param p_c: the crossover rate
    @param p_m: the mutation rate
    @param pop_size: the population size, which is size of population dataframe
    @param gens: the number of generations
    @param initial_pop: the initial population to use
    @return: (a pandas series of best fitness per generation, overall best chromosome for all generations)
    '''
    def runDE(self, num_hidden, dataset, p_c, p_m, pop_size = 50, gens = 30, initial_pop = None):
        nextGen = self.generationDE(num_hidden, dataset, p_c, p_m)
        initial_pop = self.createPopulation(num_hidden, dataset, pop_size) if initial_pop is None else initial_pop
        fitness_list = []
        def recurse(pop_df, i, curr_best_chr, curr_best_fit):
            (chr, fit) = self.getBest(pop_df)
            (best_c, best_f) = (chr, fit) if fit > curr_best_fit else (curr_best_chr, curr_best_fit)
            fitness_list.append(fit)
            return (fitness_list, best_c) if i == gens else recurse(nextGen(pop_df), i + 1, best_c, best_f)
        return recurse(initial_pop, 0, None, 0)
  
    '''
    @param num_hidden: the number of hidden layers
    @param p_c: the crossover rate
    @param p_m: the mutation rate
    '''
    def generationPSO(self, num_hidden, dataset, p_c, p_m, c1, c2, v):
        def f(pop_df, pbest, gbest):
            lpbest = np.array(pbest)
            lgbest = np.array(gbest)
            for i in range(pop_df.shape[0]):
                r=random.uniform(0, 1)
                chrome_at_i = pop_df["Chromosome"][i]
                neg_chrome_at_i = [ -x for x in chrome_at_i]
                #v[i].add(c1*r1*((neg_chrome_at_i).add(pbest)) + c2*r2*((neg_chrome_at_i).add(gbest)))
                v[i] = v[i] + c1*r*(np.array(neg_chrome_at_i) + lpbest) + c2*r*(np.array(neg_chrome_at_i) + lgbest)
                chrome_at_i += v[i]
            return self.populationDF(num_hidden,  pd.Series(pop_df["Chromosome"]), dataset)
        return f
    '''
    @param num_hidden: the number of hidden layers
    @param p_c: the crossover rate
    @param p_m: the mutation rate
    @param pop_size: the population size, which is size of population dataframe
    @param gens: the number of generations
    @param initial_pop: the initial population to use
    @return: (a pandas series of best fitness per generation, overall best chromosome for all generations)
    '''
    def runPSO(self, num_hidden, dataset, p_c, p_m , c1, c2, pop_size = 50, gens = 50, initial_pop = None):
        velocity = pd.DataFrame(1, index = range(40), columns=range(50))
        nextGen = self.generationPSO(num_hidden, dataset, p_c, p_m, c1, c2, velocity)
        initial_pop = self.createPopulation(num_hidden, dataset, pop_size) if initial_pop is None else initial_pop
        velocity = pd.DataFrame(1, index = range(len(initial_pop["Chromosome"][0])), columns=range(initial_pop.shape[0]))
        fitness_list = []
        def recurse(pop_df, i, curr_best_chr, curr_best_fit):
            (chr, fit) = self.getBest(pop_df)
            (best_c, best_f) = (chr, fit) if fit > curr_best_fit else (curr_best_chr, curr_best_fit)
            fitness_list.append(fit)
            
            return (fitness_list, best_c) if i == gens else recurse(nextGen(pop_df, chr, best_c), i + 1, best_c, best_f) 
                # 
        return recurse(initial_pop, 0, None, 0)











