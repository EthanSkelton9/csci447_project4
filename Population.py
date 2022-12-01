from neuralNet import Neural_Net
import pandas as pd
import numpy as np
import functools
import random
class Population:
    def __init__(self, data):
        self.data = data
        self.NN = Neural_Net(data)

    '''
    @param hidden_layers_num
    @return: a pandas data frame that has randomly generated weights with fitness, probability, and location
    '''
    def createPopulation(self, num_hidden, size = 50):
        target_length = len(self.data.classes) if self.data.classes is not None else 1
        nrows = self.data.df.shape[1] - 1
        hidden_vector = self.data.hidden_vectors[num_hidden - 1] if num_hidden > 0 else []
        weight_series = pd.Series(size * [0]).map(lambda i: self.NN.list_weights(nrows, hidden_vector, target_length))
        chromosome_series = weight_series.map(self.NN.matrix_to_list)
        return self.populationDF(num_hidden, chromosome_series, weight_series)

    def populationDF(self, num_hidden, chromosome_series, ws = None):
        weight_series = chromosome_series.map(lambda c: self.NN.list_to_matrix(c, num_hidden)) if ws is None else ws
        fitness_series = weight_series.map(lambda ws: self.fitness()(ws=ws))
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
    @param pop_df: the dataframe that represents a population
    @return: a tuple of indices used for the chosen parents
    '''
    def selection(self, pop_df):
        l = pop_df["Location"]
        (r1, r2) = (random.random(), random.random())
        parent_indices = list(pd.Series([r1, r2]).map(lambda r: l.loc[l.map(lambda l: l > r)].index[0]))
        return tuple(pop_df.loc[parent_indices, "Chromosome"])

    def perchance(self, prob, f):
        def g(*args):
            return f(*args) if random.random() < prob else args
        return g

    def generation(self, num_hidden, pop_df, p_c):
        chromosome_list = []
        for i in range(pop_df.shape[0] // 2):
            (p1, p2) = self.selection(pop_df)
            (c1, c2) = self.perchance(p_c, self.NN.crossover)(p1, p2)
            chromosome_list += [c1, c2]
        return self.populationDF(num_hidden, pd.Series(chromosome_list))








