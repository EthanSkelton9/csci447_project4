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
    def createPopulationWeights(self, hidden_layers_num, size = 50):
        target_length = len(self.data.classes) if self.data.classes is not None else 1
        nrows = self.data.df.shape[1] - 1
        #hidden_vector = self.data.hidden_vectors[hidden_layers_num - 1]    Need to set up default hidden vectors
        hidden_vector = [3, 2]
        weight_list = pd.Series(size * [0]).map(lambda i: self.NN.list_weights(nrows, hidden_vector, target_length))
        chromosome_list = weight_list.map(self.NN.matrix_to_list)
        fitness_list = weight_list.map(lambda ws: self.fitness()(ws = ws))
        prob_list = self.NN.prob_distribution(fitness_list.to_numpy().reshape(-1, 1)).flatten()
        loc_list = prob_list.cumsum()
        df = pd.DataFrame({"Chromosome": chromosome_list, "Fitness": fitness_list,
                           "Probability": prob_list, "Location": loc_list})
        df.to_csv("InitialPopulation.csv")
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
        return tuple(pd.Series([r1, r2]).map(lambda r: l.loc[l.map(lambda l: l > r)].index[0]))




