from neuralNet import Neural_Net
import pandas as pd
import numpy as np
import functools
class Population:
    def __init__(self, data):
        self.data = data
        self.NN = Neural_Net(data)

    '''
    @param hidden_layers_num
    @return: a pandas data frame that has randomly generated weights with fitness, probability, and location
    '''
    def createPopulationWeights(self, hidden_layers_num):
        target_length = len(self.data.classes) if self.data.classes is not None else 1
        nrows = self.data.df.shape[1] - 1
        #hidden_vector = self.data.hidden_vectors[hidden_layers_num - 1]    Need to set up default hidden vectors
        hidden_vector = [3, 2]
        weight_list = pd.Series(50 * [0]).map(lambda i: self.NN.list_weights(nrows, hidden_vector, target_length))
        fitness_list = weight_list.map(lambda ws: self.NN.fitness()(ws = ws))
        prob_list = self.NN.prob_distribution(fitness_list.to_numpy().reshape(-1, 1)).flatten()
        addLocation = lambda locs, prob: locs + [locs[-1] + prob] if len(locs) > 0 else [prob]
        loc_list = functools.reduce(addLocation, prob_list, [])
        df = pd.DataFrame({"Weights": weight_list, "Fitness": fitness_list,
                           "Probability": prob_list, "Location": loc_list})
        return df


