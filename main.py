from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
from neuralNet import Neural_Net 
import os
import numpy as np
from Population import Population



def main_Ian(i):
    if i == 1:
        DD = DataDictionary()
        data = DD.dataobject(True, "ForestFires")
        NN  = Neural_Net(data)
        result = NN.predict_set()(hidden_vector = [4, 4], error=False)
        fitness = NN.fitness()(hidden_vector = [4, 4])
        print("Result: {}".format(result))
        print("Fitness: {}".format(fitness))
        print("Classes: {}".format(data.classes))
        print("Target: {}".format(NN.targetdf()))
    if i == 2:
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        NN = Neural_Net(data)
        target_length = len(NN.data.classes) if NN.data.classes is not None else 1
        nrows = NN.data.df.shape[1] - 1
        print("nrows: {}".format(nrows))
        hidden_vector = [3, 2]
        ws = NN.list_weights(nrows, hidden_vector, target_length)
        print("Weights: {}:".format(ws))
        print("Number of Entries in List of Weights: {}".format(nrows * 3 + 3 * 2 + 2 * 1))
        chr = NN.matrix_to_list(ws)
        print("Chromosome? {}".format(chr))
        print("Number of Entries in List of Weights: {}".format(len(chr)))
        print("Back to Weights: {}".format(NN.list_to_matrix(chr, [3, 2])))
        NN.listofweights()
    if i == 3:
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        P = Population(data)
        population = P.createPopulationWeights([3, 2], 50)
        print(population)
        print(P.chooseParents(population))

def mainEthan(): 
    DD = DataDictionary()
    data = DD.dataobject(True, "ForestFires")
    NN  = Neural_Net(data)  
    NN.listofweights()
    
if __name__=="__main__":
    main_Ian(3)
    #mainEthan()

    