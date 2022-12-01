from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
from neuralNet import Neural_Net 
import os
import numpy as np
from Population import Population



def main_Ian(i):
    if i == 1:
        pass
    if i == 2:
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        NN = Neural_Net(data)
        target_length = len(NN.data.classes) if NN.data.classes is not None else 1
        nrows = NN.data.df.shape[1] - 1
        print("nrows: {}".format(nrows))
        hidden_vector = data.hidden_vectors[1]
        ws = NN.list_weights(nrows, hidden_vector, target_length)
        print("Weights: {}:".format(ws))
        num_genes = 0
        layers = [nrows] + hidden_vector + [1]
        for i in range(len(layers) - 1):
            num_genes = num_genes + layers[i] * layers[i + 1]
        print("Number of Entries in List of Weights: {}".format(num_genes))
        chr = NN.matrix_to_list(ws)
        print("Chromosome? {}".format(chr))
        print("Number of Genes in Chromosome: {}".format(len(chr)))
        print("Back to Weights")
        print(NN.list_to_matrix(chr, 2))
        #NN.listofweights()
    if i == 3:
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        P = Population(data)
        population = P.createPopulationWeights([3, 2], 50)
        print(population)
        (pi1, pi2) = P.selection(population)
        (p1, p2) = tuple(population.loc[[pi1, pi2], "Chromosome"])
        (c1, c2) = P.NN.crossover(p1, p2)
        print("Parent 1: {}".format(p1))
        print("Child 1: {}".format(c1))
        print("Parent 2: {}".format(p2))
        print("Child 2: {}".format(c2))
        print("---")
        print("Try Ian's Code")
        print("---")
        (c1, c2) = P.NN.crossover_Ian(p1, p2)
        print("Parent 1: {}".format(p1))
        print("Child 1: {}".format(c1))
        print("Parent 2: {}".format(p2))
        print("Child 2: {}".format(c2))



def mainEthan(): 
    DD = DataDictionary()
    data = DD.dataobject(True, "ForestFires")
    NN  = Neural_Net(data)  
    NN.listofweights()
    
if __name__=="__main__":
    main_Ian(2)
    #mainEthan()

    