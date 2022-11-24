from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
from neuralNet import Neural_Net 
import os
import numpy as np



def main_Ian():
    def f1():
        DD = DataDictionary()
        data = DD.dataobject(True, "ForestFires")
        NN  = Neural_Net(data)
        result = NN.predict_set()([4, 4], error=False)
        fitness = NN.fitness()([4, 4])
        print("Result: {}".format(result))
        print("Fitness: {}".format(fitness))
        print("Classes: {}".format(data.classes))
        print("Target: {}".format(NN.targetdf()))
    def f2():
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
        print("Chromosome? {}".format(NN.matrix_to_list(ws)))
        print("Number of Entries in List of Weights: {}".format(len(NN.matrix_to_list(ws))))

    return f2()



def mainEthan(): 
    DD = DataDictionary()
    data = DD.dataobject(True, "ForestFires")
    NN  = Neural_Net(data)  
    NN.listofweights()
    
if __name__=="__main__":
    #main_Ian()
    mainEthan()

    