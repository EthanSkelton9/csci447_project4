from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
from neuralNet import Neural_Net 
import os
import numpy as np



def main_Ian():
    def f1():
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        NN  = Neural_Net(data)
        result = NN.predict_set()([4, 4], error=False)
        fitness = NN.fitness()([4, 4])
        print("Result: {}".format(result))
        print("Fitness: {}".format(fitness))
        print("Classes: {}".format(data.classes))
        print("Target: {}".format(NN.targetdf()))
    return f1()



def mainEthan(): 
    pass  
    
if __name__=="__main__":
    main_Ian()
    #main()

    