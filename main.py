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
        result = NN.predict_set(20)([4, 4])
        print(result)
    return f1()



def mainEthan(): 
    pass  
    
if __name__=="__main__":
    main_Ian()
    #main()

    