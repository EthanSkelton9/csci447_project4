from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
from neuralNet import Neural_Net 
import os
import numpy as np
from Population import Population
from CrossValidation import CrossValidation



def main_Ian(i):
    if i == 1:
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        CV = CrossValidation(data)
        start_hyp_dict = {'p_c':[0.8, 0.9], 'p_m':[0.01, 0.03], 'pop_size':[30, 50]}
        CV.test(start_hyp_dict)
    if i == 2:
        DD = DataDictionary()
        data = DD.dataobject(True, "Glass")
        CV = CrossValidation(data)
        start_hyp_dict = {'p_c': [0.75, 0.8, 0.85, 0.9, 0.95],
                          'p_m': [0.01, 0.02, 0.03, 0.04, 0.05],
                          'pop_size': [10, 20, 30, 40, 50]}
        CV.analysis("DE",1,['p_c', 'p_m', 'pop_size'], start_hyp_dict)
    if i == 3:
        DD = DataDictionary()
        data = DD.dataobject(True, "SoyBean")
        P = Population(data)
        (fitness, chr) = P.run(num_hidden=2, p_c=0.95, p_m=0.01, pop_size=50, gens=50)
        print(P.predict_with_chr(chr, 2))
    if i == 4:
        DD = DataDictionary()
        data = DD.dataobject(True, "Hardware")
        CV = CrossValidation(data)
        start_hyp_dict = {'p_w': [0.1, 0.3, 0.5, 0.7, 0.9], 
                          'p_c': [1.3, 1.5, 1.7, 1.9, 2.1],
                          'pop_size': [10, 20, 30, 40, 50]}
        CV.analysis("PSO",0,['p_w', 'p_c', 'pop_size'], start_hyp_dict)
    if i == 5:
        DD = DataDictionary()
        data = DD.dataobject(True, "Hardware")
        CV = CrossValidation(data)
        start_hyp_dict = {'p_c': [0.75, 0.8, 0.85, 0.9, 0.95],
                          'p_m': [0.01, 0.02, 0.03, 0.04, 0.05],
                          'pop_size': [10, 20, 30, 40, 50]}
        CV.analysis("GA",1,['p_c', 'p_m', 'pop_size'], start_hyp_dict)






def mainEthan(): 
    DD = DataDictionary()
    data = DD.dataobject(True, "SoyBean")
    P = Population(data)
    chr = P.runPSO(num_hidden=2, dataset = data.df, p_c=0.95, p_m=0.01, c1 = 1, c2 = 1, pop_size=50, gens=50)
    # print(P.predict_with_chr(chr, 2))
    
if __name__=="__main__":
    main_Ian(2)
    #main_Ian(5)
    #mainEthan()


    