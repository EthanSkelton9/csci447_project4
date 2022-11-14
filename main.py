from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os
import numpy as np



def main_Ian():
    def f1():
        DD = DataDictionary()
        DD.dataobjects(False, ["SoyBean","Abalone","Glass", "Hardware", "BreastCancer", "ForestFires"])
    return f1()



def mainEthan(): 
    pass  
    
if __name__=="__main__":
    main_Ian()
    #main()

    