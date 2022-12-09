import random
from neuralNet import Neural_Net
from preprocessing import Preprocessing
import pandas as pd
from DataDictionary import DataDictionary
import os
import numpy as np
from CrossValidation import CrossValidation
from itertools import product as prod
from Population import Population

class Video():
    def sampleOutput(self):
        DD = DataDictionary()
        dataobjects = DD.dataobjects(True, ["SoyBean", "Hardware"])
        models = ["GA", "DE", "PSO"]
        hyp_dicts = [{'p_c': 0.95, 'p_m': 0.01, 'pop_size': 10}, # For GA
                {'p_c': 0.95, 'p_m': 0.01, 'pop_size': 10}, # For DE
                {'p_w': 0.1, 'p_c': 1.3, 'pop_size': 10}]   # For PSO
        for (data, (model, hyp_dict)) in prod(dataobjects, zip(models, hyp_dicts)):
            print("===========================================================")
            print("Using dataset {} and model {}, we predict on a test fold.".format(data, model))
            CV = CrossValidation(data)
            p = CV.stratified_partition(10)
            test_set = CV.training_test_dicts(data.df, p)[1][0]
            sample = CV.predict_from_series(model, 2, test_set)(hyp_dict)
            print("Our Predicted Sample:")
            print(sample)

    def operations_GA(self):
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        P = Population(data)
        print("#########################################")
        pop_df = P.createPopulation(0, data.df, 4)
        print("Population")
        print(pop_df)
        print("===============================")
        (p1, p2) = P.selection(pop_df)
        print("From Selection ... ")
        print("Parent 1: {}".format(p1))
        print("Parent 2: {}".format(p2))
        print("---------------------")
        (c1, c2) = P.crossover(p1, p2)
        print("After Crossover ...")
        print("Child 1: {}".format(c1))
        print("Child 2: {}".format(c2))
        print("~~~~~~~~~~~~~")
        mc2 = P.mutation(c2, 0.5)
        print("Mutated Child 2 with Mutation Rate of 0.5")
        print(mc2)

    def operations_DE(self):
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        P = Population(data)
        print("#########################################")
        pop_df = P.createPopulation(0, data.df, 4)
        print("Population")
        print(pop_df)
        print("===============================")
        (p1, p2) = P.selection(pop_df)
        print("From Selection ... ")
        print("Parent 1: {}".format(p1))
        print("Parent 2: {}".format(p2))
        print("---------------------")
        (mp1, mp2) = tuple(pd.Series([p1, p2]).map(lambda c: P.mutation(c, 0.5)))
        print("Mutated Parents with Mutation Rate of 0.5")
        print("Mutated 1: {}".format(mp1))
        print("Mutated 2: {}".format(mp2))
        print("~~~~~~~~~~~~~")
        (c1, c2) = P.crossover(mp1, mp2)
        print("After Crossover")
        print("Child 1: {}".format(c1))
        print("Child 2: {}".format(c2))



    def operations_PSO(self):
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        P = Population(data)
        print("#########################################")
        pop_df = P.createPopulation(0, data.df, 4)
        print("Population")
        print(pop_df)
        print("==============PSO=================")
        v = pd.DataFrame(1, index = range(len(pop_df["Chromosome"][0])), columns=range(pop_df.shape[0]))
        print("Take the first and second Chomosome of the population")
        print("First-----------")
        print(pop_df['Chromosome'][0])
        print("Second-----------")
        print(pop_df['Chromosome'][1])

        if(pop_df['Fitness'][0] > pop_df["Fitness"][1]):
            lbest = np.array(pop_df['Chromosome'][0])
            gbest = np.array(pop_df['Chromosome'][0])
        else:
            lbest = np.array(pop_df['Chromosome'][1])
            gbest = np.array(pop_df['Chromosome'][1])
        print("\n")
        print("Compare both of the Chromosomes and decide which is best")
        
        print("Local Best before first generation")
        print(lbest)
        print("\n")
        print("Global Best before first generation")
        print(gbest)
        print("\n")
        r=random.uniform(.1, 1)
        chrome_at_0 = pop_df["Chromosome"][0]
        neg_chrome_at_0 = [ -x for x in chrome_at_0]

        chrome_at_1 = pop_df["Chromosome"][1]
        neg_chrome_at_1 = [ -x for x in chrome_at_1]
        print("Go through and calculate velocities to change the chromosome")
        v[0] = 0.7*v[0] + 1.7*r*(np.array(neg_chrome_at_0) + lbest) + 1.7*r*(np.array(neg_chrome_at_0) + gbest)
        r=random.uniform(0, 1)
        v[1] = 0.7*v[0] + 1.7*r*(np.array(neg_chrome_at_1) + lbest) + 1.7*r*(np.array(neg_chrome_at_1) + gbest)
        
        print("Velocity calculated for Chromosome 1")
        print(v[0])
        print("Velocity calculated for Chromosome 2")
        print(v[1])

        c0 = chrome_at_0 + v[0]
        c1 = chrome_at_1 + v[1]
        
        print("==========================================================================")
        print("New Chromosome 0 after being added to the velocities")
        print(c0.tolist())
        print("New Chromosome 1 after being added to the velocities")
        print(c1.tolist())
        if(sum(c0)>sum(c1)):
            lbest = c0
        else:
            lbest = c1
        print("Take the fitness of the two Chromosomes")
        print("New Local Best")
        print(lbest)
        print("==========================================================================")
        print("Original gbest:")
        print(gbest)
        print("Now we have to calculate to see if there is a new global best")
        if(sum(lbest) > sum(gbest)):
            gbest = lbest
        print("Check to see if gbest need to be changed")
        print(gbest)
        print("#########################################")


    def average_performance(self):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        DD = DataDictionary()
        data = DD.dataobject(True, "Abalone")
        P = Population(data)
        "Abalone_Model"
        print("#########################################")
        
        #regression ---------------------------------
        print("==============================Abalone Dataset==============================")
        analysis0GA = pd.read_csv("Abalone/Abalone_Model_GA_0_HiddenLayers_Analysis.csv", index_col=0)
        analysis1GA = pd.read_csv("Abalone/Abalone_Model_GA_1_HiddenLayers_Analysis.csv", index_col=0)
        analysis2GA = pd.read_csv("Abalone/Abalone_Model_GA_2_HiddenLayers_Analysis.csv", index_col=0)
        print("+++++++++++++++++++++++|Abalone GA 0 Layers|+++++++++++++++++++++++")
        print(analysis0GA)
        print("+++++++++++++++++++++++|Abalone GA 1 Layers|+++++++++++++++++++++++")
        print(analysis1GA)
        print("+++++++++++++++++++++++|Abalone GA 2 Layers|+++++++++++++++++++++++")
        print(analysis2GA)


        analysis0DE = pd.read_csv("Abalone/Abalone_Model_DE_0_HiddenLayers_Analysis.csv", index_col=0)
        analysis1DE = pd.read_csv("Abalone/Abalone_Model_DE_1_HiddenLayers_Analysis.csv", index_col=0)
        analysis2DE = pd.read_csv("Abalone/Abalone_Model_DE_2_HiddenLayers_Analysis.csv", index_col=0)
        print("+++++++++++++++++++++++|Abalone DE 0 Layers|+++++++++++++++++++++++")
        print(analysis0DE)
        print("+++++++++++++++++++++++|Abalone DE 1 Layers|+++++++++++++++++++++++")
        print(analysis1DE)
        print("+++++++++++++++++++++++|Abalone DE 2 Layers|+++++++++++++++++++++++")
        print(analysis2DE)


        analysis0PSO = pd.read_csv("Abalone/Abalone_Model_PSO_0_HiddenLayers_Analysis.csv", index_col=0)
        analysis1PSO = pd.read_csv("Abalone/Abalone_Model_PSO_1_HiddenLayers_Analysis.csv", index_col=0)
        analysis2PSO = pd.read_csv("Abalone/Abalone_Model_PSO_2_HiddenLayers_Analysis.csv", index_col=0)
        print("+++++++++++++++++++++++|Abalone PSO 0 Layers|+++++++++++++++++++++++")
        print(analysis0PSO)
        print("+++++++++++++++++++++++|Abalone PSO 1 Layers|+++++++++++++++++++++++")
        print(analysis1PSO)
        print("+++++++++++++++++++++++|Abalone PSO 2 Layers|+++++++++++++++++++++++")
        print(analysis2PSO)






























