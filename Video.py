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
        print("===============================")
        velocity = pd.DataFrame(1, index=range(len(pop_df["Chromosome"][0])), columns=range(pop_df.shape[0]))
        (chr, fit) = P.getBest(pop_df)

