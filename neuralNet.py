import pandas as pd
import math
import numpy as np
import random
from tail_recursive import tail_recursive
import functools

class Neural_Net:

    def __init__(self, data):
        self.data = data

    '''
    @param data: the data set we are using
    @return: function that takes a row of the data and returns its numpy vector
    '''
    def vec(self, data = None):
        data = self.data if data == None else data
        '''
        @param i: index of the data set
        @return: numpy row vector of the data with a 1 appended at the beginning for the bias weight
        '''
        def f(i):
            return np.array(data.df.loc[i, data.features_ohe])
        return f
    '''
    @param matrix: a numpy matrix that represents the linear transformation
    @return: a function that takes another matrix and multiplies by the first matrix
    '''
    def linear_transformation(self, matrix):
        def f(matrix2):
            return matrix @ matrix2
        return f

    '''
    @param vector: a numpy vector
    @return: a vector that has each entry divided by the sum of the original
    '''
    def prob_distribution(self, matrix):
        return matrix @ np.linalg.inv(np.diag(np.ones(matrix.shape[0]) @ matrix))

    '''
    @param x: a real number
    @return: the sigmoid value
    '''
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    '''
    @param x: a vector of real numbers
    @return: a vector where each element is the respective sigmoid value
    '''
    def sigmoid_v(self, x):
        return np.vectorize(self.sigmoid)(x)

    '''
    @param x: a vector of real numbers
    @return: a vector where each element is the respective derivative sigmoid value
    '''
    def dsigmoid_v(self, x):
        if x is None:
            return 1
        else:
            return x * (1 - x)
    '''
    @param f: left function
    @param g: right function
    @return: a composed function that evaluates the left function and then the right function
    '''
    def left_compose(self, f, g):
        return lambda x: g(f(x))

    '''
    @param f_list: a list of functions
    @return: a function that is the left composition of the given functions
    '''
    def lcompose_fxns(self, f_list):
        return functools.reduce(self.left_compose, f_list)

    '''
    @param n: the row dimension
    @param m: the column dimension
    @return: a random n by m matrix
    '''
    def rand_w(self, n, m):
        return (.01 - (-.01)) * np.random.random_sample((n, m)) - .01

    '''
    @param index: an index of a dataframe
    @return: a randomly shuffled index
    '''
    def permute(self, index):
        return random.sample(index, len(index))

    '''
    @param: an ordered index of the classes
    @return: a function that takes an index of the data and returns a vector with a one in its corresponding class
    '''
    def targetdf(self):
        targets = self.data.df.loc[:, "Target"]
        def f_classification():
            mdf = pd.DataFrame(np.zeros((self.data.df.shape[0], len(self.data.classes))))
            for i in mdf.index:
                mdf.loc[i, :] = self.data.classes.map(lambda cl: int(cl == targets[i]))
            return mdf
        return f_classification() if self.data.classes is not None else targets

    '''
    @param pred_vec: a numpy vector of predicted values
    @return: mean squared error
    '''
    def mse_np(self, pred_vec):
        actual_vec = np.array(self.targetdf())
        return math.pow(np.linalg.norm(pred_vec - actual_vec), 2) / len(pred_vec)
    '''
    @param pred_df: a matrix showing predicted probabilities for respective classes
    @return: cross entropy error
    '''
    def ce_np(self, pred_matrix):
        return - np.sum(np.sum(self.targetdf() * np.vectorize(np.log)(np.transpose(pred_matrix))))
    '''
    @param pred: numpy array of predictions
    @return: the appropriate error
    '''
    def error_np(self, pred):
        return self.ce_np(pred) if self.data.classification else self.mse_np(pred)

    '''
    @param predicted: a series of predicted values
    @param actual: a series of actual target values
    @return: the mean squared error
    '''
    def mean_squared_error(self, predicted):
        actual = self.data.df.loc[predicted.index, "Target"]
        return math.pow(np.linalg.norm(predicted.to_numpy() - actual.to_numpy()), 2) / len(predicted)
    
    def cross_entropy(self, predicted, targetvec):
        error = 0
        for i in predicted.index:
            error =+ -(targetvec(i) @ np.vectorize(np.log)(predicted[i][0]))
        return error
    
    def calc_error(self,y,r,data):
        if data.classification:
            return self.cross_entropy(y,  r)
        else:
            return self.mean_squared_error(y)

    '''
    @param predicted: numpy array of predicted values
    @return: a pandas series
    '''
    def to_series(self, predicted):
        return pd.DataFrame(predicted.transpose())

    '''
    @param pred_df: a dataframe of probabilities that it is the respective class
    @return: pandas index of predicted classes
    '''
    def predict_classes(self, pred_df):
        getclass = lambda i: self.data.classes[pred_df.loc[i, :].map(lambda x: x == max(pred_df.loc[i, :]))][0]
        return pd.Series(pred_df.index.map(getclass))

    def prediction(self, predicted):
        pred_class = predicted.copy()
        # print(predicted)
        for i in predicted.index:
            index = (np.where(predicted[i][0] == max(predicted[i][0])))[0][0]
            pred_class[i] = self.data.classes[index]
        # print(pred_class)
        return pred_class
    '''
    @param weights: a list of matrices
    @return: a function that takes a vector of features through the network
    '''
    def network_transformation(self, weights, error = False):
        sigmoid_compose = lambda f, g: self.lcompose_fxns([f, self.sigmoid_v, g])
        lin_trans = pd.Series(weights).map(self.linear_transformation)
        nt = functools.reduce(sigmoid_compose, lin_trans)
        (exp, probdist, ser, pred) = (np.vectorize(np.exp), self.prob_distribution, self.to_series, self.predict_classes)
        (beg, end) = ([nt, exp, probdist], [ser, pred]) if self.data.classification else ([nt], [ser])
        f_list = beg + [self.error_np] if error else beg + end
        return self.lcompose_fxns(f_list)

    '''
    @param n: the size of the data subset we are predicting
    @param df: the data set we are predicting on
    @return: a function that takes a hidden vector and weights and returns predicted values or error
    '''
    def predict_set(self, n = None, df = None):
        df = df if df is not None else self.data.df
        n = min(df.shape[0], n) if n is not None else df.shape[0]
        data_matrix = df.head(n).loc[:, self.data.features_ohe].to_numpy().transpose()
        target_length = len(self.data.classes) if self.data.classes is not None else 1
        nrows = self.data.df.shape[1] - 1
        def f(hidden_vector, ws = None, error = False):
            ws = pd.Series(self.list_weights(nrows, hidden_vector, target_length)) if ws is None else ws
            return self.network_transformation(ws, error)(data_matrix)
        return f

    '''
    @param n: the size of the data subset we are predicting
    @param df: the data set we are predicting on
    @return: a function that takes a hidden vector and weights and returns the fitness
    '''
    def fitness(self, n = None, df = None):
        pred_set = self.predict_set(n, df)
        def f(hidden_vector, ws = None):
            return 1 / pred_set(hidden_vector, ws, error = True)
        return f

    '''
    @param vec_func: function that takes an index value and gives the sample from the dataset as a vector
    @param r: series of actual target values
    @param eta: the learning rate
    @param index: the index of the data set we are updating
    @return ((permuted index, final weight vector), series of predicted target values)
    @used in: stochastic_online_gd
    '''
    def online_update(self, vec_func, r, eta, alpha, index):
        '''
        @param index_remaining: index left to iterate through
        @param w: the current weight matrix
        @param y_acc: the current set of accumulated predictions
        @return: new index, final weight matrix, and complete set of predictions after iterated through index
        '''

        @tail_recursive
        def f(index_remaining, ws, ss, y_acc):
            if len(index_remaining) == 0:  # if there is nothing more to iterate through
                y_idx, y_values = zip(*y_acc)  # unzip to get y index and y values
                return (self.permute(index), ws, ss, pd.Series(y_values, y_idx))
            else:
                i = index_remaining[0]  # the next index value
                x = vec_func(i)  # the next sample vector
                zs = [x] + self.calc_Hidden(ws, x, len(ws) - 1)                   # the input and hidden layers
                if self.data.classification:
                    yi = np.vectorize(np.exp)(ws.iloc[-1] @ zs[-1]).reshape(1, -1)                 # gives the exponent at each component
                    yi = yi / np.sum(yi)                                          # normalizes the vector
                else:
                    yi = (ws.iloc[-1] @ zs[-1])[0]                                # return a real value
                error = np.array([r(i) - yi])                                     # return errors at each of the outputs
                grads = []
                wzs = zip(ws, zs)
                previous_z = None
                for (w, z) in list(wzs)[::-1]:
                    grads = [np.outer(error * self.dsigmoid_v(previous_z), z)] + grads   # create gradient
                    error = error @ w                                               # back propagate error
                    previous_z = z
                if alpha != 0:
                    grads = pd.Series(zip(ss, grads)).map(lambda sg: alpha * sg[0] + (1-alpha) * sg[1]) #average grad
                new_ws = pd.Series(zip(ws, grads)).map(lambda wg: wg[0] + eta * wg[1])           #calculate new weights
                new_ss = None if ss is None else grads                                        #calculate new gradients
                return f.tail_call(index_remaining[1:], new_ws, new_ss, y_acc + [(i, yi)])
        return f


    '''
    @param data: the data set we are using
    @param n: the size of the subset of the data we are using
    @return a function that takes hyperparameters eta and max error and returns a series of predicted target values
    '''
    def stochastic_online_gd(self, df = None, n = None):
        df = self.data.df if df is None else df
        if n is None: n = df.shape[0]
        vec_func = self.vec(self.data)  # create vector function for data
        base_index = random.sample(list(df.index), k=n)  # create a shuffled index for iteration
        if self.data.classification:
            classes = self.data.df["Target"].unique()        # create unique class list
            r = self.targetvec(True, pd.Index(classes))      # create function that returns vector of a class
            target_length = len(classes)                     # target length needs to be number of classes
        else:
            r = self.targetvec(False)                        # create function that returns class
            target_length = 1                                # target length is just 1 for returning a real value
        '''
        @param eta: the learning rate
        @param max_error: the maximum tolerance used
        @return: function that uses the hyperparameters to return a series of predicted values
        '''

        def f(eta, hidden_vector, alpha = 0):
            nrows = self.data.df.shape[1] - 1
            ws_init = pd.Series(self.list_weights(nrows, hidden_vector, target_length))  # initial randomized weights
            ss_init = None if alpha is None else ws_init.map(lambda w: np.zeros(w.shape))  # initial gradients
            '''
            @param index: the index to iterate through
            @param start_w: the starting weight matrix to use for the epoch
            @return: new permuted index, weight matrix learned from data, and a series of predicted values
            '''

            def epoch(index, start_w, start_s, alpha):
                return self.online_update(vec_func, r, eta, alpha, index)(index, start_w, start_s, [])

            '''
            @param index: index after an epoch or starting the first epoch
            @param w: weight matrix after an epoch or starting the first epoch
            @param y: current prediction
            @return: the final prediction 
            '''
            def evaluate(index, w, s, y = None, prev_y = None, prev_error = None):
                if y is None:
                    new_index, final_w, final_s, new_y = epoch(index, w, s, alpha)  # run through first epoch
                    return evaluate(new_index, final_w, final_s, new_y)
                else:
                    if prev_error is None:
                        error = self.calc_error(y, r, self.data)                         # calculate error
                        new_index, final_w, final_s, new_y = epoch(index, w, s, alpha)
                        return evaluate(new_index, final_w, final_s, new_y, y, error)
                    else:
                        error = self.calc_error(y, r, self.data)
                        if error < prev_error:
                            new_index, final_w, final_s, new_y = epoch(index, w, s, alpha)
                            return evaluate(new_index, final_w, final_s, new_y, y, error)
                        else:
                            return prev_y

            return evaluate(base_index, ws_init, ss_init)

        return f

    #------------------------------------------------------------------------------------------------------
    
    '''
    create_weights() - creates a matrix of weights randomly
    @param length: length of hidden layers we want
    @param nrow: number of rows that we have before the hidden layer
    @return: a matrix of weights with dimensions length and nrows
    '''
    def create_rand_weights(nrows, length):
        matrix = np.random.rand(length, nrows)
        return matrix
    
    '''
    all_weights() - creates a list of all the weight matrix
    @param - length - number of hiden layers that we want
    @nfeatures - number of features that are in the dataset
    @layers - the vector of hidden layers
    @return weights - list of the weights that are calculated
    '''
    def list_weights(self, nfeatures, hid_layers, target_len):
        weights = [] #list of weight matrices that we will be returning
        layers = hid_layers.copy()
        layers.insert(0,nfeatures) #add nfeatures
        layers.append(target_len) #add target layer
        num_hidden = len(layers)
        
        for i in range(num_hidden-1):
            weights.append(Neural_Net.create_rand_weights(layers[i], layers[i+1]))
          
        return weights    
       
    
    
    '''
    calc_Hidden - calculates the hidden layers on the weights
    @param weights[] - weights  between the layers 
    @param row: the row of features values that we are using
    @param data - the data that we want to read in
    @length - number of hidden layers that we have
    
    @return hidden_layers - returns the hidden layers that we created
    '''
    def calc_Hidden(self, weights, row, num_hidden):
        hidden_layers = []
        layers = []
        
        layers.append(row)
        
        if(num_hidden == 0): #if there are no hidden layers
            return hidden_layers
        else:
            hidden_layers.append(self.sigmoid_v(weights[0]@row)) #find the first hidden layer
            for i in range(num_hidden-1):
                hidden_layers.append(self.sigmoid_v(weights[i+1]@hidden_layers[i])) #all hidden layers after
                
            return hidden_layers #return the hidden layers 
            
        return hidden_layers #return the hidden layers
    
    def classification_error(self, predictions):
        error = 0
        length = len(predictions)
        predictions = pd.DataFrame(self.prediction(predictions))
        predictions["Target"] = self.data.df["Target"]
        for i in predictions.index:
            pred = predictions.loc[i][0]
            target = predictions.loc[i]["Target"]
            if pred == target:
                error += 1
            else:
                error = error
        return 1-error/length
    
    def regression_error(self, predictions):
        error = 0
        length = len(predictions)
        predictions = pd.DataFrame(predictions)
        predictions["Target"] = self.data.df["Target"]
        for i in predictions.index:
            pred = predictions.loc[i][0]
            target = predictions.loc[i]["Target"]
            error += abs(pred - target)
        return error/length
    
    def tuning(self, df):
        def f(vector, eta, alpha):
            prev_error = 0
            error = 0
            i = 0

            # if next_layer:
            num_hidden = len(vector) + 1
            vector = vector + [0]
            while prev_error == 0 or error <= prev_error:
                if error != 0:
                    prev_error = error
                vector[num_hidden - 1] = i + 1
                predictions = self.stochastic_online_gd(df)(eta, hidden_vector = vector, alpha = alpha)
                # print(predictions)
                if self.data.classification:
                    error = self.classification_error(predictions)
                else:
                    error = self. regression_error(predictions)
                i += 1
            return i - 1
        return f

    '''
    crossover : with two parents create two new children for the next genereation
    @param p1: parent 1 
    @param p2: parent 2
    
    @return p1: kid 1, p2: kid 2
    '''
    def crossover(self, p1, p2):
        #finds the length of the chromosome
        length = len(p1)
        
        #sets each chromosome to a list so we can manipulate them
        p1 = list(p1)
        p2 = list(p2)
        
        #find a random place to split the chromosome on
        k = random.randint(0, length)
        
        #go through and create the kids
        for i in range(k, length):
            p1[i], p2[i] = p2[i], p1[i]
        p1 = ''.join(p1)
        p2 = ''.join(p2)
        #return the kids
        return p1, p2
        
    '''
    mutation: send in a chromosome and change it by a certain value
    
    @param chromosome : the set of strings that make up the chromosome
    
    @return chromosome : new mutated chromosome
    '''
    def mutation(self, chromosome):
        rand_num = random.uniform(-1, 1) #uniform number to add or subtract by
        
        chromosome += rand_num #add random number
        
        return chromosome

    #turn a matrix into a list
    def matrix_to_list(self, matrix):
        newlist = []
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                for k in range(len(matrix[i][j])):
                    newlist.append(matrix[i][j][k])
        return newlist
    
    
    #turn a list back into a matrix
    def list_to_matrix(self, list, hiddenlayers):
        listoflists = []
        indexList = 0
        for i in range(len(hiddenlayers)-1):
            innerlist = []
            for j in range(hiddenlayers[i + 1]):
                innerinnerlist = []
                for k in range(hiddenlayers[i]):
                    innerinnerlist.append(list[indexList])
                    indexList += 1
                innerlist.append(innerinnerlist)
            listoflists.append(innerlist)
        return np.array(listoflists, dtype = object )
        

    #driver to see if i can create matrix from list and vice versa
    def listofweights(self):
        target = 1
        if (self.data.classification):
            target = len(self.data.Target.unique())
        hiddenlayers = [3,4]
        nfeature = len(self.data.columns)
        matrix = self.list_weights(nfeature, hiddenlayers, 1)
        print(matrix)
        
        hiddenlayers.insert(0,nfeature)
        hiddenlayers.append(target)
        list = self.matrix_to_list(matrix)
        print("matrix to list -------------")
        print(list)
        matrix2 = self.list_to_matrix(list, hiddenlayers)
        print("list to matrix -----------")
        print(matrix2)
