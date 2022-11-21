import pandas as pd
import os


'''
Preprocessing class will allow us to create a clean dataset from the raw data that we give it
'''
class Preprocessing:
    '''
    __init__: will initialize the preprocessing class based on the location that the data is
    @param data_loc: the string location where the file we want to read in is
    '''
    def __init__(self, name, data_loc, columns, target_name, replace, classification, hidden_vectors):
        self.name = name
        self.data_loc = data_loc #set the data location of the class equal to the data location that was sent in
        self.df = None #set the actual data to None for the moment
        self.columns = columns #we need to say what the features are
        self.target_name = target_name
        self.replace = replace
        self.classification = classification
        self.hidden_vectors = hidden_vectors



    def __str__(self):
        return self.name

    '''
    save: saves the dataframe to its folder with a given suffix
    '''
    def save(self, suffix = None):
        self.df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_{}.csv".format(str(self), suffix))
        
    '''
    readcsv: will take the location of the file and convert it to pandas
    @return self.data - panda df of the csv for us to work with
    '''   
    def readcsv(self):
        dn = self.data_loc
        self.df = pd.read_csv(dn) #read the data location to pandas dataframe
        return self.df
    
    '''
    clean_missing: removes '?' that are in the dataframe and replaces them with another value
    @param replace: the value to replace the missing value with
    '''
    def clean_missing(self):
        if self.replace == None:
            pass
        else:
            for col_name in self.df.columns:
                self.df[col_name] = self.df[col_name].replace(['?'], [self.replace])
                
            self.df['Bare Nuclei'] = pd.to_numeric(self.df['Bare Nuclei'])
            # print(self.df['Bare Nuclei'][23])

    '''
    add_raw_data: defines the dataframe by the raw data
    '''
    def set_to_raw_data(self):
        self.df = pd.read_csv(self.data_loc, header = None)
    '''
    add_column_names: adds the column names to the data as well as define the features and target
    '''
    def add_column_names(self, df = None):
        if df == None: df = self.df
        df.columns = self.columns                  # Define columns of the data frame with what is given in initialization.
        target_column = df.pop(self.target_name)  # Remove the target column.
        self.features = df.columns                # Define the features of the data by the remaining columns.
        df.insert(len(df.columns), 'Target', target_column)  # Insert the target column at the end.
        self.df = df                              # Define the dataframe for the object.
    '''
    one_hot: transforms data by doing one hot encoding
    '''
    def one_hot(self):
        (features_numerical, features_categorical) = ([], [])
        features_categorical_ohe = []
        for f in self.features:
            try:
                self.df[f].apply(pd.to_numeric)  #sees if the column data can be considered numerica
                features_numerical.append(f)     #adds the column name as a numerical feature
            except:
                features_categorical.append(f)   #adds the column name as a categorical feature
                categories = set(self.df[f])     #creates set of all categories from this categorical feature
                for cat in categories:
                    features_categorical_ohe.append("{}_{}".format(f, cat))  #adds a one hot encoding categorical column
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        one_hot_df = pd.get_dummies(self.df, columns=self.features_categorical) #applies pandas one hot encoding
        self.features_ohe = features_numerical + features_categorical_ohe #defines new feature set after ohe
        target_column = one_hot_df.pop('Target')  #remove target column
        one_hot_df.insert(len(one_hot_df.columns), 'Target', target_column)  #add target column to the end
        self.df = one_hot_df  #redefines dataframe by one hot encoding
    '''
    z_score_normalize: normalizes the data by applying the z score
    '''
    def z_score_normalize(self):
        for col in self.features_ohe:
            std = self.df[col].std() #computes standard deviation
            if std != 0:
                self.df[col] = (self.df[col] - self.df[col].mean()) / std #column is normalized by z score

    