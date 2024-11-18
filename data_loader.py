import pandas as pd
import statsmodels.api as sm
import numpy as np
from numpy import matrix
from sklearn.model_selection import train_test_split


## Creates superclass DataLoader.
class DataLoader:
    ## Constructs instance variables.
    ## @ param y_col: string with the name of the y variable
    ## @ param x_cols: list with the names of the x variables
    def __init__(self, y_col, x_cols):
        self._dataset = None
        self._y_col = y_col
        self._x_cols = x_cols
        self._org_dataset = None

    ## Abstract method to be specified in the subclasses.
    def readData(self):
        raise NotImplementedError
    
    ## Organizes the dataset to only have the columns the user needs.
    def _organize(self):
        columns = []
        columns.append(self._y_col)
        for elem in self._x_cols:
            columns.append(elem)
        self._org_dataset = self._dataset.loc[:, columns]

    ## @return matrix of x
    @property
    def x(self):
        if self._org_dataset is None:
            self._organize()
        X = matrix(self._org_dataset.iloc[:,1:].to_numpy())
        return X
    
    ## @return matrix of x with a constant (an intercept term)
    @property
    def _add_constant(self):
        X_const = np.insert(self.x, 0, 1, axis=1)
        return X_const
    

    ## Allows the user to access the matrix of x.
    @property
    def getX(self):
        X = self._add_constant
        return np.transpose(X)
    
    ## Allows the user to access the transposed matrix of x.
    @property
    def getXt(self):
        Xt = (self._add_constant)
        return Xt

    ## Allows the user to access the matrix of Y.
    @property
    def getY(self):
        if self._org_dataset is None:
            self._organize()
        Y = matrix(self._org_dataset.iloc[:,0].to_numpy())
        return Y
    
    ## Allows the user to access the transposed matrix of y.
    @property
    def getYt(self):
        Yt = self.getY.transpose()
        return Yt

## Enables the object to fetch data from statsmodel datasets.
class StatsModel(DataLoader):
    def __init__(self, y_col, x_cols):
        super().__init__(y_col, x_cols)

    ## @param data_name: name of dataset
    ## @param package_name: name of R dataset; if not specified, program searches through built-in datasets
    def readData(self, data_name, package_name=None):
        try:
            ## Handles specific R datasets by package name and dataset name.
            if package_name != None:
                self._dataset = sm.datasets.get_rdataset(data_name, package_name).data
            
            # Handles specific built-in datasets only by dataset name.
            if package_name == None:
                path = getattr(sm.datasets, data_name)
                self._dataset = path.load_pandas().data

        except Exception as e:
            print(f"Error reading data: {e}")

## Enables the object to fetch data from a CSV dataset.
class CSV(DataLoader):
    def __init__(self, y_col, x_cols):
        super().__init__(y_col, x_cols)
    
    ## @param file_path: URL to the designated file
    def readData(self, file_path):
        self._dataset = pd.read_csv(file_path)