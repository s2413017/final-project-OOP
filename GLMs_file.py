import numpy as np
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

## GeneralizedLinearModel takes input variables x and y to fit beta parameters and predict the μ.
## @param x: matrix with predictor variables of shape (p+1, N)
## @param y: matrix with dependent variables of shape (1, N)
class GeneralizedLinearModel:
    def __init__(self, x, y):
        self._x = x # shape (p+1, N) or (p+1, N_train)
        self._y = y # shape (1, N) or (1, N_train)
        self._x_test = None # shape (p+1, N_test)
        self._params = None # shape (p+1, 1) or (p+1,)
        self._mu = None # shape (1, N_train) or (1, N_test)

    ## Splits dataset into train and test.
    def split(self):
        # When splitting the dataset, we first transpose self._x and self._y to have shapes (N, p+1) and (N, 1).
        x_train, x_test, y_train, y_test = train_test_split(np.transpose(self._x), np.transpose(self._y), test_size=0.3, random_state=42)
        # Update instance variables to match the train and test matrices.
        self._x = np.transpose(x_train) # shape (N_train, p+1) -> (p+1, N_train)
        self._y = np.transpose(y_train) # shape (N_train, 1) -> (1, N_train)
        self._x_test = np.transpose(x_test) # shape (N_test, p+1) -> (p+1, N_test)
    
    ## Negloglik should be implemented in the subclasses (abstract method).
    ## @param params: numpy array of beta parameters estimated in fit method
    ## @param x: matrix of predictor variables
    ## @param y: matrix of dependent variables
    ## @return the negative log likelihood
    def negloglik(self, params, x, y):
        raise NotImplementedError
    
    ## Minimizes the negative log likelihood to estimate the unknown beta parameters.
    ## @return list of estimated beta parameters
    def fit(self):
        # Invokes split method to separate the data.
        self.split()
        num_params = self._x.shape[0] # number of columns in self._x
        init_params = np.repeat(0.1, num_params) # 0.1 in all columns with the shape (p+1,)
        # Minimize the negative log likelihood by inputting self._x and self._y and providing initial parameters.
        results = minimize(self.negloglik, init_params, args =(self._x,self._y))
        self._params = np.array(results['x']).reshape(len(results['x']),1) # from a list create a numpy array with shape(p+1,1)
        print("\nOptimization is finished!\nThe estimated beta values are:", results['x'])
        return results['x']
    
    ## Predicts the μ based on beta parameters, train matrices and the testing predictors.
    ## @return the estimated μ parameters
    def predict(self):
        # Invokes fit method to estimate beta parameters.
        self.fit()
        # Invokes negloglik method to calculate μ (self._mu).
        self.negloglik(self._params,self._x_test,self._y)
        self._mu = np.asarray(self._mu).flatten()
        print(f"The estimated mean values are: {self._mu}")
        return self._mu

## Subclass specifies the negative log likelihood for a Normal distribution.
class NormalDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def negloglik(self, params, x, y):
        # Calculates eta by multiplying x transposed with beta parameters.
        new_x = np.transpose(x) # shape (p+1, N) -> (N, p+1)
        eta = np.matmul(new_x, params) # shape of params (p+1, 1)
        # Calculates mu using the link function.
        self._mu = eta # shape (N, 1)
        # Computes the log likelihood of the function.
        llik = np.sum(norm.logpdf(y, self._mu))
        return -llik
    
## Subclass specifies the negative log likelihood for a Poisson distribution.
class PoissonDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def negloglik(self, params, x, y):
        # Calculates eta by multiplying x transposed with beta parameters.
        new_x = np.transpose(x) # shape (p+1, N) -> (N, p+1)
        eta = np.matmul(new_x, params) # shape of params (p+1, 1)
        # Calculates mu using the link function.
        self._mu = np.exp(eta) # shape (N, 1)
        # Computes the log likelihood of the function.
        llik = np.sum(poisson.logpmf(y, self._mu))
        # Returns the negative log likelihood.
        return -llik
    
## Subclass specifies the negative log likelihood for a Bernoulli distribution.
class BernoulliDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def negloglik(self, params, x, y):
        # Calculates eta by multiplying x transposed with beta parameters.
        new_x = np.transpose(x) # shape (p+1, N) -> (N, p+1)
        eta = np.matmul(new_x, params) # shape of params (p+1, 1)
        # Calculates mu using the link function.
        self._mu = np.exp(eta)/(1+np.exp(eta)) # shape (N, 1)
        # Computes the log likelihood of the function.
        llik = np.sum(bernoulli.logpmf(y, self._mu))
        # Returns the negative log likelihood.
        return -llik