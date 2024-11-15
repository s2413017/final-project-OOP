import numpy as np
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize

## Generalized Linear Model take
#
class GeneralizedLinearModel:
    ## Initialize object with the following instance variables.
    # @param x matrix of covariates
    # @param y dependent variable
    def __init__(self, x, y):
        ## Constructor
        self._x = x
        self._y = y
        self._params = None
        self._eta = None
        self._mu = None
        self._loglik = None        

    def fit(self):
        initial_params = np.repeat(0.1, self._x.shape[1])
        result = minimize(self.negllik, initial_params)
        self._params = result["x"]
        return print(f"The estimated beta values are: {self._params}")
    
    def negllik(self, params):
        raise NotImplementedError

    def loglik(self):
        raise NotImplementedError
    
    def predict(self):
        try: 
            self.fit()
            return print(f"The estimated mean values are: {self._mu}")
        except:
            print("Method failed.")

class NormalDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)
    
    def negllik(self, params):
        self._params = params
        self._eta = np.dot(self._x, self._params)
        self._mu = self._eta
        return -self.loglik()
    
    ## Computes the log likelihood.
    #
    def loglik(self):
        # Takes variable y and the link function to get the log likelihood.
        self._loglik = np.sum(norm.logpdf(self._y,self._mu))
        return self._loglik
    
    
class PoissonDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)
    
    def negllik(self, params):
        self._params = params
        self._eta = np.dot(self._x, self._params)
        self._mu = np.exp(self._eta)
        return -self.loglik()
    
    ## Computes the log likelihood.
    #
    def loglik(self):
        # Takes variable y and the link function to get the log likelihood.
        self._loglik = np.sum(poisson.logpmf(self._y,self._mu))
        return self._loglik

class BernoulliDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)
        
    def negllik(self, params):
        self._params = params
        self._eta = np.dot(self._x, self._params)
        self._mu = np.exp(self._eta)/(1+np.exp(self._eta))
        return -self.loglik()
    
    ## Computes the log likelihood.
    #
    def loglik(self):
        # Takes variable y and the link function to get the log likelihood.
        self._loglik = np.sum(bernoulli.logpmf(self._y,self._mu))
        return self._loglik