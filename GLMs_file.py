import numpy as np
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from data_loader import DataLoader, CSV, StatsModel

class GeneralizedLinearModel:
    def __init__(self, x, y):
        # X should have the structure (p+1, N)
        self._x = x
        # Y should have the structure (1, N)
        self._y = y
        # X test should have the structure (p+1,N_test)
        self._x_test = None
        # Parameters should be a matrix with the shape (p+1,1)
        self._params = None
        # Mu should be a matrix of shape (1, N_train) or (1, N_test)
        self._mu = None

    def split(self):
        # Splits dataset into train and test.
        # X should be transposed into shape (N, p+1), Y into shape (N, 1).
        x_train, x_test, y_train, y_test = train_test_split(np.transpose(self._x), np.transpose(self._y), test_size=0.3, random_state=42)
        # Change x and y to the training matrices.
        # x_train has the shape (N_train, p+1), y_train has the shape (N_train, 1), x_test has the shape (N_test, p+1).
        self._x = np.transpose(x_train)
        self._y = np.transpose(y_train)
        # Update the instance variable with the x_test matrix.
        self._x_test = np.transpose(x_test)
    
    def negloglik(self):
        raise NotImplementedError
    
    def fit(self):
        self.split()
        # Attributing the number of columns to num_params.
        num_params = self._x.shape[0]
        # Inital parameters have values 0.1 in all columns.
        # Should have the shape (p+1,).
        init_params = np.repeat(0.1, num_params)
        # print(init_params.shape)
        # Should have shape x=(p+1,N_train), y=(1,N_train)
        results = minimize(self.negloglik, init_params, args =(self._x,self._y))
        # From a list create a numpy array with shape(p+1,1).
        self._params = np.array(results['x']).reshape(len(results['x']),1)
        print("Optimization is finished!\nThe estimated beta values are:", results['x'])
        return results['x']
    
    def predict(self):
        self.fit()
        self.negloglik(self._params,self._x_test,self._y)
        self._mu = np.asarray(self._mu).flatten()
        print(f"The estimated mean values are: {self._mu}")
        return self._mu
    
class NormalDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def negloglik(self, params, x, y):
        # Calculates eta by multiplying x transposed with beta parameters.
        # Should be (N, p+1) x (p+1, 1)
        new_x = np.transpose(x)
        eta = np.matmul(new_x, params)
        # Calculates mu using the link function.
        # Should be (N, 1)
        self._mu = eta
        # Computes the log likelihood of the function.
        llik = np.sum(norm.logpdf(y, self._mu))
        # Returns the negative log likelihood.
        return -llik

class PoissonDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def negloglik(self, params, x, y):
        # Calculates eta by multiplying x transposed with beta parameters.
        # Should be (N, p+1) x (p+1, 1)
        new_x = np.transpose(x)
        eta = np.matmul(new_x, params)
        # Calculates mu using the link function.
        # Should be (N, 1)
        self._mu = np.exp(eta)
        # Computes the log likelihood of the function.
        llik = np.sum(poisson.logpmf(y, self._mu))
        # Returns the negative log likelihood.
        return -llik
    
class BernoulliDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def negloglik(self, params, x, y):
        # Calculates eta by multiplying x transposed with beta parameters.
        # Should be (N, p+1) x (p+1, 1)
        new_x = np.transpose(x)
        eta = np.matmul(new_x, params)
        # Calculates mu using the link function.
        # Should be (N, 1)
        self._mu = np.exp(eta)/(1+np.exp(eta))
        # Computes the log likelihood of the function.
        llik = np.sum(bernoulli.logpmf(y, self._mu))
        # Returns the negative log likelihood.
        return -llik

# # NORMAL TEST

# norm_example = StatsModel('income', ['education','prestige'])
# norm_example.readData("Duncan","carData")
# norm_example = NormalDistr(norm_example.getX, norm_example.getY)
# norm_example.predict()

# # POISSON TEST

# poiss_example = CSV("breaks", ["wool", "tension"])
# poiss_example.readData("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")
# poiss_example = PoissonDistr(poiss_example.getX, poiss_example.getY)
# poiss_example.predict()

# # BERNOULLI TEST

# bern_example = StatsModel("GRADE", ["GPA","TUCE","PSI"])
# bern_example.readData("spector")
# bern_example = BernoulliDistr(bern_example.getX, bern_example.getY)
# bern_example.predict()