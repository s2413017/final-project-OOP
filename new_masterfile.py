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
        # Parameters should be a matrix with the shape (p+1,1).
        self._params = None
        # Mu should be a list with the length of (N_train) or (N_test)
        self._mu = None

    def split(self):
        # Splits dataset into train and test.
        # X should be transposed into shape (N, p+1), Y into shape (N, 1).
        x_train, x_test, y_train, y_test = train_test_split(np.transpose(self._x), np.transpose(self._y), test_size=0.3, random_state=42)
        # Change x and y to the training matrices.
        # x_train has the shape (N_train, p+1), y_train has the shape (N_train, 1), x_test has the shape (N_test, p+1).
        self._x = np.transpose(x_train)
        # print("x", self._x.shape)
        self._y = np.transpose(y_train)
        # print("y", self._y.shape)
        # Change X to the testing matrix.
        self._x_test = np.transpose(x_test)
        # print("xtest", self._x_test.shape)
    
    def loglik(self):
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
        # print(self._x.shape)
        # print(self._y.shape)
        results = minimize(self.loglik, init_params, args =(self._x,self._y))
        # From a list create a numpy matrix with shape(p+1,1).
        self._params = np.array(results['x']).reshape(len(results['x']),1)
        # print(self._params.shape)
        # print(len(self._params)) #len should be p+1
        print("Optimization is finished!\nThe estimated beta values are:", results['x'])
        return results['x']
    
    def predict(self):
        self.fit()
        self.loglik(self._params,self._x_test,self._y)
        self._mu = np.asarray(self._mu).flatten()
        print(f"The estimated mean values are: {self._mu}")
        return self._mu
    
class NormalDistr(GeneralizedLinearModel):
    def __init__(self, x, y):
        super().__init__(x, y)

    def loglik(self, params, x, y):
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

    def loglik(self, params, x, y):
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

    def loglik(self, params, x, y):
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

# NORMAL TEST

# cars = StatsModel('income', ['education','prestige'])
# cars.readData("Duncan","carData")

# x1 = cars.getX
# y1 = cars.getY

# sample = NormalDistr(x1, y1)
# # sample.fit()
# # sample.split()
# sample.predict()

# POISSON TEST

# breaks = CSV("breaks", ["wool", "tension"])
# breaks.readData("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")

# x2 = breaks.getX
# y2 = breaks.getY

# sample2 = PoissonDistr(x2,y2)
# sample2.predict()

# BERNOULLI TEST

# sample3 = StatsModel("GRADE", ["GPA","TUCE","PSI"])
# sample3.readData("spector")

# x3 = sample3.getX
# y3 = sample3.getY

# sample3 = BernoulliDistr(x3,y3)
# sample3.predict()