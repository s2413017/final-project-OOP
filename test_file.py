import pytest
from GLMs_file import GeneralizedLinearModel, NormalDistr, PoissonDistr, BernoulliDistr
from data_loader import DataLoader, CSV, StatsModel
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np

# NORMAL TEST

# cars = StatsModel('income', ['education','prestige'])
# cars.readData("Duncan", "carData")

# x_train, x_test, y_train, y_test = train_test_split(cars.getXt, cars.getYt, test_size=0.3, random_state=42)
# normal_mod = sm.GLM(y_train, x_train, family=sm.families.Gaussian())
# fitted_mod = normal_mod.fit()
# print(fitted_mod.params)
# print(fitted_mod.predict(x_test))

# POISSON TEST

# breaks = CSV("breaks", ["wool", "tension"])
# breaks.readData("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")

# x_train, x_test, y_train, y_test = train_test_split(breaks.getXt, breaks.getYt, test_size=0.3, random_state=42)
# poisson_mod = sm.GLM(y_train, x_train, family=sm.families.Poisson())
# fitted_mod = poisson_mod.fit()
# print(fitted_mod.params) # numpy ndarray
# print(fitted_mod.predict(x_test))

# BERNOULLI TEST

# random = StatsModel("GRADE", ["GPA","TUCE","PSI"])
# random.readData("spector")

# x_train, x_test, y_train, y_test = train_test_split(random.getXt, random.getYt, test_size=0.3, random_state=42)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# logit_mod = sm.GLM(y_train, x_train, family=sm.families.Binomial())
# fitted_mod = logit_mod.fit()
# print(fitted_mod.params) # numpy ndarray
# print(fitted_mod.predict(x_test))

#------------------------------------------------------------------------------------------------------------
# norm_example = StatsModel('income', ['education','prestige'])
# norm_example.readData("Duncan","carData")
# norm_example_ = NormalDistr(norm_example.getX, norm_example.getY)

# poiss_example = CSV("breaks", ["wool", "tension"])
# poiss_example.readData("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")
# poiss_example_ = PoissonDistr(poiss_example.getX, poiss_example.getY)

# bern_example = StatsModel("GRADE", ["GPA","TUCE","PSI"])
# bern_example.readData("spector")
# bern_example_ = BernoulliDistr(bern_example.getX, bern_example.getY)

def test_dataset(example, source):
    if isinstance(example, NormalDistr):
        try:
            ## Prepare normal_mod with test and train data.
            x_train, x_test, y_train, y_test = train_test_split(source.getXt, source.getYt, test_size=0.3, random_state=42)
            normal_mod_fitted = sm.GLM(y_train, x_train, family=sm.families.Gaussian()).fit()
            # ## Test Normal distribution.
            assert np.allclose(normal_mod_fitted.params, example.fit(), atol=1e-05, equal_nan=False)
            example = NormalDistr(source.getX, source.getY)
            assert np.allclose(normal_mod_fitted.predict(x_test), example.predict(), atol=1e-05, equal_nan=False)
        except Exception as e:
            print("Error: ", e)

    elif isinstance(example, PoissonDistr):
        try:
            ## Prepare poisson_mod with test and train data.
            x_train, x_test, y_train, y_test = train_test_split(source.getXt, source.getYt, test_size=0.3, random_state=42)
            poisson_mod_fitted  = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
            ## Test Poisson distribution.
            assert np.allclose(poisson_mod_fitted.params, example.fit(), atol=1e-05, equal_nan=False)
            example = PoissonDistr(source.getX, source.getY)
            assert np.allclose(poisson_mod_fitted.predict(x_test), example.predict(), atol=1e-05, equal_nan=False)
        except Exception as e:
            print("Error: ", e)

    elif isinstance(example, BernoulliDistr):
        try:
            ## Prepare bernoulli_mod with test and train data.
            x_train, x_test, y_train, y_test = train_test_split(source.getXt, source.getYt, test_size=0.3, random_state=42)
            # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            logit_mod_fitted = sm.GLM(y_train, x_train, family=sm.families.Binomial()).fit()
            ## Test Bernoulli distribution.
            assert np.allclose(logit_mod_fitted.params, example.fit(), atol=1e-05, equal_nan=False)
            example = BernoulliDistr(source.getX, source.getY)
            assert np.allclose(logit_mod_fitted.predict(x_test), example.predict(), atol=1e-05, equal_nan=False)
        except Exception as e:
            print("Error: ", e)
    else:
        print("ERROR")

# test_dataset(norm_example_, norm_example)
# test_dataset(poiss_example_, poiss_example)
# test_dataset(bern_example_, bern_example)
