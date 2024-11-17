import pytest
from new_masterfile import GeneralizedLinearModel, NormalDistr, PoissonDistr, BernoulliDistr
from data_loader import DataLoader, CSV, StatsModel
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np

# NORMAL TEST

# cars = StatsModel('income', ['education','prestige'])
# cars.readData("Duncan", "carData")

# x1 = cars.getXt
# y1 = cars.getYt
# print(x1.shape)
# print(y1.shape)

# x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42)
# # print(x_train.shape) # i receive (31,3)
# # print(y_train.shape) # i receive (31,1)
# # print(x_test.shape) # i receive (14,3)
# # print(y_test.shape) # i receive (14,1)

# # (45,3) for x and (45,1) for y
# normal_mod = sm.GLM(y_train, x_train, family=sm.families.Gaussian())
# fitted_mod = normal_mod.fit()
# print(fitted_mod.params)
# print(fitted_mod.predict(x_test))

# POISSON TEST

# breaks = CSV("breaks", ["wool", "tension"])
# breaks.readData("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")

# x2 = breaks.getXt
# y2 = breaks.getYt
# x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=42)
# poisson_mod = sm.GLM(y_train, x_train, family=sm.families.Poisson())
# fitted_mod = poisson_mod.fit()
# print(fitted_mod.params) # numpy ndarray
# print(fitted_mod.predict(x_test))

# BERNOULLI TEST

# random = StatsModel("GRADE", ["GPA","TUCE","PSI"])
# random.readData("spector")

# x3 = random.getXt
# y3 = random.getYt

# x_train, x_test, y_train, y_test = train_test_split(x3, y3, test_size=0.3, random_state=42)
# logit_mod = sm.GLM(y_train, x_train, family=sm.families.Binomial())
# fitted_mod = logit_mod.fit()
# print(fitted_mod.params) # numpy ndarray
# print(fitted_mod.predict(x_test))

#------------------------------------------------------------------------------------------------------------
# dataset = PoissonDistr(x1,y1)

# def test_dataset(dataset):
#     if dataset is NormalDistr:
#         ## Test Normal distribution.
#         assert normal_mod.fit().params == dataset.fit()
#         assert normal_mod.predict(x_test)

#     if dataset is PoissonDistr:
#         ## Test Poisson distribution.
#         assert poisson_mod.fit().params == dataset.fit()

#     if dataset is BernoulliDistr:
#         ## Test Bernoulli distribution.
#         assert logit_mod.fit().params == dataset.fit()

# test_dataset(dataset)
