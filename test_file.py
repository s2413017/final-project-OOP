import pytest
from GLMs_file import GeneralizedLinearModel, NormalDistr, PoissonDistr, BernoulliDistr
from data_loader import DataLoader, CSV, StatsModel
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np

## Tests the dataset to see if the created model corresponds to the calculations in statsmodels.
## @param example: object of the superclass GeneralizedLinearModel
## @param source: object of the superclass DataLoader
## @return error statement if assertions are False
def test_dataset(example, source):
    if isinstance(example, NormalDistr):
        try:
            # Prepare normal_mod with test and train data.
            x_train, x_test, y_train, y_test = train_test_split(source.getXt, source.getYt, test_size=0.3, random_state=42)
            normal_mod_fitted = sm.GLM(y_train, x_train, family=sm.families.Gaussian()).fit()
            # Test Normal distribution.
            assert np.allclose(normal_mod_fitted.params, example.fit(), atol=1e-05, equal_nan=False)
            example = NormalDistr(source.getX, source.getY)
            assert np.allclose(normal_mod_fitted.predict(x_test), example.predict(), atol=1e-05, equal_nan=False)
        except Exception as e:
            print("Error: ", e)

    elif isinstance(example, PoissonDistr):
        try:
            # Prepare poisson_mod with test and train data.
            x_train, x_test, y_train, y_test = train_test_split(source.getXt, source.getYt, test_size=0.3, random_state=42)
            poisson_mod_fitted  = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
            # Test Poisson distribution.
            assert np.allclose(poisson_mod_fitted.params, example.fit(), atol=1e-05, equal_nan=False)
            example = PoissonDistr(source.getX, source.getY)
            assert np.allclose(poisson_mod_fitted.predict(x_test), example.predict(), atol=1e-05, equal_nan=False)
        except Exception as e:
            print("Error: ", e)

    elif isinstance(example, BernoulliDistr):
        try:
            # Prepare bernoulli_mod with test and train data.
            x_train, x_test, y_train, y_test = train_test_split(source.getXt, source.getYt, test_size=0.3, random_state=42)
            logit_mod_fitted = sm.GLM(y_train, x_train, family=sm.families.Binomial()).fit()
            # Test Bernoulli distribution.
            assert np.allclose(logit_mod_fitted.params, example.fit(), atol=1e-05, equal_nan=False)
            example = BernoulliDistr(source.getX, source.getY)
            assert np.allclose(logit_mod_fitted.predict(x_test), example.predict(), atol=1e-05, equal_nan=False)
        except Exception as e:
            print("Error: ", e)
    else:
        print("ERROR")