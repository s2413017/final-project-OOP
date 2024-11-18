from data_loader import DataLoader, CSV, StatsModel
from GLMs_file import GeneralizedLinearModel, NormalDistr, PoissonDistr, BernoulliDistr


norm_example = StatsModel('income', ['education','prestige'])
norm_example.readData("Duncan","carData")
norm_example_ = NormalDistr(norm_example.getX, norm_example.getY)
norm_example_.predict()

poiss_example = CSV("breaks", ["wool", "tension"])
poiss_example.readData("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")
poiss_example_ = PoissonDistr(poiss_example.getX, poiss_example.getY)
poiss_example_.predict()

bern_example = StatsModel("GRADE", ["GPA","TUCE","PSI"])
bern_example.readData("spector")
bern_example_ = BernoulliDistr(bern_example.getX, bern_example.getY)
bern_example_.predict()