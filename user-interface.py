import argparse, textwrap

# Creates object called parser.
parser = argparse.ArgumentParser(prog="General Linear Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                                                       General Linear Model
                                     ------------------------------------------------------------
                                     The General Linear Model takes a selected dataset, for which 
                                     we specify which are the predicting parameters and which is
                                     the parameter to be predicted. It uses the values in the
                                     dataset to train the model and then tests it on a certain
                                     number of samples.
                                    '''),
        epilog=textwrap.dedent('''\
                                     ------------------------------------------------------------
                                     The following information was used to invoke the model:
                                            --model : defined the family of distributions
                                            --source : defined how and where to access the dataset
                                            --dset : gave details about the specific dataset
                                            -p : invoked name of the dataset package
                                            -x : selected predictors
                                            -y : selected the variable to predict
                               
                                     ''')
                    )
parser.print_help()

# Adds one argument with default value and specifying datatype.
parser.add_argument("--model", default="normal", choices=["normal","poisson","bernoulli"], type=str,
                    help = "You can only choose the following models: normal, poisson, bernoulli")
parser.add_argument("--source", default="csv", choices=["csv","statsmodel"], type=str,
                    help = "You can only choose the following models: csv, statsmodel")
parser.add_argument("--dset", default="", type=str,
                    help = "If you chose CSV: Specify the file path, either by URL or path to local folder. If you chose StatsModel: Specify name of the dataset.")
parser.add_argument("-p","--package", default=None, type=str, nargs="?",
                    help = "If you chose CSV: Skip. If you chose StatsModel and it is not a part of the R dataset package write the name of the package.")
parser.add_argument("-x","--predictors", default="", type=str, nargs="+",
                    help = "Write all the x variables used for the model.")
parser.add_argument("-y","--predicted", default="", type=str,
                    help = "Write the name of the y variable you want to predict.")

# Passes all arguments into an object that retrive arguments with property-decorator style.
args = parser.parse_args()


from GLMs_file import GeneralizedLinearModel, NormalDistr, PoissonDistr, BernoulliDistr
from data_loader import DataLoader, CSV, StatsModel
from test_file import test_dataset

if args.source == "csv":
    example = CSV(args.predicted, args.predictors)
    example.readData(args.dset)
elif args.source == "statsmodel":
    example = StatsModel(args.predicted, args.predictors)
    if args.package == None:
        example.readData(args.dset)
    if args.package != None:
        example.readData(args.dset, args.package)

if args.model == "normal":
    result_model = NormalDistr(example.getX, example.getY)
elif args.model == "poisson":
    result_model = PoissonDistr(example.getX, example.getY)
elif args.model == "bernoulli":
    result_model = BernoulliDistr(example.getX, example.getY)

test_dataset(result_model, example)

# test for normal:
## python user-interface.py --model normal --source statsmodel --dset Duncan -p carData -x education prestige -y income

# test for poisson:
## python user-interface.py --model poisson --source csv --dset https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv -x wool tension -y breaks

# test for bernoulli:
## python user-interface.py --model bernoulli --source statsmodel --dset spector -x GPA TUCE PSI -y GRADE
