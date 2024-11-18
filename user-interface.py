import argparse, textwrap

# Creates object called parser.
parser = argparse.ArgumentParser(prog="General Linear Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                                                       General Linear Model
                                     ------------------------------------------------------------
                                     The General Linear Model processes a chosen dataset,
                                     specifying predictor variables and the target variable.
                                     It trains the model using the dataset and tests it on
                                     a defined subset of samples to get the predicted outcome.
                                    '''),
        epilog=textwrap.dedent('''\
                                     ------------------------------------------------------------
                                     The following information was used to invoke the model:
                                            --model : family of distributions
                                            --source : how and where to access the dataset
                                            --dset : details about the specific dataset
                                            -p : name of the dataset package
                                            -x : selected predictors
                                            -y : selected dependent variable
                               
                                     Important note: Keep in mind the results we are comparing
                                     can differ between each other as there is an absolute
                                     tolerance parameter set at 1e-05.
                               
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

# Imports all classes from different files.
from GLMs_file import GeneralizedLinearModel, NormalDistr, PoissonDistr, BernoulliDistr
from data_loader import DataLoader, CSV, StatsModel
from test_file import test_dataset

# Connects to data_loader based on the input.
if args.source == "csv":
    example = CSV(args.predicted, args.predictors)
    example.readData(args.dset)
elif args.source == "statsmodel":
    example = StatsModel(args.predicted, args.predictors)
    if args.package == None:
        example.readData(args.dset)
    if args.package != None:
        example.readData(args.dset, args.package)

# Connects to GLMs_file based on the input.
if args.model == "normal":
    result_model = NormalDistr(example.getX, example.getY)
elif args.model == "poisson":
    result_model = PoissonDistr(example.getX, example.getY)
elif args.model == "bernoulli":
    result_model = BernoulliDistr(example.getX, example.getY)

# Calls the test function.
test_dataset(result_model, example)
