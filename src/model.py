import argparse
import json
import yaml

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.ensemble import AdaBoostClassifier

from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

from sklearn.decomposition import PCA


from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler


#from sklearn.metrics import accuracy_score

# Import keras modules
from keras.models import Sequential
from keras.layers import Dense


from helper_functions import load_data, multilabel_print_scores, multiclass_print_scores, binary_print_scores, remove_feature, remove_outliers_IsolationForest, MultiLabelClassifier
from helper_functions import check_required_args, validate_n_components, make_nn_model, create_powerset

from pipelines import ID_Remover, make_ovr_pipe, make_br_pipe, make_cc_pipe, make_lp_pipe, make_lp_ex_pipe, make_lp_pca_pipe, make_lp_GSCV_pipe, make_lp_brfc_pipe, make_multiclass_pipe, make_binary_pipe

# GLOBAL CONFIG VARIABLES
CONTAMINATION = 0.08

PIPE_DICT = {
    'ovr': make_ovr_pipe,
    'br': make_br_pipe,
    'cc': make_cc_pipe,
    'lp': make_lp_pipe,
    'lp_ex': make_lp_ex_pipe,
    'lp_pca': make_lp_pca_pipe,
    'lp_gscv': make_lp_GSCV_pipe,
    'lp_brfc': make_lp_brfc_pipe,
    'multiclass': make_multiclass_pipe
}

#### Functions ####

def parse_args():
    """ Function to parse arguments from commnad line

    Positional Argument
    -------------------
    Mode: str
        'grid', 'pipe', 'pipe-mc', 'nn'
    
    Optional Arguments
    ------------------
    n_components: float [0, 1]
        Potentially list of floats. e.g. [0.92, 0.95, 0.99] Specify variance to determine no. of PCA components to be used
        Needs to be specified when mode = 'grid'
    
    n_estimators: int
        Potentially list of floats. e.g. [50, 100] Determine no. estimators in the base estimators of pipeline
        Needs to be specified when mode = 'grid'
    
    pipeline: str
        Specify which pipeline to run when mode = 'pipe'
    
    Returns
    -------
    args: list
        List of arguments parsed from command line
    
    """

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(
        prog = "Model Training",
        description = "Train a pipeline on Enzyme Multi Label Dataset"
    )

    # Adding positional argument
    parser.add_argument('mode', type=str, help='grid->GridSearchCV, pipe->Run multi-label pipe, pipe-mc-> Run multi-class pipe, nn->Neural Network')

    # Adding optional argument for pca n_compnents in the case where mode = 'grid'

    parser.add_argument('--n_components', '-c', nargs='+', type=validate_n_components, help='List of PCA n_component values for GridSearchCV')

    # Adding optional argument for classifiers n_estimators

    parser.add_argument('--n_estimators', '-e', nargs='+', type=int, help='List of n_estimators values for GridSearchCV')

    # Adding optional argument for choosing pipeline
    parser.add_argument('--pipeline', '-p', type=str, help='Pipeline Selection')
    
    # Parse the command_line arguments
    args = parser.parse_args()

    # Check both n_components and n_estimators are present if mode is 'grid'
    check_required_args(args)

    return args

def load_config_yaml(config_name):
    """ Load config file
    
    Parameters
    ----------
    config_name: str
        Name of config file
    
    Returns
    -------
    dict
        Key-value pairs defined in config
    """

    with open(config_name) as file:
        config = yaml.safe_load(file)
    
    return config

def run_pipe(X_train, y_train, X_test, y_test):
    """ Run specific multilabel pipe when mode = "pipe"
    
    SELECTED_PIPE is parsed from command line and used with PIPE_DICT to call the specified pipe.
    Call multilabel_print_scores to compute accuracy and roc auc scores.

    Parameters
    ----------
    X_train: DataFrame
        Training features
    
    y_train: DataFrame
        Training labels i.e. 2 columns EC1 and EC2
    
    X_test: DataFrame
        Test features

    y_test: DataFrame
        Test labels i.e. 2 columns EC1 and EC2
    
    """

    pipe = PIPE_DICT[SELECTED_PIPE](numerical_features, categorical_features)
    pipe.fit(X_train, y_train)
 
    multilabel_print_scores(pipe, X_test, y_test, SELECTED_PIPE)


def run_mc_pipe(X_train, y_train_multiclass, X_test, y_test_multiclass):
    """ Run multiclass pipe when mode = "pipe-mc"
    
    Hardcoded to call make_multiclass_pipe
    Call multiclass_print_scores to compute accuracy and roc auc scores.

    Also call make_binary_pipe on EC1 and EC2 separately and
    call binary_print_scores to compute accuracy and roc auc scores on the binary classification

    Parameters
    ----------
    X_train: DataFrame
        Training features
    
    y_train: DataFrame
        Training labels i.e. 1 column of integer labels
    
    X_test: DataFrame
        Test features

    y_test: DataFrame
        Test labels i.e. 1 column of integer labels
    
    """

    X_train.columns.tolist()

    pipe = make_multiclass_pipe(numerical_features, categorical_features)

    # pipe.fit(X_train, y_train_multiclass, ada__sample_weight=sample_weights)
    pipe.fit(X_train, y_train_multiclass)
 
    multiclass_print_scores(pipe, X_test, y_test_multiclass, SELECTED_PIPE)

    pipe_EC1 = make_binary_pipe()
    pipe_EC1.fit(X_train, y_train['EC1'])

    binary_print_scores(pipe_EC1, X_test, y_test['EC1'], 'EC1')

    pipe_EC2 = make_binary_pipe()

    # Insert class_weights and see what happens
    pipe_EC2.fit(X_train, y_train['EC2'])

    binary_print_scores(pipe_EC2, X_test, y_test['EC2'], 'EC2')

def run_gridsearch():
    """Run gridsearch on pipe instance from calling make_lp_GSCV_pipe()"""

    # Define parameter grid to search.
    # N_COMPONENTS_PARAM and N_ESTIMATORS_PARAM parsed from command line

    param_grid = {
        'pca__n_components': N_COMPONENTS_PARAM,
        'lp_rf__classifier__n_estimators': N_ESTIMATORS_PARAM
    }

    # Create the GridSearchCV object

    try:
        GSCV_pipe = make_lp_GSCV_pipe()
        grid_search = GridSearchCV(
                            estimator=GSCV_pipe, 
                            param_grid = param_grid,
                            cv=5, #default is StratifiedKFold
                            scoring='accuracy')
        
        # Fit training data. Grid Search object will perform grid search and 
        # then fit the estimator with the best params

        grid_search.fit(X_train, y_train)
    except:
        raise RuntimeError("Check list of numbers for -c -e")

    # Print best parameters and score

    print("Best parameters for lp_gscv: ", grid_search.best_params_)
    print("Best score for lp_gscv: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    #best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    print(f"The accuracy score of best model from gridsearchcv is {score}")

def run_nn():
    """ Create keras fully connect model using make_nn_model, fit, predict and compute
    accuracy and roc auc scores
    """
    n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
    model = make_nn_model(n_inputs, n_outputs)
    model.fit(X_train, y_train, verbose=1, epochs=40)

    # Predictions are in form of probabilities
    y_prob = model.predict(X_test)

    # Rounding to produce the labels
    y_pred = y_prob.round()
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"The accuracy score of neural network is {accuracy}")
    print(f"The roc auc score of neural network is {roc_auc}")

def main():

    if MODE == 'pipe':

        run_pipe(X_train, y_train, X_test, y_test)
    
    if MODE == 'pipe_mc':

        run_mc_pipe(X_train, y_train_multiclass, X_test, y_test_multiclass)

    if MODE == "grid":

        run_gridsearch()

    if MODE == "nn":
        
        run_nn()


if __name__ == "__main__":
    
    # Parse arugments from command lines and assign to variables
    args = parse_args()
    
    MODE = args.mode
    N_COMPONENTS_PARAM = args.n_components
    N_ESTIMATORS_PARAM = args.n_estimators
    SELECTED_PIPE = args.pipeline

    # Read config.py ot config.yaml
    try:
        with open('../config.py', "r") as inp:
            config = json.load(inp)
    
    except:
        raise FileNotFoundError("config.py is not found")
    
    try:
        config_yaml = load_config_yaml("../config.yaml")
    except:
        raise FileNotFoundError("config.yaml is not found")

    
    # Load and split datasets. Assign to variables
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = load_data(**config['load_data'])

    # Create multiclass version of labels 
    y_train_multiclass, y_test_multiclass = create_powerset(y_train, y_test)

    # Compute class and sample weights in case weights need to be passed to classifiers to account
    # for imbalanced datasets

    # Calculate class weights
    all_classes = np.unique(y_train_multiclass)
    class_weights = compute_class_weight("balanced", classes=all_classes, y=y_train_multiclass)

    # Calculate sample weights
    sample_weights = compute_sample_weight(class_weight=dict(zip(all_classes, class_weights)), y=y_train_multiclass)
    

    main()

