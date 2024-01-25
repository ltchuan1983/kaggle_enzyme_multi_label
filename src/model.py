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

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import keras_tuner

import h2o
from h2o.automl import H2OAutoML


from helper_functions import load_data, split_data, save_data_to_db, datagen_from_db, multilabel_print_scores, multiclass_print_scores, binary_print_scores, remove_feature, remove_outliers_IsolationForest, MultiLabelClassifier
from helper_functions import check_required_args, validate_n_components, make_nn_model, make_tunable_nn_model, create_powerset

from pipelines import ID_Remover, make_ovr_pipe, make_br_pipe, make_cc_pipe, make_lp_pipe, make_lp_ex_pipe, make_lp_pca_pipe, make_lp_GSCV_pipe, make_lp_brfc_pipe, make_multiclass_pipe, make_binary_pipe, make_simplegrid_pipe, make_stacking_pipe

# Default batch size unless specified otherwise by config
BATCH_SIZE = 32 

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
    
    y_train_multiclass: DataFrame
        Training labels i.e. 1 column of integer labels
    
    X_test: DataFrame
        Test features

    y_test_multiclass: DataFrame
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
        'rf__n_estimators': N_ESTIMATORS_PARAM
        #'lp_rf__classifier__n_estimators': N_ESTIMATORS_PARAM
    }

    # Create the GridSearchCV object

    try:
        simplegrid_pipe = make_simplegrid_pipe(numerical_features, categorical_features)
        grid_search = GridSearchCV(
                            estimator=simplegrid_pipe, 
                            param_grid = param_grid,
                            cv=5, #default is StratifiedKFold
                            scoring='accuracy')
        
        # Fit training data. Grid Search object will perform grid search and 
        # then fit the estimator with the best params

        grid_search.fit(X_train, y_train_multiclass)
    except:
        raise RuntimeError("Check list of numbers for -c -e")

    # Print best parameters and score

    print("Best parameters for lp_gscv: ", grid_search.best_params_)
    print("Best score for lp_gscv: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    #best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    score = accuracy_score(y_test_multiclass, y_pred)
    
    print(f"The accuracy score of best model from gridsearchcv is {score}")

def run_bayes():
    print("Running Bayesian Optimization")

    search_space = {
        'pca__n_components': Real(0.92, 0.98),
        'rf__n_estimators': Integer(50, 120)
    }

    bayes_pipe = make_simplegrid_pipe(numerical_features, categorical_features)
    lp_bayes_search = BayesSearchCV(bayes_pipe, search_space, n_iter=32,
                                    scoring="accuracy", n_jobs=-1, cv=5)
    # n_jobs=-1 -> number of jobs set to number of CPU cores

    def on_step(optim_result):
        """Callback to print score after each iteration"""
        print(f"Iteration {optim_result.func_vals.shape[0]}, Best Score: {-optim_result.fun.max()}")
        # score = -optim_result.fun
        # print("best score: %s" % score)
        # if score >= 0.98:
        #     print('Interrupting!')
        #     return True
    
    np.int = int # Because np.int deprecated

    lp_bayes_search.fit(X_train, y_train_multiclass, callback=on_step) #on_step prints score after each iteration
    print(lp_bayes_search.best_params_)
    print(lp_bayes_search.best_score_)

def run_nn():
    """ Create keras fully connect model using make_nn_model, fit, predict and compute
    accuracy and roc auc scores
    """
    STEPS_PER_EPOCH = len(X_train)/BATCH_SIZE
    # n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
    n_inputs, n_outputs = X_train.shape[1], 4
    print(n_inputs)
    model = make_nn_model(n_inputs, n_outputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train_multiclass, verbose=1, batch_size=BATCH_SIZE, epochs=10)

    # Each epoch will start a new datagen instance
    #model.fit(datagen_from_db(32), verbose=1, batch_size=BATCH_SIZE, epochs=40, steps_per_epoch=STEPS_PER_EPOCH)

    # Predictions are in form of probabilities
    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict(X_test_scaled)

    # Rounding to produce the labels
    #y_pred = y_prob.round()
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_test_multiclass, y_pred)
    roc_auc = roc_auc_score(y_test_multiclass, y_prob, multi_class='ovr')
    
    print(f"The accuracy score of neural network is {accuracy}")
    print(f"The roc auc score of neural network is {roc_auc}")

def run_tunable_nn():
    """ Create keras fully connect model using make_nn_model, fit, predict and compute
    accuracy and roc auc scores

    Hyperparameter tuning with Keras Tuner
    """
    
    n_inputs, n_outputs = X_train.shape[1], 4
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Can try RandomSearch or Hyerband too
    tuner = keras_tuner.BayesianOptimization(
            make_tunable_nn_model,
            seed=808,
            objective='val_loss',
            max_trials=15,
            directory='../models',
            project_name='keras_tuner_Bayes'
    )

    tuner.search(X_train_scaled, y_train_multiclass, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test_multiclass))

    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print('Best hyperparameters')
    print(best_hyperparameters.values)

    # Build model from best hyperparameters
    #best_model = tuner.hypermodel.build(best_hyperparameters)
    #best_model.fit(X_train_scaled, y_train_multiclass, verbose=1, batch_size=BATCH_SIZE, epochs=10)


    best_model = tuner.get_best_models()[0]
    y_prob = best_model.predict(X_test_scaled)

    # Rounding to produce the labels
    #y_pred = y_prob.round()
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_test_multiclass, y_pred)
    roc_auc = roc_auc_score(y_test_multiclass, y_prob, multi_class='ovr')
    
    print(f"The accuracy score of neural network is {accuracy}")
    print(f"The roc auc score of neural network is {roc_auc}")

def run_autoML():
    """ Use H2O AutoML to obtain best model.
    Need to convert pandas dataframe to H2O Frame for training/predicting
    Then need to convert back to pandas dataframe or lists for evaluation
    
    Tunable parameters: max_models, balance_classes
    """
    h2o.init()

    train_h2o_df = h2o.H2OFrame(train_df)
    test_h2o_df = h2o.H2OFrame(X_test)

    train_h2o_df['Powerset_Label'] = train_h2o_df['Powerset_Label'].asfactor()

    #train_h2o_df.describe(chunk_summary=True)

    aml = H2OAutoML(max_models=25, balance_classes=True, seed=1)
    aml.train(training_frame=train_h2o_df, y='Powerset_Label')

    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))

    best_model = aml.get_best_model()

    # Save model

    h2o.save_model(model=best_model, path='../models/automl/best_model', force=True)

    loaded_model = h2o.load_model(path='../models/automl/best_model/DeepLearning_grid_1_AutoML_1_20240123_205139_model_1')

    loaded_model_output = loaded_model.predict(test_h2o_df).as_data_frame()
    y_pred = loaded_model_output['predict']
    y_prob = loaded_model_output[['p0', 'p1', 'p2', 'p3']].apply(lambda row: row.tolist(), axis=1).tolist()
    #print(y_prob)

    accuracy = accuracy_score(y_test_multiclass, y_pred)
    roc_auc = roc_auc_score(y_test_multiclass, y_prob, multi_class='ovr')
    
    print(f"The accuracy score of AutoML best model is {accuracy}")
    print(f"The roc auc score of AutoML best model is {roc_auc}")


def run_stacking_pipe(X_train, y_train_multiclass, X_test, y_test_multiclass):
    """ Run multiclass pipe when mode = "stacking"
    
    Hardcoded to call make_stacking_pipe
    Call multiclass_print_scores to compute accuracy and roc auc scores.

    Parameters
    ----------
    X_train: DataFrame
        Training features
    
    y_train_multiclass: DataFrame
        Training labels i.e. 1 column of integer labels
    
    X_test: DataFrame
        Test features

    y_test_multiclass: DataFrame
        Test labels i.e. 1 column of integer labels
    
    """

    pipe = make_stacking_pipe(numerical_features, categorical_features)

    # pipe.fit(X_train, y_train_multiclass, ada__sample_weight=sample_weights)
    pipe.fit(X_train, y_train_multiclass)
 
    multiclass_print_scores(pipe, X_test, y_test_multiclass, "Stacking Ensemble")

def main():
    """ Main function to run different modes depending on command line arguments """

    if MODE == 'pipe':

        run_pipe(X_train, y_train, X_test, y_test)
    
    if MODE == 'pipe_mc':

        run_mc_pipe(X_train, y_train_multiclass, X_test, y_test_multiclass)

    if MODE == "grid":

        run_gridsearch()

    if MODE == "nn":
        
        run_nn()
    
    if MODE == "nn_tune":

        run_tunable_nn()
    
    if MODE == "bayes":

        run_bayes()
    
    if MODE == "autoML":

        run_autoML()
    
    if MODE == "stacking":

        run_stacking_pipe(X_train, y_train_multiclass, X_test, y_test_multiclass)


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
    
    # GLOBAL CONFIG VARIABLES
    CONTAMINATION = config["CONTAMINATION"]
    BATCH_SIZE = config["BATCH_SIZE"]

    
    # Load and split datasets. Assign to variables
    data_df = load_data(config)
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = split_data(data_df, config)

    # Create multiclass version of labels 
    y_train_multiclass, y_test_multiclass = create_powerset(y_train, y_test)

    # Save entire data to "data_all" table database
    save_data_to_db(data_df, "data_all")

    # Save train data to "data_train" table in database

    y_train_multiclass_pd = pd.DataFrame(y_train_multiclass, columns=["Powerset_Label"])

    # X_train contains index from the original df while y_train_multiclass_pd contains new index
    # Need to reset index for both to concat properly

    X_train.reset_index(drop=True, inplace=True)
    y_train_multiclass_pd.reset_index(drop=True, inplace=True)


    train_df = pd.concat([X_train, y_train_multiclass_pd], axis=1)
    save_data_to_db(train_df, "data_train_multiclass")

    #datagen_from_db(32)

    # Compute class and sample weights in case weights need to be passed to classifiers to account
    # for imbalanced datasets

    # Calculate class weights
    all_classes = np.unique(y_train_multiclass)
    class_weights = compute_class_weight("balanced", classes=all_classes, y=y_train_multiclass)

    # Calculate sample weights
    sample_weights = compute_sample_weight(class_weight=dict(zip(all_classes, class_weights)), y=y_train_multiclass)

    
    main()

