import pandas as pd
import numpy as np
import sqlite3
import json


from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from skmultilearn.problem_transform import LabelPowerset

import argparse

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


def load_data(config):
    """ Load and process traindata

    Load data using filepath and labels_to_drop from config passed into the function
    Add engineered features to enrich learning 

    Parameters
    ----------
    config: JSON object
        JSON object providing the filepath (str) and labels_to_drop (list of str)
        
    Returns
    -------
    DataFrame
        Dataset containing 2 labels "EC1" and "EC2"
    """

    traindata_filepath = config["load_data"]["traindata_filepath"]
    labels_to_drop = config["load_data"]["labels_to_drop"]

    # Load dataset
    df = pd.read_csv(traindata_filepath)

    # Add newly engineered features
    df = add_features(df)

    # Drop unnecessary target columns
    df = df.drop(columns=labels_to_drop)

    return df

def split_data(df, config):
    """  Function to split features from labels, split into train and test set
    (Not using a separate test dataset, but using the split set from train data as the test set)
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing the whole dataset
    
    config: JSON object
        JSON object providing all_labels (list of str), target_labels (list of str) and categorical_features (list of str)
    """

    # all_labels, target_labels, labels_to_drop from config.py
    target_labels = config["load_data"]["target_labels"]
    categorical_features = config["features"]["categorical_features"]

    all_columns = df.columns.tolist()
    

    features = [column for column in all_columns if column not in target_labels]
    features_no_id = [column for column in all_columns if column not in target_labels and column != 'id']

    numerical_features = [feature for feature in features_no_id if feature not in categorical_features]

    # Split dataset into X and y
    X = df[features]
    y = df[target_labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'No. of datapoints in train dataset: {len(X_train)}')
    print(f'No. of datapoints in test dataset: {len(X_test)}')

    return X_train, X_test, y_train, y_test, numerical_features, categorical_features

# Define a mapping function for DataFrame data types to SQLite data types
def map_dtype_to_sqlite(dtype):
    if dtype == 'int64':
        return 'INTEGER'
    elif dtype == 'float64':
        return 'REAL'
    elif dtype.startswith('datetime'):
        return 'TEXT'  # Assuming datetime columns should be stored as text
    else:
        return 'TEXT'  # Default to TEXT for other types

def save_data_to_db(df, db_name):

    # Connect to the SQLite Database
    conn = sqlite3.connect('enzyme_multilabel.db')
    cursor = conn.cursor()

    # Create the table data dynamically based on the df provided
    # Column headings not in quotes
    # Function to map pandas dtypes to sqlite dtypes
    columns_and_types = ', '.join([f'{col} {map_dtype_to_sqlite(df[col].dtype)}' for col in df.columns])
 

    # Set unqiue constraint on id column so that duplicate rows are not added in future
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db_name} ({columns_and_types}, UNIQUE(id))")
    # cursor.execute("ALTER TABLE data ADD CONSTRAINT unique_id UNIQUE (id)")


    # Extract column headings
    columns = df.columns.tolist()

    # Need to put column headings in quotes to refer properly
    columns_quoted = [f"'{column}'" for column in columns]
    columns_str = ', '.join(columns_quoted)

    # Create SQL query dynamically
    query = f"INSERT OR IGNORE INTO {db_name} ({columns_str}) VALUES ({', '.join(['?']*len(columns))})"

    # Convert DataFrame rows into tuples
    data_tuples = [tuple(row) for row in df.values]
    
    # Insert data into db using executemany
    cursor.executemany(query, data_tuples)

    # Execute a query to retrieve 5 random rows
    # cursor.execute('SELECT * FROM data ORDER BY RANDOM() LIMIT 5')

    # Fetch the results
    # rows = cursor.fetchall()

    # Print the results
    # for row in rows:
    #     print(row)

    # Execute a query to get the number of columns and rows
    cursor.execute(f'PRAGMA table_info({db_name})')
    columns_info = cursor.fetchall()

    # Get the number of columns
    num_columns = len(columns_info)

    # Execute a query to get the number of rows
    cursor.execute(f'SELECT COUNT(*) FROM {db_name}')
    num_rows = cursor.fetchone()[0]

    # Print the results
    print(f'Number of columns in {db_name}: {num_columns}')
    print(f'Number of rows in {db_name}: {num_rows}')

    # Extract column headings from the fetched results
    column_headings = [column[1] for column in columns_info]

    # Column headings are in quotes
    # print(column_headings)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

def datagen_from_db(batch_size):
    conn = sqlite3.connect("enzyme_multilabel.db", check_same_thread=False)
    cursor = conn.cursor()
    offset = 0

    while True:

        sql = f"""
            SELECT *
            FROM data_train_multiclass
            LIMIT {batch_size}
            OFFSET {offset}
        """

        cursor.execute(sql)
        data = cursor.fetchall()
        # data returned is a list

        if not data:
            # If there is no more data, reset the offset to start from the beginning
            offset = 0
            continue

        X = [row[:-1] for row in data]
        y = [row[-1] for row in data]
        # offset specifies how many rows to skip before selecting the rows
        offset += batch_size
        yield np.asarray(X), np.asarray(y)


class MultiLabelClassifier(BaseEstimator, ClassifierMixin): # Take note to inherit ClassifierMixin, not TransformerMixin
    
    """Custom multi-label classifier by wrapping a given classifier around a base estimator.
    This can work but hard to access estimator attributes to perform GridSearchCV

    Parameters
    ----------
    estimator: classifier for single label
        Instance such as RandomForestClassifier
    
    classifier: classifier wrapper for multi label
        Instance from skmultilearn.problem_transform 
    
    Attributes
    ----------
    estimator: base classifier

    classifier: classifier wrapper
    """

    def __init__(self, estimator, classifier):
        self.estimator = estimator
        self.classifier = classifier
    
    def fit(self, X, y):
        """Fit customer classifier against train features and labels

        Parameters
        ----------
        X: DataFrame
            Training features
        
        y: DataFrame
            Labels for training
        """
        self.model = self.classifier(self.estimator()) # Take note to pass in base(), not just base
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions on features
        
        Parameters
        ----------
        X: Dataframe
            Features for prediction
        
        Returns
        -------
        y_pred: array-like
            Predictions
        """
        self.y_pred = self.model.predict(X)
        return self.y_pred


def multilabel_print_scores(pipe, X_test, y_test, pipe_name):
    """ Function to compute accuracy and roc auc for multilabel predictions
    
    Used to evaluate performance of multilabel predictions i.e. multi-column outputs for EC1 and EC2
    Evaluation done against a test set i.e. X_test, y_test

    Parameters:
    ----------

    pipe: Pipeline
        Instance of Pipeline to be evaluated. From sklearn.pipeline or imblearn.pipeline

    X_test: DataFrame
        DataFrame containing features from test dataset
    
    y_test: DataFrame
        DataFrame containing labels from test dataset i.e. 2 columns EC1 and EC2
    
    pipe_name: str
        String describing the pipeline to be evaluated

    """

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Need toarray() to convert <class 'scipy.sparse._lil.lil_matrix'> into 2 columns of probabilities
    # Need to perform this to prevent DataConversion warning 
    roc_auc = roc_auc_score(y_test, y_proba.toarray(), multi_class="ovo", average="macro")

    print(f"The accuracy score of {pipe_name} is {accuracy}")
    print(f"The roc auc score of {pipe_name} is {roc_auc}")

def multiclass_print_scores(pipe, X_test, y_test, pipe_name):
    """ Function to compute accuracy and roc auc for multiclass predictions
    
    Used to evaluate performance of multiclass predictions i.e. single column of integer labels for
    combinatorial pairs of EC1 and EC2 labels
    Evaluation done against a test set i.e. X_test, y_test

    Parameters:
    ----------

    pipe: Pipeline
        Instance of Pipeline to be evaluated. From sklearn.pipeline or imblearn.pipeline

    X_test: DataFrame
        DataFrame containing features from test dataset
    
    y_test: DataFrame
        DataFrame containing labels from test dataset i.e. 1 column of integer labels
    
    pipe_name: str
        String describing the pipeline to be evaluated

    """
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")

    print(f"The accuracy score of {pipe_name} is {accuracy}")
    print(f"The roc auc score of {pipe_name} is {roc_auc}")

def binary_print_scores(pipe, X_test, y_test, pipe_name):
    """Function to compute accuracy and roc auc for binary predictions
    
    Used to evaluate performance of binary classification i.e. single column of binary labels
    Evaluation done against a test set i.e. X_test, y_test

    Parameters:
    ----------

    pipe: Pipeline
        Instance of Pipeline to be evaluated. From sklearn.pipeline or imblearn.pipeline

    X_test: DataFrame
        DataFrame containing features from test dataset
    
    y_test: DataFrame
        DataFrame containing labels from test dataset i.e. 2 columns EC1 and EC2
    
    pipe_name: str
        String describing the pipeline to be evaluated

    Returns
    -------
    None

    """

    y_pred = pipe.predict(X_test)
    # [:, 1] to select only probabilities for the positive class
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print(y_proba.shape)
    # y_pred = y_pred.ravel()
    # y_test = y_test.values
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"The accuracy score of {pipe_name} is {accuracy}")
    print(f"The roc auc score of {pipe_name} is {roc_auc}")

# "drop" is a ready-made FunctionTransformer
# function to remove select columns

def remove_feature(df, features_to_remove):
    """ Function to remove column or columns of features
    
    Function to be wrapped by FunctionTransformer to create a custom transformer to remove
    the selected columns. Features_to_remove can be single or multiple (as a list)
    Perform the same as "drop" (ready_made FunctionTransformer)
    
    Parameters
    ----------
    df: DataFrame
        DataFrame containing features of which some are to be removed
    
    features_to_remove: str or list
        String or list of strings to define the features to be removed
    
    Returns
    -------
    DataFrame
        DataFrame after selected features are removed
    """

    return df.drop(columns=[features_to_remove])


# Function to remove outliers via IsolationForest

def remove_outliers_IsolationForest(X, y=None, contam=0.08):
    """ Function to remove outliers from dataset
    
    Remove outliers using IsolationForest. Can be used to remove outliers only on the features or on both
    features and labels together
    
    Parameters
    ----------
    X: DataFrame
        DataFrame containing features
    
    y: DataFrame or None
        DataFrame containing labels
    
    contam: float
        Value defining the amount of contamination in dataset. Should be in (0, 0.5])
        
    Returns
    -------
    DataFrame OR DataFrame, DataFrame
        Cleaned dataframe of features or cleaned dataframes of features and labels
    
    """
    
    model = IsolationForest(contamination=contam)

    outliers = model.fit_predict(X)   

    X_cleaned = X[outliers != -1]

    if y is None:
        return X_cleaned
    else:
        y_cleaned = y[outliers != -1]
        return X_cleaned, y_cleaned

# def parse_boolean(value):
#     try:
#         return bool(eval(value))
#     except:
#         raise argparse.ArgumentTypeError("Invalid boolean value. Use 'True' or 'False'.")

def check_required_args(args):
    """ Function to ensure values for both n_components and n_estimators are given if mode is 'grid'
    
    Both n_components and n_estimators need to be defined if mode is grid. Each of them should be 
    
    Parameters
    ----------
    args: list
            List of arguments parsed from command line
    """

    if (args.mode == 'grid') and (args.n_components is None or args.n_estimators is None):
        raise argparse.ArgumentTypeError("Both --n_components and --n_estimators are required when isGridSearch is True.")
    
def validate_n_components(value):
    """ Function to validate argument for n_components in argparse
    
    Argument used to select the number of PCA components based on the variance to be explained. Therefore needs to 
    be validated between 0 and 1. Argument will be used in PCA step in pipelines
    
    Parameter
    ---------
    value: str
            String parsed from command line input and to be converted to float
    
    """

    value = float(value)

    # Check if value is between 0 and 1. Raise error if out of range
    if 0 <= value <= 1:
        return value
    else:
        raise argparse.ArgumentTypeError(f"Value must be between 0 and 1 (inclusive)")

def make_nn_model(n_inputs, n_outputs, random_seed=808):
    """ Make a fully connected neural network using keras
    
    Create a fully connected neural network comprising of 1x input layer, 1x hidden layer and 1x output layer.
    
    Parameters
    ----------
    n_inputs: int
            Number of inputs i.e. number of features
            
    n_output: int
            Number of outputs i.e. number of labels to predict per datapoint
    
    Returns
    -------
    Keras Sequential model
        Model that can be used for fitting and predicting
        
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = Sequential()
    model.add(Dense(128, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.5, seed=random_seed))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def make_tunable_nn_model(hp):
    """ Make a tunable neural network using keras and keras-tuner
    
    Create a fully connected neural network comprising of 1x input layer, 1x hidden layer and 1x output layer.
    
    Parameters
    ----------
    n_inputs: int
            Number of inputs i.e. number of features
            
    n_output: int
            Number of outputs i.e. number of labels to predict per datapoint
    
    Returns
    -------
    Keras Sequential model
        Model that can be used for fitting and predicting
        
    """
    random_seed = config["RANDOM_SEED"]
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = Sequential()

    # Input layer
    model.add(Dense(units=hp.Int('input_dense', min_value=64, max_value=512, step=16), input_dim=47, kernel_initializer='he_uniform', activation='relu'))

    # Tunable number of blocks containing one dense and one dropout layer
    for i in range(hp.Int('num_dense_dropout', 1, 2)):
        model.add(Dense(units=hp.Int('hidden_dense', min_value=16, max_value=64, step=16), kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(hp.Float('hidden_dropout', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(4, activation='softmax'))

    # Setting up choices for different optimizers
    hp_optimizer = hp.Choice("Optimizer", values=['Adam', 'SGD'])

    if hp_optimizer == 'Adam':
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    elif hp_optimizer == 'SGD':
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
        nesterov = True
        momentum = 0.9
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp_optimizer, metrics=['accuracy'])
    return model

def create_powerset(y_train, y_test):
    """Create powerset from multi-labels
    
    Combine y_train and y_test into one dataset, transform into powerset and remove the original columns 
    "EC1" and "EC2". Combining y_train and y_test and transforming the combined dataset ensures that 
    the powerset labels are consistent across the train and test data
    
    Paramters
    ---------
    y_train: DataFrame
            Labels for the training dataset. 2 columns, ['EC1', 'EC2']
    
    y_test: DataFrame
            Labels for the test dataset. 2 columns, ['EC1', 'EC2']

    Returns
    -------
    DataFrame, DataFrame
        train and test labels containing the powerset i.e. integer labels for various combinatorial pairs of 'EC1' and 'EC2'
    """

    # Combine train and test data into one dataset
    y_combined = pd.concat([y_train, y_test], axis=0)

    # Create powerset labels
    lp_transformer = LabelPowerset()
    y_combined['powerset'] = lp_transformer.transform(y_combined)

    # Remove the original multi-label columns
    y_combined_multiclass = y_combined.drop(columns=['EC1', 'EC2'])

    # Split the transformed dataset back into train and test set
    y_train_multiclass = y_combined_multiclass.iloc[:len(y_train)]
    y_test_multiclass = y_combined_multiclass.iloc[len(y_train):]

    # Need to convert to numpy and call ravel() for the "all_classes" argument and input shapes to work
    y_train_multiclass = y_train_multiclass.to_numpy().ravel()
    y_test_multiclass = y_test_multiclass.to_numpy().ravel()

    return y_train_multiclass, y_test_multiclass
 

def add_features(df):
    """Add engineered features to improve classification
    
    List of engineered features taken from https://www.kaggle.com/code/lukabarbakadze/multilabel-classification-challange
    
    Parameter
    ---------
    
    df: pandas DataFrame
        DataFrame to which newly engineered features are added
        
    Return
    ------
        DataFrame

    """

    df['SpanEstateIndex'] = df['MaxAbsEStateIndex'] - df['MinEStateIndex']
    df['MaxEstate_percent'] = df['MaxAbsEStateIndex'] / (df['SpanEstateIndex'] + 1e-12)
    df['MinEstate_percent'] = df['MinEStateIndex'] / (df['SpanEstateIndex'] + 1e-12)
    df['BertzCT_MaxAbsEStateIndex_Ratio'] = df['BertzCT'] / (df['MaxAbsEStateIndex'] + 1e-12)
    df['BertzCT_ExactMolWt_Product'] = df['BertzCT'] * df['ExactMolWt']
    df['NumHeteroatoms_FpDensityMorgan1_Ratio'] = df['NumHeteroatoms'] / (df['FpDensityMorgan1'] + 1e-12)
    df['VSA_EState9_EState_VSA1_Ratio'] = df['VSA_EState9'] / (df['EState_VSA1'] + 1e-12)
    df['PEOE_VSA10_SMR_VSA5_Ratio'] = df['PEOE_VSA10'] / (df['SMR_VSA5'] + 1e-12)
    df['Chi1v_ExactMolWt_Product'] = df['Chi1v'] * df['ExactMolWt']
    df['Chi2v_ExactMolWt_Product'] = df['Chi2v'] * df['ExactMolWt']
    df['Chi3v_ExactMolWt_Product'] = df['Chi3v'] * df['ExactMolWt']
    df['EState_VSA1_NumHeteroatoms_Product'] = df['EState_VSA1'] * df['NumHeteroatoms']
    df['PEOE_VSA10_Chi1_Ratio'] = df['PEOE_VSA10'] / (df['Chi1'] + 1e-12)
    df['MaxAbsEStateIndex_NumHeteroatoms_Ratio'] = df['MaxAbsEStateIndex'] / (df['NumHeteroatoms'] + 1e-12)
    df['BertzCT_Chi1_Ratio'] = df['BertzCT'] / (df['Chi1'] + 1e-12)

    return df

# Read config.py
try:
    with open('../config.py', "r") as inp:
        config = json.load(inp)

except:
    raise FileNotFoundError("config.py is not found")
