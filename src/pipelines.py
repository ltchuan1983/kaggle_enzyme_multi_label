import pandas as pd

from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE

import xgboost as xgb
from catboost import CatBoostClassifier

from helper_functions import remove_feature, MultiLabelClassifier, remove_outliers_IsolationForest

def make_numerical_transformer():
    """Prepare pipeline to transform numerical features

    Transformation includes standard scaling prior to PCA
    
    Returns
    -------
    Pipeline
    """

    # Define transformer for numerical features
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA())
    ])

    return numeric_transformer

def make_categorical_transformer():

    """Prepare pipeline to one-hot encode categorical features
    
    Returns
    -------
    Pipeline
    """

    # Define transformer for categorical features

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder())
    ])

    return categorical_transformer

def make_preprocessor(numerical_features, categorical_features):
    """ Create ColumnTransformer that performs different transformations on numerical and 
    categorical features
    
    Parameters
    ----------
    numerical_features: list
        list of strs specifying the names of numerical features
    
    categorical_features: list
        list of strs specifying the names of categorical features
    
    Returns
    -------
    ColumnTransformer
        
    """
    
    numeric_transformer = make_numerical_transformer()
    categorical_transformer = make_categorical_transformer()

    # Column Transformer 
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor

def make_voting_classifier():
    """Make voting classifier comprising of CatBoostClassifier, ExtraTreesClassifier,
    logistic regression (with polynomial features), KNeighborsClassifier
    
    auto_class_weights set to "balanced" to account for imbalanced datasets
    
    Returns
    -------
    VotingClassifier
    """

    voting_clf = VotingClassifier([
        ('cat', CatBoostClassifier(n_estimators=100, loss_function="MultiClass", auto_class_weights="Balanced", verbose=0, l2_leaf_reg=12, depth=6)),
        ('et', ExtraTreesClassifier(n_estimators=100, min_samples_leaf=30, random_state=1)),
        ('lr', make_pipeline(PolynomialFeatures(2, include_bias=True),
                             LogisticRegression(C=0.01, max_iter=700))),
        ('knn', KNeighborsClassifier(n_neighbors=500, weights='distance'))

        ], 
        voting='soft',
        weights=[0.3, 0.2, 0.4, 0.1]
    )

    return voting_clf

# Custom Transformer to remove the 'id' column

ID_Remover = FunctionTransformer(remove_feature, kw_args={'features_to_remove': 'id'})

# Define and fit OneVsRest Pipeline

def make_ovr_pipe(numerical_features, categorical_features):
    """ Pipe for OneVsRest using RandomForestClassifier as base estimator"""
    ovr_pipe = Pipeline([
        ('id_remover', ID_Remover),
        #('onevsrest_rf', MultiLabelClassifier(RandomForestClassifier, OneVsRestClassifier))
        ('onevsrest_rf', OneVsRestClassifier(RandomForestClassifier()))
    ])

    return ovr_pipe

def make_br_pipe(numerical_features, categorical_features):
    """ Pipe for BinaryRelevance using RandomForestClassifier as base estimator"""
    br_pipe = Pipeline([
        ('id_remover', ID_Remover),
        ('binaryrelevance_rf', MultiLabelClassifier(RandomForestClassifier, BinaryRelevance))
    ])

    return br_pipe

def make_cc_pipe(numerical_features, categorical_features):
    """ Pipe for ClassifierChain using RandomForestClassifier as base estimator"""
    cc_pipe = Pipeline([
        ('id_remover', ID_Remover),
        ('classifierchain_rf', MultiLabelClassifier(RandomForestClassifier, ClassifierChain))
    ])

    return cc_pipe


def make_lp_pipe(numerical_features, categorical_features):
    """ Pipe for LabelPowerset using RandomForestClassifier as base estimator"""
    lp_pipe = Pipeline([
        ('id_remover', ID_Remover),
        ('labelpowerset_rf', LabelPowerset(RandomForestClassifier()))
    ])

    return lp_pipe

def make_lp_ex_pipe(numerical_features, categorical_features):
    """ Pipe for LabelPowerset using RandomForestClassifier as base estimator
    
    Remove outliers before classification
    """
    lp_ex_pipe = IMBPipeline([
        ('id_remover', ID_Remover),
        ('outlier_remover', FunctionSampler(func=remove_outliers_IsolationForest, kw_args={'contam': 0.1}, validate=False)),
        ('classifierchain_rf', MultiLabelClassifier(RandomForestClassifier, LabelPowerset))
    ])

    return lp_ex_pipe

def make_lp_pca_pipe(numerical_features, categorical_features):
    """ Pipe for LabelPowerset using AdaBoostClassifier as base estimator
    
    Include Standard Scaling and PCA before classification. Optional step to transform data into DataFrame.
    """

    lp_pca_pipe = Pipeline([
        ('id_remover', ID_Remover),
        #('outlier_remover', FunctionSampler(func=remove_outliers_IsolationForest, validate=False)),
        ('scaler', StandardScaler()),
        #('to_dataframe', FunctionTransformer(lambda x: pd.DataFrame(x, columns=features_no_id))),
        ('pca', PCA(n_components=0.92)),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
        ('classifierchain_rf', MultiLabelClassifier(AdaBoostClassifier, LabelPowerset))
    ])

    return lp_pca_pipe

def make_lp_GSCV_pipe(numerical_features, categorical_features):
    """ Pipe for LabelPowerset using AdaBoostClassifier as base estimator
    
    Perform Standard Scaling and PCA on numerical features. Perform one-hot encoding on categorical features.
    Pipe to be used when command line argument mode = "grid" 

    Parameters:
    -----------
    numerical_features: list
        list of str containing label names for the numerical features

    categorical_features: list
        list of str containing label names for the categorical features

    Returns:
    --------
    Pipeline

    """
    # Preprocessing numerical & categorical features separately make it slightly worse

    preprocessor = make_preprocessor(numerical_features, categorical_features)

    lp_GSCV_pipe = Pipeline([
        ('id_remover', ID_Remover),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
        #('lp_rf', MultiLabelClassifier(estimator=AdaBoostClassifier, classifier=LabelPowerset))
        ('lp_rf', LabelPowerset(AdaBoostClassifier())) #default n_estimators = 50
    ])

    return lp_GSCV_pipe

def make_multiclass_pipe(numerical_features, categorical_features):

    """ Pipe performing multiclass classification with voting classifier
    
    Perform Standard Scaling and PCA on numerical features. Perform one-hot encoding on categorical features.

    Parameters:
    -----------
    numerical_features: list
        list of str containing label names for the numerical features

    categorical_features: list
        list of str containing label names for the categorical features

    Returns:
    --------
    Pipeline

    """

    preprocessor = make_preprocessor(numerical_features, categorical_features)

    voting_clf = make_voting_classifier()

    multiclass_pipe = IMBPipeline([
        ('id_remover', ID_Remover),
        #('smote', SMOTE(sampling_strategy="auto", random_state=42)),
        ('preprocessor', preprocessor),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
        ('voter', voting_clf)
    ])

    # SMOTE did not improve performance and was therefore left out. 

    # multiclass_pipe = Pipeline([
    #     ('id_remover', ID_Remover),
    #     ('scaler', StandardScaler()),
    #     ('pca', PCA()),
    #     ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
    #     ('ada', CatBoostClassifier(loss_function="MultiClass", auto_class_weights="Balanced", verbose=0, l2_leaf_reg=7)) #default n_estimators = 50
    # ])

    return multiclass_pipe

def make_binary_pipe():
    """Pipe running binary classification for a single label
    
    Pipe is to be fitted with features and a single label.
    For CatBoostClassifier, auto_class_weights is set to "Balanced" to handle imbalanced dataset
    """

    binary_pipe = Pipeline([
        ('id_remover', ID_Remover),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
        ('ada', CatBoostClassifier(auto_class_weights="Balanced", verbose=0, l2_leaf_reg=7)) #default n_estimators = 50
    ])

    return binary_pipe

def make_simplegrid_pipe(numerical_features, categorical_features):

    """ Pipe performing multiclass classification with CatBoostClassifier
    
    Pipe to be used when command line argument mode = "grid" 

    Parameters:
    -----------
    numerical_features: list
        list of str containing label names for the numerical features

    categorical_features: list
        list of str containing label names for the categorical features

    Returns:
    --------
    Pipeline

    """

    simplegrid_pipe = IMBPipeline([
        ('id_remover', ID_Remover),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
        ('rf', RandomForestClassifier())
        #('cat', CatBoostClassifier(auto_class_weights="Balanced", verbose=0, l2_leaf_reg=7))
    ])

    # SMOTE did not improve performance and was therefore left out. 

    # multiclass_pipe = Pipeline([
    #     ('id_remover', ID_Remover),
    #     ('scaler', StandardScaler()),
    #     ('pca', PCA()),
    #     ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
    #     ('ada', CatBoostClassifier(loss_function="MultiClass", auto_class_weights="Balanced", verbose=0, l2_leaf_reg=7)) #default n_estimators = 50
    # ])

    return simplegrid_pipe

def make_lp_brfc_pipe(numerical_features, categorical_features):
    """Pipe running LabelPowerset using EasyEnsembleClassifier as base estimator
    
    Using EasyEnsembleClassifier from imblearn.ensemble to handle imbalanced datasets
    """
    # Preprocessing numerical & categorical features separately make it slightly worse

    lp_bfrc_pipe = Pipeline([
        ('id_remover', ID_Remover),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
    #     ('lp_rf', MultiLabelClassifier(estimator=AdaBoostClassifier, classifier=LabelPowerset))
        ('lp_rf', LabelPowerset(EasyEnsembleClassifier(sampling_strategy="not majority"))) #default n_estimators = 50
    ])

    return lp_bfrc_pipe

def make_stacking_pipe(numerical_features, categorical_features):

    """ Pipe performing multiclass classification with stacking ensemble
    
    Perform Standard Scaling and PCA on numerical features. Perform one-hot encoding on categorical features.

    Parameters:
    -----------
    numerical_features: list
        list of str containing label names for the numerical features

    categorical_features: list
        list of str containing label names for the categorical features

    Returns:
    --------
    Pipeline

    """

    preprocessor = make_preprocessor(numerical_features, categorical_features)

    # voting_clf = VotingClassifier([
    # ('cat', CatBoostClassifier(n_estimators=100, loss_function="MultiClass", auto_class_weights="Balanced", verbose=0, l2_leaf_reg=12, depth=6)),
    # ('et', ExtraTreesClassifier(n_estimators=100, min_samples_leaf=30, random_state=1)),
    # ('lr', make_pipeline(PolynomialFeatures(2, include_bias=True),
    #                         LogisticRegression(C=0.01, max_iter=700))),
    # ('knn', KNeighborsClassifier(n_neighbors=500, weights='distance'))

    # ], 
    # voting='soft',
    # weights=[0.3, 0.2, 0.4, 0.1]
    # )

    # Create base learners
    
    base_learners_1 = [
                    ('knn', KNeighborsClassifier(n_neighbors=500, weights='distance')), 
                    # ('lr', make_pipeline(PolynomialFeatures(2, include_bias=True),
                    #          LogisticRegression(C=0.01, max_iter=700)))

    ]
    
    base_learners_2 = [
                    ('cat', CatBoostClassifier(n_estimators=500, loss_function="MultiClass", auto_class_weights="Balanced", verbose=0, l2_leaf_reg=12, depth=6)),
                    # ('knn', KNeighborsClassifier(n_neighbors=500, weights='distance')), 
                    # ('lr', make_pipeline(PolynomialFeatures(2, include_bias=True),
                    #          LogisticRegression(C=0.01, max_iter=700)))

    ]

    # Initialize stacking classifiers

    # layer_2 = StackingClassifier(estimators=base_learners_2, final_estimator=LogisticRegression(C=0.01, max_iter=700))

    # stackclf = StackingClassifier(estimators=base_learners_1, final_estimator=layer_2)

    stackclf = StackingClassifier(estimators=base_learners_2, final_estimator=LogisticRegression(C=0.01, max_iter=700))

    stacking_pipe = IMBPipeline([
        ('id_remover', ID_Remover),
        ('preprocessor', preprocessor),
        ('to_dataframe2', FunctionTransformer(lambda x: pd.DataFrame(x))),
        ('stack', stackclf)
    ])

    return stacking_pipe
