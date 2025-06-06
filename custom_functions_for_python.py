################################################################################
############################ Import Python modules #############################
################################################################################

from time import time
import os

# For data processing and visualization
import numpy as np
import pandas as pd
from scipy.stats import uniform, loguniform
import re
import matplotlib.pyplot as plt
from functools import partial

# For machine learning workflow
import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate, LeaveOneGroupOut, StratifiedGroupKFold, learning_curve, HalvingRandomSearchCV, ValidationCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
import joblib

################################################################################
############################# Custom Functions #################################
################################################################################

######################## Supervised Classifier Building ########################

column_rename_map_from_copilot = {
	'FWS_length': 'FWS_Length',
	'FWS_total': 'FWS_Total',
	'FWS_maximum': 'FWS_Maximum',
	'FWS_average': 'FWS_Average',
	'FWS_inertia': 'FWS_Inertia',
	'FWS_centreOfGravity': 'FWS_Center_of_gravity',
	'FWS_fillFactor': 'FWS_Fill_factor',
	'FWS_asymmetry': 'FWS_Asymmetry',
	'FWS_numberOfCells': 'FWS_Number_of_cells',
	'FWS_sampleLength': 'Sample_Length',
	'FWS_timeOfArrival': 'Arrival_Time',
	'FWS_first': 'FWS_First',
	'FWS_last': 'FWS_Last',
	'FWS_minimum': 'FWS_Minimum',
	'FWS_swscov': 'FWS_SWS_covariance',
	'FWS_variableLength': 'FWS_Length(50%)',
	'Sidewards_Scatter_length': 'Sidewards_Scatter_Length',
	'Sidewards_Scatter_total': 'Sidewards_Scatter_Total',
	'Sidewards_Scatter_maximum': 'Sidewards_Scatter_Maximum',
	'Sidewards_Scatter_average': 'Sidewards_Scatter_Average',
	'Sidewards_Scatter_inertia': 'Sidewards_Scatter_Inertia',
	'Sidewards_Scatter_centreOfGravity': 'Sidewards_Scatter_Center_of_gravity',
	'Sidewards_Scatter_fillFactor': 'Sidewards_Scatter_Fill_factor',
	'Sidewards_Scatter_asymmetry': 'Sidewards_Scatter_Asymmetry',
	'Sidewards_Scatter_numberOfCells': 'Sidewards_Scatter_Number_of_cells',
	'Sidewards_Scatter_sampleLength': 'Sidewards_Scatter_Length(50%)',
	'Sidewards_Scatter_timeOfArrival': 'Sidewards_Scatter_Arrival_Time',
	'Sidewards_Scatter_first': 'Sidewards_Scatter_First',
	'Sidewards_Scatter_last': 'Sidewards_Scatter_Last',
	'Sidewards_Scatter_minimum': 'Sidewards_Scatter_Minimum',
	'Sidewards_Scatter_swscov': 'Sidewards_Scatter_SWS_covariance',
	'Sidewards_Scatter_variableLength': 'Sidewards_Scatter_varLength(50%)',
	'Fl_Yellow_length': 'Fl_Yellow_Length',
	'Fl_Yellow_total': 'Fl_Yellow_total',
	'Fl_Yellow_maximum': 'Fl_Yellow_Maximum',
	'Fl_Yellow_average': 'Fl_Yellow_Average',
	'Fl_Yellow_inertia': 'Fl_Yellow_Inertia',
	'Fl_Yellow_centreOfGravity': 'Fl_Yellow_Center_of_gravity',
	'Fl_Yellow_fillFactor': 'Fl_Yellow_Fill_factor',
	'Fl_Yellow_asymmetry': 'Fl_Yellow_Asymmetry',
	'Fl_Yellow_numberOfCells': 'Fl_Yellow_Number_of_cells',
	'Fl_Yellow_sampleLength': 'Fl_Yellow_Length(50%)',
	'Fl_Yellow_timeOfArrival': 'Fl_Yellow_Arrival_Time',
	'Fl_Yellow_first': 'Fl_Yellow_First',
	'Fl_Yellow_last': 'Fl_Yellow_Last',
	'Fl_Yellow_minimum': 'Fl_Yellow_Minimum',
	'Fl_Yellow_swscov': 'Fl_Yellow_SWS_covariance',
	'Fl_Yellow_variableLength': 'Fl_Yellow_varLength(50%)',
	'Fl_Orange_length': 'Fl_Orange_Length',
	'Fl_Orange_total': 'Fl_Orange_total',
	'Fl_Orange_maximum': 'Fl_Orange_Maximum',
	'Fl_Orange_average': 'Fl_Orange_Average',
	'Fl_Orange_inertia': 'Fl_Orange_Inertia',
	'Fl_Orange_centreOfGravity': 'Fl_Orange_Center_of_gravity',
	'Fl_Orange_fillFactor': 'Fl_Orange_Fill_factor',
	'Fl_Orange_asymmetry': 'Fl_Orange_Asymmetry',
	'Fl_Orange_numberOfCells': 'Fl_Orange_Number_of_cells',
	'Fl_Orange_sampleLength': 'Fl_Orange_Length(50%)',
	'Fl_Orange_timeOfArrival': 'Fl_Orange_Arrival_Time',
	'Fl_Orange_first': 'Fl_Orange_First',
	'Fl_Orange_last': 'Fl_Orange_Last',
	'Fl_Orange_minimum': 'Fl_Orange_Minimum',
	'Fl_Orange_swscov': 'Fl_Orange_SWS_covariance',
	'Fl_Orange_variableLength': 'Fl_Orange_varLength(50%)',
	'Fl_Red_length': 'Fl_Red_Length',
	'Fl_Red_total': 'Fl_Red_total',
	'Fl_Red_maximum': 'Fl_Red_Maximum',
	'Fl_Red_average': 'Fl_Red_Average',
	'Fl_Red_inertia': 'Fl_Red_Inertia',
	'Fl_Red_centreOfGravity': 'Fl_Red_Center_of_gravity',
	'Fl_Red_fillFactor': 'Fl_Red_Fill_factor',
	'Fl_Red_asymmetry': 'Fl_Red_Asymmetry',
	'Fl_Red_numberOfCells': 'Fl_Red_Number_of_cells',
	'Fl_Red_sampleLength': 'Fl_Red_Length(50%)',
	'Fl_Red_timeOfArrival': 'Fl_Red_Arrival_Time',
	'Fl_Red_first': 'Fl_Red_First',
	'Fl_Red_last': 'Fl_Red_Last',
	'Fl_Red_minimum': 'Fl_Red_Minimum',
	'Fl_Red_swscov': 'Fl_Red_SWS_covariance',
	'Fl_Red_variableLength': 'Fl_Red_varLength(50%)',
	'Curvature_length': 'Curvature_Length',
	'Curvature_total': 'Curvature_Total',
	'Curvature_maximum': 'Curvature_Maximum',
	'Curvature_average': 'Curvature_Average',
	'Curvature_inertia': 'Curvature_Inertia',
	'Curvature_centreOfGravity': 'Curvature_Center_of_gravity',
	'Curvature_fillFactor': 'Curvature_Fill_factor',
	'Curvature_asymmetry': 'Curvature_Asymmetry',
	'Curvature_numberOfCells': 'Curvature_Number_of_cells',
	'Curvature_sampleLength': 'Curvature_Length(50%)',
	'Curvature_timeOfArrival': 'Curvature_Arrival_Time',
	'Curvature_first': 'Curvature_First',
	'Curvature_last': 'Curvature_Last',
	'Curvature_minimum': 'Curvature_Minimum',
	'Curvature_swscov': 'Curvature_SWS_covariance',
	'Curvature_variableLength': 'Curvature_varLength(50%)',
	'Forward_Scatter_Left_length': 'Forward_Scatter_Left_Length',
	'Forward_Scatter_Left_total': 'Forward_Scatter_Left_Total',
	'Forward_Scatter_Left_maximum': 'Forward_Scatter_Left_Maximum',
	'Forward_Scatter_Left_average': 'Forward_Scatter_Left_Average',
	'Forward_Scatter_Left_inertia': 'Forward_Scatter_Left_Inertia',
	'Forward_Scatter_Left_centreOfGravity': 'Forward_Scatter_Left_Center_of_gravity',
	'Forward_Scatter_Left_fillFactor': 'Forward_Scatter_Left_Fill_factor',
	'Forward_Scatter_Left_asymmetry': 'Forward_Scatter_Left_Asymmetry',
	'Forward_Scatter_Left_numberOfCells': 'Forward_Scatter_Left_Number_of_cells',
	'Forward_Scatter_Left_sampleLength': 'Forward_Scatter_Left_Length(50%)',
	'Forward_Scatter_Left_timeOfArrival': 'Forward_Scatter_Left_Arrival_Time',
	'Forward_Scatter_Left_first': 'Forward_Scatter_Left_First',
	'Forward_Scatter_Left_last': 'Forward_Scatter_Left_Last',
	'Forward_Scatter_Left_minimum': 'Forward_Scatter_Left_Minimum',
	'Forward_Scatter_Left_swscov': 'Forward_Scatter_Left_SWS_covariance',
	'Forward_Scatter_Left_variableLength': 'Forward_Scatter_Left_varLength(50%)',
	'Forward_Scatter_Right_length': 'Forward_Scatter_Right_Length',
	'Forward_Scatter_Right_total': 'Forward_Scatter_Right_Total',
	'Forward_Scatter_Right_maximum': 'Forward_Scatter_Right_Maximum',
	'Forward_Scatter_Right_average': 'Forward_Scatter_Right_Average',
	'Forward_Scatter_Right_inertia': 'Forward_Scatter_Right_Inertia',
	'Forward_Scatter_Right_centreOfGravity': 'Forward_Scatter_Right_Center_of_gravity',
	'Forward_Scatter_Right_fillFactor': 'Forward_Scatter_Right_Fill_factor',
	'Forward_Scatter_Right_asymmetry': 'Forward_Scatter_Right_Asymmetry',
	'Forward_Scatter_Right_numberOfCells': 'Forward_Scatter_Right_Number_of_cells',
	'Forward_Scatter_Right_sampleLength': 'Forward_Scatter_Right_Length(50%)',
	'Forward_Scatter_Right_timeOfArrival': 'Forward_Scatter_Right_Arrival_Time',
	'Forward_Scatter_Right_first': 'Forward_Scatter_Right_First',
	'Forward_Scatter_Right_last': 'Forward_Scatter_Right_Last',
	'Forward_Scatter_Right_minimum': 'Forward_Scatter_Right_Minimum',
	'Forward_Scatter_Right_swscov': 'Forward_Scatter_Right_SWS_covariance',
	'Forward_Scatter_Right_variableLength': 'Forward_Scatter_Right_varLength(50%)'
}


def splitTrainingSet(training_set, target_name, group_name, weight_name):
    """
    Splits the training set into three objects:
    - X_train: features table
    - y_train: target array
    - groups: group array (used in group cross-validation)
    Drops optional columns only if they are present.
    """
    y_train = training_set[target_name]
    # List of columns to drop if present
    columns_to_drop = [target_name, group_name, weight_name, "Particle_ID", "TriggerLevel"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in training_set.columns]
    X_train = training_set.drop(columns=existing_columns_to_drop)
    groups = training_set[group_name]
    sample_weights = training_set[weight_name]
    return X_train, y_train, groups, sample_weights
  


def createParametersList(logreg_clf, svm_clf, rf_clf, hgb_clf):
  """This function creates a list with all the hyperparameters range to be used in the tuning process, it needs the classifiers used as argument 
  (you can modify the arguments of this function if you want to use other classifiers than the default ones, you can also modify the default hyperparameters and their range)"""
  
  class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)
    
    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
  
  class uniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = uniform(a, b)
    
    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
  
  # SelectKBest
  #FEATURES_SIZES = uniform_int(1, 15)
  # Logistic Regression
  COST_logreg = loguniform(2**-10, 2**0)
  L1_RATIO = uniform(0.0, 1.0)
  # SVM
  POLY_DEGREE = uniform_int(2, 4)
  GAMMA = loguniform(2**-10, 2**10)
  # Random Forest
  MTRY = uniform(0.1, 0.9)
  SAMPLE_FRACTION = uniform(0.1, 0.9)
  NTREES = uniform_int(100, 900)
  # Histogram-based Gradient Boosting
  MAX_DEPTH = uniform_int(1, 14)
  MAX_FEATURES = uniform(0.1, 0.9)
  LEARNING_RATE = loguniform(2**-10, 2**0)
  
  # Build individual models grids
  # Logistic Regression
  parameters_logreg ={
    #'selector__k': FEATURES_SIZES,
    'classifier': [logreg_clf],
    'classifier__C': COST_logreg,
    'classifier__l1_ratio': L1_RATIO,
  }
  # Support Vector Machines
  parameters_svm = {
    #'selector__k': FEATURES_SIZES,
    'classifier': [svm_clf],
    'classifier__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    'classifier__degree': POLY_DEGREE,
    'classifier__gamma': GAMMA,
  }
  # Random Forest
  parameters_rf = {
    #'selector__k': FEATURES_SIZES,
    'classifier': [rf_clf],
    'classifier__n_estimators': NTREES,
    'classifier__max_features': MTRY,
    'classifier__max_samples': SAMPLE_FRACTION,
  }
  # Histogram-based Gradient Boosting
  parameters_hgb = {
    #'selector__k': FEATURES_SIZES,
    'classifier': [hgb_clf],
    'classifier__max_iter': NTREES,
    'classifier__max_depth': MAX_DEPTH,
    'classifier__max_features': MAX_FEATURES,
    'classifier__learning_rate' : LEARNING_RATE,
    
  }
  # Finally pool all dictionaries together in a list
  parameters = [parameters_logreg, parameters_svm, parameters_rf, parameters_hgb]
  return parameters



def applyNestedCrossValidation(rng, training_set, target_name, group_name, weight_name, cores, select_K):
  """This function applies the nested cross validation process
  It starts by splitting the training set, set the classifiers to be used and the base pipeline
  Then it creates the parameters list to be used in tuning, sets the spliting objects for cross-validation
  Finally the inner cross-validation object is created and passed to the cross_validate for the outer cross-validation"""
  
  # Make sure the numbers given as argument are integer
  cores = int(cores)
  select_K = int(select_K)
  
  # Split the training set into X_train, y_train, groups and sample weights objects
  X_train, y_train, groups, sample_weights = splitTrainingSet(training_set, target_name, group_name, weight_name)
  
  # Set the classifiers to be tested
  dummy_clf = DummyClassifier(random_state = rng).set_fit_request(sample_weight=True)
  logreg_clf = LogisticRegression(random_state = rng, class_weight = 'balanced', penalty = 'elasticnet', solver = 'saga').set_fit_request(sample_weight=True)
  svm_clf = SVC(random_state = rng, class_weight = 'balanced').set_fit_request(sample_weight=True)
  rf_clf = RandomForestClassifier(random_state = rng, class_weight = 'balanced').set_fit_request(sample_weight=True)
  hgb_clf = HistGradientBoostingClassifier(random_state = rng, class_weight = 'balanced').set_fit_request(sample_weight=True)
  
  # Set the feature selection method
  #nfeats = X_train.shape[1]
  features_selector = SelectKBest(score_func = partial(mutual_info_classif, random_state=rng), k = select_K)
  
  # Set the base pipeline
  pipe = Pipeline([
    ('selector', features_selector),
    ('scaler', StandardScaler().set_fit_request(sample_weight=True)),
    ('classifier', 'passthrough')
    ])
  
  # Set the hyperparameters lists
  # For dummy classifier
  parameters_dummy = {
    'strategy': ('most_frequent', 'prior', 'stratified', 'uniform'),
  }
  
  # All other classifiers
  parameters = createParametersList(logreg_clf = logreg_clf, svm_clf = svm_clf, rf_clf = rf_clf, hgb_clf = hgb_clf)
  
  # How I used to have it, now causes error "ValueError: could not convert string to float: 'OraNano1'" 
  # I wonder if that indicates that I am passing in variables differently now. I must be, since this used to work?
  # I think a duplicate label column is being passed in as a predictor when before it was not
  inner_cv = StratifiedGroupKFold(n_splits=2, shuffle = True, random_state = 42)
  outer_cv = StratifiedGroupKFold(n_splits=2, shuffle = True, random_state = 42)
  #This is what lucinda has, and it leaves the dummy training forever:
  #inner_cv = LeaveOneGroupOut()
  #outer_cv = LeaveOneGroupOut()
  
  # scoring function
  scorer = make_scorer(matthews_corrcoef).set_score_request(sample_weight=True)
  
  # Set the inner cross-validation  objects (hyperparameters tuning)
  # Dummy classifier (done in grid search because of few number of candidates)
  inner_dummy = GridSearchCV(
    estimator = dummy_clf, 
    param_grid = parameters_dummy, 
    cv = inner_cv, 
    n_jobs = cores, 
    scoring = scorer, 
    verbose=10
  )
  
  # For all other classifiers (halving random search because of the huge number of candidates)
  inner_models_halving = HalvingRandomSearchCV(
    estimator = pipe, 
    param_distributions = parameters, 
    factor = 2, 
    min_resources = 50, 
    cv = inner_cv, 
    error_score = "raise", 
    return_train_score = True,
    n_jobs = cores, 
    scoring = scorer, 
    verbose=10, 
    random_state = rng
  )
  
  print("Starting Nested Cross Validation ... \n")
  
  # Run the nested cross-validation 
  # For Dummy classifier
  dummy_start_time = time()
  outer_dummy = cross_validate(
    estimator = inner_dummy, 
    X = X_train, 
    y = y_train, 
    cv = outer_cv, 
    scoring = scorer, 
    verbose=10, 
    return_estimator=True, 
    error_score = "raise", 
    params={"sample_weight": sample_weights, "groups": groups}
  )
  dummy_stop_time = time() - dummy_start_time
  print(f"Time elapsed for Dummy Classifier : {dummy_stop_time} s \n")
  
  # Get the outer cross-validation  scores
  scores_dummy = outer_dummy['test_score']
  print(f"Outer CV scores for Dummy Classifier = {scores_dummy} \n \n")
  
  # For all other classifiers
  hrscv_start_time = time()
  outer_models = cross_validate(
    estimator = inner_models_halving, 
    X = X_train, 
    y = y_train,
    cv = outer_cv, 
    scoring = scorer, 
    verbose=10, 
    return_estimator=True, 
    error_score = "raise", 
    params={"sample_weight": sample_weights, "groups": groups}
  )
  hrscv_stop_time = time() - hrscv_start_time
  print(f"Time elapsed for Halving Randomized Search : {hrscv_stop_time} s \n")
  
  # Get the outer cross-validation  scores
  scores = outer_models['test_score']
  print(f"Outer CV scores = {scores} \n")
  
  return outer_models, inner_models_halving, scores
def fitFinalClassifier(inner_models_halving, filename_finalFittedModel, training_set, target_name, group_name, weight_name):
  """Runs a final hyperparameters tuning process after nested cross-validation and save the best candidate as final model"""
  # Split the training set into X_train, y_train and groups objects
  X_train, y_train, groups, sample_weights = splitTrainingSet(training_set, target_name, group_name, weight_name)
  
  # Train and fit final classifier with hyperparameters tuning
  print("Fitting final classifier on entire data set... \n")
  cv_start_time = time()
  final_search = inner_models_halving.fit(X = X_train, y = y_train, groups = groups, sample_weight = sample_weights) # ou classifier__sample_weight
  cv_stop_time = time() - cv_start_time
  print(f"Done ! Time elapsed : {cv_stop_time} s \n")
  
  # Get best classifier
  fitted_final_classifier = final_search.best_estimator_
  
  # Save final classifier
  joblib.dump(fitted_final_classifier, filename_finalFittedModel)
  return fitted_final_classifier


def saveNestedCvResults(outer_models, filename_cvResults, scores):
  """Extract the nested cross-validation results and save them as a table"""
  
  # Get all the inner cross-validation objects from each outer fold
  models = outer_models['estimator']
  
  # For each of these objects, extract the cross-validation results, add the outer split id and score
  list_of_dataframes = []
  split = 1
  
  for i, model in enumerate(models):
      cv_results = pd.DataFrame(model.cv_results_)
      cv_results['outer_splits'] = split
      cv_results['outer_split_test_score'] = scores[split-1]
      list_of_dataframes.append(pd.DataFrame(cv_results))
      split = split + 1
  
  # Concatenate the tables of all outer splits
  all_dfs = pd.concat(list_of_dataframes)
  
  # Rename the classifiers name stored in the param_classifier column to only keep the name (and not all the details of their argument)
  all_dfs['param_classifier'] = all_dfs['param_classifier'].astype(str).apply(lambda x: re.search(r'^[^\(]+(?=\()', x).group() if re.search(r'^[^\(]+(?=\()', x) else None)
  
  # Remove params column (makes the export to csv weird because of the commas)
  list_cols = ["params"]
  all_dfs = all_dfs.drop(columns=list_cols)
  
  # Save the final full table
  all_dfs.to_csv(filename_cvResults)



def calibrateClassifier(fitted_final_classifier, validation_set, target_name, group_name, weight_name, cores, filename_finalCalibratedModel):
  """Function to calibrate a fitted classifier to get a probabilistic classifier"""
  
  cores = int(cores)
  
  # Split the validation set into X_valid, y_valid, groups and sample weights objects
  X_valid, y_valid, groups, sample_weights = splitTrainingSet(validation_set, target_name, group_name, weight_name)
  
  # Set the cross validation splitter
  cv = list(LeaveOneGroupOut().split(X_valid, y_valid, groups=groups))
  
  print("Calibrating classifier ...")
  calibration_start_time = time()
  
  # Calibration
  calibrated_classifier = CalibratedClassifierCV(
    estimator=FrozenEstimator(fitted_final_classifier), 
    method = 'isotonic', 
    cv = cv, 
    n_jobs = cores
  )
  
  fitted_calibrated_classifier = calibrated_classifier.fit(
    X = X_valid, 
    y = y_valid, 
    sample_weight = sample_weights
  )
  
  print("Done ! Saving calibrated classifier ...")
  
  joblib.dump(fitted_calibrated_classifier, filename_finalCalibratedModel)
  
  calibration_stop_time = time() - calibration_start_time
  print(f"Done ! Time elapsed : {calibration_stop_time} s \n")



def buildLearningCurve(n_sizes, lc_k, cores, filename_learningCurve, fitted_final_classifier, training_set, target_name, group_name, weight_name, rng):
  """This function builds and save the plot of a learning curve to assess that the final model doesn't under/over-fit"""
  
  # Make sure the numbers given as argument are integer
  lc_k = int(lc_k)
  cores = int(cores)
  n_sizes = int(n_sizes)
  
  # Split the training set into X_train, y_train, groups and sample weights objects
  X_train, y_train, groups, sample_weights = splitTrainingSet(training_set, target_name, group_name, weight_name)
  
  # Set the training set sizes to try
  train_sizes = np.linspace(0.1, 1.0, num = n_sizes, endpoint = True)
  
  # Set the cross validation splitter
  cv = LeaveOneGroupOut()
  
  # scoring function
  scorer = make_scorer(matthews_corrcoef).set_score_request(sample_weight=True)
  
  # Unlike for cross_validate, we don't need to enable metadata to be able to use the group argument for CV if needed so we can disable the metadata routing
  sklearn.set_config(enable_metadata_routing=False)
  
  # Build the learning curve
  display = LearningCurveDisplay.from_estimator(
    fitted_final_classifier, # classifier with a fit method
    X_train, # full training set
    y_train, # target
    train_sizes = train_sizes, # list of training set sizes to try
    cv = cv, # the CV strategy
    score_type = "both", # score both train and test scores to check under/overfitting
    scoring = scorer, # scoring method
    score_name = "Matthews Correlation Coefficient", # name of the scoring method as it will appear on the plot
    std_display_style = "errorbar", # display error bars
    n_jobs = cores, # number of cores to use
    groups = groups, # if the CV strategy uses groups
    sample_weight = sample_weights # if the CV strategy uses groups
  )
  
  # Make log axis and add title
  _ = display.ax_.set(xscale = "log", title = "Learning curve of the final classifier")
  
  # Save the plot
  plt.savefig(filename_learningCurve)


def getFinalResults(fitted_final_classifier):
  """Function to print the main results of the process which are the selected features and the final classifier"""
  
  # Get the selected features
  selected_features = fitted_final_classifier['selector'].get_feature_names_out()
  
  # Print the results
  print(f"The features selected with Mutual Information are : \n {selected_features} \n")
  print(f"The final Classifier is : \n {fitted_final_classifier}")


def buildSupervisedClassifier(training_set, validation_set, target_name, group_name, weight_name, select_K, cores, n_sizes, filename_cvResults, filename_finalFittedModel, filename_finalCalibratedModel, filename_learningCurve):
  """Function that runs the entire supervised classifier building process by calling the other custom functions"""
  # Set the random state for reproducibility
  rng = np.random.RandomState(42)
  
  # To be able to use groups and sample weights
  sklearn.set_config(enable_metadata_routing=True)
  
  # Run the nested cross-validation step
  outer_models, inner_models_halving, scores = applyNestedCrossValidation(rng, training_set, target_name, group_name, weight_name, cores, select_K)
  
  # Fit and save final model
  fitted_final_classifier = fitFinalClassifier(inner_models_halving, filename_finalFittedModel, training_set, target_name, group_name, weight_name)
  
  # Save the nested cross-validation detailed results
  saveNestedCvResults(outer_models, filename_cvResults, scores)
  
  # Print the main results
  getFinalResults(fitted_final_classifier)
  
  # Build and save the learning curve
  buildLearningCurve(n_sizes, cores, filename_learningCurve, fitted_final_classifier, training_set, target_name, group_name, weight_name, rng)
  
  # Calibrate the classifier
  calibrateClassifier(fitted_final_classifier, validation_set, target_name, group_name, weight_name, cores, filename_finalCalibratedModel)


########################## Apply Supervised Classifier #########################

def loadClassifier(model_path, classifier_name):
  """Function to load the saved final supervised classifier and everything needed for prediction"""
  
  # Load the model
  fitted_final_classifier = joblib.load(classifier_name)
  
  # Get unique class names
  classes = fitted_final_classifier['classifier'].classes_
  
  # Get the features used in the pipeline (before feature selection)
  features = fitted_final_classifier['selector'].feature_names_in_
  return fitted_final_classifier, classes, features


def predictPhyto(model_path, classifier_name, predict_name, data):
  """Function to predict phytoplankton class from a supervised classifier"""
  
  # Load classifier, unique class names and the features used in the pipeline (before feature selection)
  fitted_final_classifier, classes, features = loadClassifier(model_path, classifier_name)
  
  # Only keep the features used to fit the final model in the table to be predicted
  X = data[features]
  
  # Classify data, predict the labels and probabilities
  preds_test = fitted_final_classifier.predict(X)
  proba_predict = pd.DataFrame(fitted_final_classifier.predict_proba(X)) # compute class prediction probabilities and store in data frame
  
  predicted_data = data
  
  # Add prediction to original test table
  predicted_data['predicted_label'] = preds_test 
  
  # Make the column names of this data frame the class names (instead of numbers)
  proba_predict = proba_predict.set_axis(classes, axis=1)
  
  # Bind both data frames by column
  full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
  
  # Save final predicted table
  full_predicted.to_csv(predict_name)
  return preds_test


def comparePrediction(data, preds_test, target_name, weight_name, cm_filename, model_path, classifier_name):
  """Function to assess the generalization performance of the final model by comparing its predicted label of the test set to the manual labels"""
  
  # Load Classifier to get the class names
  fitted_final_classifier, classes, features = loadClassifier(model_path, classifier_name)
  
  # Get the manual labels
  y_test = data[target_name]
  sample_weights = data[weight_name]
  
  
  # Print the results of various metrics scores between manual and predicted labels
  print(f"Test Matthews Correlation Coefficient score = {matthews_corrcoef(y_test, preds_test)} \n")
  print(f"Test Balanced Accuracy score = {balanced_accuracy_score(y_test, preds_test)} \n")
  print(f"Classification report : \n {classification_report(y_test, preds_test, sample_weight = sample_weights)} \n", file=file)

  report = pd.DataFrame(classification_report(y_test, preds_test, output_dict=True, sample_weight = sample_weights))
  report['metric'] = ('precision', 'recall', 'f1-score', 'support')
  report.to_csv(report_filename, index=False)
  print("Classification report saved ! \n")
  
  # Create the confusion matrix between manual and predicted labels
  cm = pd.DataFrame(confusion_matrix(y_test, preds_test, labels = classes))
  
  # Set classes as column and row names 
  cm = cm.set_axis(classes, axis=1) 
  cm = cm.set_axis(classes, axis=0)
  
  # Save confusion matrix as csv
  cm.to_csv(cm_filename)


def predictTestSet(model_path, classifier_name, predict_name, data, target_name, weight_name, cm_filename):
  """Function to predict the test set specifically, calls two other custom function for prediction and comparison to manual labels"""
  preds_test = predictPhyto(model_path, classifier_name, predict_name, data)
  comparePrediction(data, preds_test, target_name, weight_name, report_filename, cm_filename, classifier_name, text_file)


def getPermutationImportance(data, nb_repeats, classifier_name, target_name, weight_name, cores, filename_importance):
  """Function to measure the permutation importance of the variables used in the final model"""
  
  # Load the model
  fitted_final_classifier, classes, features = loadClassifier(model_path, classifier_name)
  
  # Get the manual labels and the features table
  y = data[target_name]
  X = data[features]
  sample_weights = data[weight_name]
  
  # Set the random state for reproducibility
  rng = np.random.RandomState(42)
  
  # Make sure the number given as argument is integer
  cores = int(cores)
    
  # to be able to use sample weights
  sklearn.set_config(enable_metadata_routing=True)
  
  print("Computation of Permutation Importance... \n")
  
  # Compute the permutation importance calculation
  permutation = permutation_importance(estimator = fitted_final_classifier, X = X, y = y, scoring = scorer, n_repeats=nb_repeats, n_jobs = cores, random_state = rng, sample_weight = sample_weights)
  print("Done ! Saving results ...")
  
  # Extract and save the results
  sorted_importances_idx = permutation.importances_mean.argsort()
  importances = pd.DataFrame(permutation.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
  importances.to_csv(filename_importance)
  print("Permutation Importance results saved ! \n")


def getSelectedFeatures(model_path, classifier_name):
  """Function to extract the features selected with the feature selector of the pipeline"""
  
  # Load model
  fitted_final_classifier = joblib.load(classifier_name)
  
  # Get the features
  selected_features = fitted_final_classifier['selector'].get_feature_names_out()
  return selected_features
