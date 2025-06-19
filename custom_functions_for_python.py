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
from sklearn.model_selection import LearningCurveDisplay, RandomizedSearchCV, GridSearchCV, cross_validate, LeaveOneGroupOut, StratifiedGroupKFold, learning_curve, HalvingRandomSearchCV, ValidationCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold
import glob
import joblib
import datetime
################################################################################
############################# Custom Functions #################################
################################################################################

######################## Supervised Classifier Building ########################

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
  # Extra line to drop svm which seems to be slow and unlikely to ever be the preferred model anyway:
  parameters = [p for p in parameters if not any(isinstance(v, list) and any('SVC' in str(clf) for clf in v) for k, v in p.items() if k == 'classifier')]  

  #This is what lucinda has, but I have not honored the groups aspect (1 cyz file = 1 group):
  #inner_cv = LeaveOneGroupOut()
  #outer_cv = LeaveOneGroupOut()
  # Possibly because Number of Splits for this method will equal to number of unique groups as in another script i define groups as df.index. That works to simply ignore groups, with this StratifiedGroupKFold instead of LeaveOneGroupOut():
  inner_cv = StratifiedGroupKFold(n_splits=2, shuffle = True, random_state = 42)
  outer_cv = StratifiedGroupKFold(n_splits=2, shuffle = True, random_state = 42)
  
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
  
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  filename_finalCalibratedModel = os.path.join(os.path.dirname(filename_finalCalibratedModel), f"final_model_{timestamp}.probabilistic_pkl")
  
  joblib.dump(fitted_calibrated_classifier, filename_finalCalibratedModel)
  
  calibration_stop_time = time() - calibration_start_time
  print(f"Done ! Time elapsed : {calibration_stop_time} s \n")




def buildLearningCurve(n_sizes, cores, filename_learningCurve, fitted_final_classifier, training_set, target_name, group_name, weight_name, rng):
  """This function builds and save the plot of a learning curve to assess that the final model doesn't under/over-fit"""
  
  # Make sure the numbers given as argument are integer
  cores = int(cores)
  n_sizes = int(n_sizes)
  
  # Split the training set into X_train, y_train, groups and sample weights objects
  X_train, y_train, groups, sample_weights = splitTrainingSet(training_set, target_name, group_name, weight_name)
  
  # Set the training set sizes to try
  train_sizes = np.linspace(0.1, 1.0, num = n_sizes, endpoint = True)
  print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
  print(X_train)

  # Set the cross validation splitter
  #cv = StratifiedGroupKFold(n_splits=2, shuffle = True, random_state = rng)
  cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=rng)

  # scoring function
  scorer = make_scorer(matthews_corrcoef).set_score_request(sample_weight=True)
  
  print("Building and saving learning curve results... \n")
  learning_start_time = time()
  
  train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
    estimator = fitted_final_classifier, # classifier with a fit method
    X = X_train, # full training set
    y = y_train, # target
    train_sizes = train_sizes, # list of training set sizes to try
    cv = cv, # the CV strategy
    scoring = scorer, # scoring method
    n_jobs = cores, # number of cores to use
    verbose = 10, 
    shuffle = True, # shuffling training data before taking prefixes of it based on train_sizes
    random_state = rng, # random state for shuffling
    error_score = 'raise', # raise error if error in scoring when fitting the estimator
    return_times = True,
    #groups = groups, # return scoring and fitting times
    #params = {"sample_weight": sample_weights, "groups": groups} # metadata params passed to fit and the scorer
  )
  
  results = pd.DataFrame({
    'train_sizes': train_sizes,
    'train_scores_mean': np.mean(train_scores, axis=1),
    'train_scores_std': np.std(train_scores, axis=1),
    'test_scores_mean': np.mean(test_scores, axis=1),
    'test_scores_std': np.std(test_scores, axis=1),
    'fit_times_mean' : np.mean(fit_times, axis=1),
    'fit_times_std' : np.std(fit_times, axis=1),
    'score_times_mean' : np.mean(score_times, axis=1),
    'score_times_std' : np.std(score_times, axis=1)
    })
  
  results.to_csv(filename_learningCurve, index=False)
  
  learning_stop_time = time() - learning_start_time
  print(f"Done ! Time elapsed : {learning_stop_time} s \n")

def getFinalResults(fitted_final_classifier):
  """Function to print the main results of the process which are the selected features and the final classifier"""
  
  # Get the selected features
  selected_features = fitted_final_classifier['selector'].get_feature_names_out()
  
  # Print the results
  print(f"The features selected with Mutual Information are : \n {selected_features} \n")
  print(f"The final Classifier is : \n {fitted_final_classifier}")


def buildSupervisedClassifier(training_set, validation_set, target_name, group_name, weight_name, select_K, cores, n_sizes, filename_cvResults, filename_finalFittedModel, filename_finalCalibratedModel, filename_learningCurve, filename_importance, plots_dir):
  """Function that runs the entire supervised classifier building process by calling the other custom functions"""
  # Set the random state for reproducibility
  rng = np.random.RandomState(42)
  
  # To be able to use groups and sample weights
  sklearn.set_config(enable_metadata_routing=True)
  print("training_set")
  print(training_set)
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


  # Compute permutation importance
  scorer = make_scorer(matthews_corrcoef).set_score_request(sample_weight=True)
  getPermutationImportance(
    data=training_set,
    nb_repeats=10,
    target_name="source_label",
    weight_name="weight",
    cores=cores,
    filename_importance=filename_importance,
    model_path=filename_finalFittedModel,
    scorer=scorer,
    plots_path = plots_dir
  )  
  # Calibrate the classifier
  calibrateClassifier(fitted_final_classifier, validation_set, target_name, group_name, weight_name, cores, filename_finalCalibratedModel)


########################## Apply Supervised Classifier #########################

def loadClassifier(model_dir):
  """Function to load the saved final supervised classifier and everything needed for prediction"""
  
  model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
  if not model_files:
    raise FileNotFoundError("No model files found.")
  latest_model = max(model_files, key=os.path.getmtime)
  print('loding latest model:')
  print(latest_model)
  # Load the latest model
  fitted_final_classifier = joblib.load(latest_model)
  
  # Get unique class names
  classes = fitted_final_classifier['classifier'].classes_
  
  # Get the features used in the pipeline (before feature selection)
  features = fitted_final_classifier['selector'].feature_names_in_
  return fitted_final_classifier, classes, features


def predictPhyto(model_path, predict_name, data):
  """Function to predict phytoplankton class from a supervised classifier"""
  
  # Load classifier, unique class names and the features used in the pipeline (before feature selection)
  fitted_final_classifier, classes, features = loadClassifier(os.path.dirname(model_path))
  
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
  return preds_test, full_predicted


def comparePrediction(data, preds_test, target_name, weight_name, report_filename, cm_filename, model_path):
  """Function to assess the generalization performance of the final model by comparing its predicted label of the test set to the manual labels"""
  
  # Load Classifier to get the class names
  fitted_final_classifier, classes, features = loadClassifier(os.path.dirname(model_path))
  
  # Get the manual labels
  y_test = data[target_name]
  sample_weights = data[weight_name]
  
  
  # Print the results of various metrics scores between manual and predicted labels
  print(f"Test Matthews Correlation Coefficient score = {matthews_corrcoef(y_test, preds_test)} \n")
  print(f"Test Balanced Accuracy score = {balanced_accuracy_score(y_test, preds_test)} \n")
  print(f"Classification report : \n {classification_report(y_test, preds_test, sample_weight = sample_weights)} \n")

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

def predictTestSet(self, model_path, predict_name, data, target_name, weight_name, cm_filename, report_filename, text_file):
    preds_test, predicted_df = predictPhyto(model_path, predict_name, data)
    if self is not None:
        self.df = predicted_df
    comparePrediction(data, preds_test, target_name, weight_name, report_filename, cm_filename, model_path)


def getPermutationImportance(data, nb_repeats, target_name, weight_name, cores, filename_importance, model_path, scorer, plots_path):
  """Function to measure the permutation importance of the variables used in the final model"""
  
  # Load the model
  fitted_final_classifier, classes, features = loadClassifier(os.path.dirname(model_path))
  
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

  df_t = importances.T
  df_t.columns = [f"repeat_{i}" for i in range(df_t.shape[1])]
  df_t["mean_importance"] = df_t.mean(axis=1)
  df_t["std_importance"] = df_t.std(axis=1)

  # Sort and select top 10
  top_features = df_t.sort_values(by="mean_importance", ascending=False).head(10)

  # Plot
  plt.figure(figsize=(10, 8))
  plt.barh(top_features.index, top_features["mean_importance"], xerr=top_features["std_importance"], color="skyblue")
  plt.xlabel("Mean Permutation Importance")
  plt.title("Top 10 Features by Permutation Importance")
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plot_filename = plots_path+"top10.png"
  plt.savefig(plot_filename)
  #plt.show()



def getSelectedFeatures(model_path):
  """Function to extract the features selected with the feature selector of the pipeline"""
  
  # Load model
  fitted_final_classifier = joblib.load(model_path)
  
  # Get the features
  selected_features = fitted_final_classifier['selector'].get_feature_names_out()
  return selected_features
