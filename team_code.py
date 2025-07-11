#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.


##################################################################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
##################################################################################################################################

from helper_code import *
# import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from os import makedirs, path
from joblib import dump, load

##################################################################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


##################################################################################################################################
# def __load_challenge_data():
# Override to exclude the disallowed features
def __load_challenge_data(data_folder):
    data = pd.read_csv(data_folder)

    label = data['inhospital_mortality'] if 'inhospital_mortality' in data.columns else pd.NA
    patient_ids = data['studyid_adm'] if 'studyid_adm' in data.columns else pd.NA

    remove_features =  [
        feature for feature in ['studyid_adm', 'inhospital_mortality'] + [f'admitabx_adm___{i}' for i in range(1, 22)] if feature in data.columns
    ]
    data = data.drop(remove_features, axis=1)

    features = data.columns
    return patient_ids, data, label, features
# END def __load_challenge_data():


##################################################################################################################################
# Train your model.
#
# def train_challenge_model():
def train_challenge_model(data_folder, model_folder, verbose) -> None:
    # Find the Challenge data
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    # Use override loader to exclude disallowed features
    patient_ids, data, label, features = __load_challenge_data(data_folder)
    if verbose >= 1:
        print(f'Training data includes {data.shape[1]} features across {data.shape[0]} samples.')

    for i in range(2):
        # Identify categorical features (CatBoost handles these natively)
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if verbose >= 1:
            print(f'Categorical features detected: {categorical_features}')

        # Convert categorical features to string due to CatBoost requirements
        for col in categorical_features:
            data[col] = data[col].astype(str)

        # Define CatBoost classifier parameters
        training_verbosity = 100 if verbose else 0
        model = CatBoostClassifier(
            iterations=1300,                        # Number of iterations
            depth=3,                                # Depth of each tree to prevent overfitting
            learning_rate=0.09,                     # Step size of udpates
            loss_function='Logloss',                # Binary classification loss (Y/N in predicting mortality)
            eval_metric='AUC',                      # Evaluation on Area Under Curve
            cat_features=categorical_features,      # Identify the categorical features
            verbose=training_verbosity,             # Training progress output per X iterations
            task_type='CPU',                        # Use CPU for training
            random_seed=27,                         # Seed to ensure reproducibility
        )

        if verbose >= 1:
            print('Training the CatBoost model...')

        model.fit(data, label, cat_features=categorical_features, verbose=training_verbosity)

        if i == 0:
            if verbose >= 1:
                print('Extracting feature importance...')

            importances = model.get_feature_importance(prettified=True)
            # Debian Training tuning identified that features with importance >= 0.5 achieve best results
            important_features = importances[importances['Importances'] >= 0.5]['Feature Id'].tolist()

            if verbose >= 1:
                print(f'Identified {len(important_features)} important features (>=0.5). Retraining model.')

            data = data[important_features]
            features = data.columns

    save_challenge_model(model_folder, model, features, verbose)

    if verbose >= 1:
        print('Exiting train_challenge_model()')
# END def train_challenge_model()



##################################################################################################################################
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
#
# def load_challenge_model():
def load_challenge_model(model_folder, verbose):
    # Check for the model first
    model_path = path.join(model_folder, 'catboost_model.pkl')
    if not path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    if verbose >= 1:
        print(f'Loading model from {model_path}...')

    model_bundle = load(model_path)
    # Unpack the model and selected features (embedded as custom metadata to the model) 
    # Workaround to run_challenge_model not having access to the model_folder
    model = model_bundle[0]
    model._feature_names_used = model_bundle[1]
    return model
# END def load_challenge_model()



##################################################################################################################################
# def run_challenge_model():
def run_challenge_model(model, data_folder, verbose):
    # Load test data
    patient_ids, data, _, _ = __load_challenge_data(data_folder)
    selected_features = getattr(model, '_feature_names_used', None)
    if selected_features is None:
        raise RuntimeError('Model does not contain embedded features data.')

    data = data[selected_features]

    if verbose >= 1:
        print(f'Running inference on {data.shape[0]} samples, {data.shape[1]} features.')

    # Categorical features must match training format
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        data[col] = data[col].astype(str)

    if verbose >= 1:
        print(f'Categorical features detected: {categorical_features}')

    # Make predictions...
    # Extract probability of class 1 (mortality)
    prediction_probabilities = model.predict_proba(data)[:, 1]

    try:
        with open('threshold.txt', 'r') as f:
            threshold = float(f.read().strip())
    except Exception as e:
        if verbose >= 1:
            print(f'Warning: threshold.txt not found or invalid. Using default threshold 0.003943. Error: {e}')
        threshold = 0.003943

    # Threshold at 0.5 for binary classification
    prediction_binary = (prediction_probabilities >= threshold).astype(int)

    if verbose >= 1:
        print('Predictions completed.')

    return patient_ids, prediction_binary, prediction_probabilities
# END def run_challenge_model()


##################################################################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


##################################################################################################################################
# Save your trained model.
# 
# def save_challenge_model():
def save_challenge_model(model_folder, model, features, verbose):
    if verbose >= 1:
        print('Model training complete. Saving model...')

    # Ensure model directory exists
    makedirs(model_folder, exist_ok=True)

    # Save the trained model
    try:
        dump((model, list(features)), path.join(model_folder, 'catboost_model.pkl'))
        with open(path.join(model_folder, 'selected_variables.txt'), 'w') as f:
            for feature in features:
                f.write(f'{feature}\n')

        if verbose >= 1:
            print('Model saved successfully.')
    
    except Exception as e:
        print(f'Error saving model: {e}')
        raise e
# END def save_challenge_model()