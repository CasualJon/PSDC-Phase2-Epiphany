#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can train the models consistently.

# This file contains functions for training models for the Challenge. You can run it as follows:
#
#   python train_model.py data model
#
# where 'data' is a folder containing the Challenge data and 'model' is a folder for saving your model.

import sys, subprocess
from helper_code import is_integer
from tuning_team_code import train_challenge_model
from optimization import optimize_threshold_eval
from run_model import run_model
from datetime import datetime
from evaluate_2024 import evaluate_model
import team_code

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    # Define the data and model foldes.
    data_folder = sys.argv[1]
    model_folder = sys.argv[2]
    output_folder = 'project_files/test_outputs'
    allow_failures = False

    # Change the level of verbosity; helpful for debugging.
    if len(sys.argv)==4 and is_integer(sys.argv[3]):
        verbose = int(sys.argv[3])
    else:
        verbose = 1

    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')

    result = subprocess.run(['bash', 'scripts/clean_data.sh'], capture_output=True, text=True, check=True)
    print(result.stdout)
    print(result.stderr)

    print('Running the model with best hyperparameters found in the tuning phase...')
    team_code.train_challenge_model(data_folder, model_folder, verbose)
    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
    try:
        evaluate_model('project_files/test_data/labels.csv', output_folder, 'project_files/test_outputs/inference_time.txt', 'threshold.txt',
                    'scale_params.json', 'factor_loadings.json', 'zscore_params.json', 'project_files/test_outputs/score.json')
    except Exception as e:
        print(f'Error during evaluation: {e}')
        print('Skipping evaluation due to error. If you want to evaluate the model, please run the evaluation script manually. CONTINUING...')

    result = subprocess.run(['bash', 'scripts/clean_data.sh'], capture_output=True, text=True, check=True)
    print(result.stdout)
    print(result.stderr)

    for iters in range (600, 1500, 200):
        for depth in range(3, 13):
            for importance in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
                for learning_rate in [0.08, 0.085, 0.09, 0.095, 0.1]:
                    print(f'Training model with iters={iters}, depth={depth}, importance={importance}, learning_rate={learning_rate}...')
                    train_challenge_model(data_folder, model_folder, verbose, iters, depth, importance, learning_rate)
                    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
                    optimization_run = optimize_threshold_eval(iters, depth, importance, learning_rate)
                    with open(f'test_outputs/optimization_results_{timestamp}.txt', 'a') as f:
                        f.write(f'Results for iters={iters}, depth={depth}, importance={importance}, learning_rate={learning_rate}:\n')
                        f.write(optimization_run + '\n\n')

                    result = subprocess.run(['bash', 'scripts/clean_data.sh'], capture_output=True, text=True, check=True)
                    print(result.stdout)
                    print(result.stderr)
