import argparse
import json
import os
import time


from azureml.core import Run

import mlflow
import mlflow.sklearn

from sklearn.metrics import r2_score

import mltable

# Based on example:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli
# which references
# https://github.com/Azure/azureml-examples/tree/main/cli/jobs/train/lightgbm/iris


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # # add arguments
    parser.add_argument("--model_input_path", type=str, help="Path to input model")
    parser.add_argument(
        "--model_base_name", type=str, help="Name of the registered model"
    )
    parser.add_argument(
        "--test_data", type=str, help="Hold out test data for evaluation"
    )
    parser.add_argument("--target_column", type=str, help="Target Column")
    parser.add_argument("--comparison_metrics", type=str, help="Output location to write comparison metrics (champion vs. challenger)")

    # parse args
    args = parser.parse_args()
    print("Path: " + args.model_input_path)

    return args


def main(args):
    """
    Register Model Example
    """
    # Set Tracking URI
    current_run = Run.get_context()
    current_experiment = current_run.experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)
    
    # Load previous model (champion)
    model_name = args.model_base_name
    model_version = str(Run.get_context().experiment.workspace.models.get(model_name).version)
    champion_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    
    # Load newly trained model (challenger)
    mlmodel_path = os.path.join(args.model_input_path, "MLmodel")
    runid = ""
    with open(mlmodel_path, "r") as modelfile:
        for line in modelfile:
            if "run_id" in line:
                runid = line.split(":")[1].strip()

    # Construct Model URI from run ID extract previously
    model_uri = "runs:/{}/outputs/mlflow-model".format(runid)
    challenger_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    
    # Get inferencing dataset
    print(args.test_data)
    tbl = mltable.load(args.test_data)
    test_df = tbl.to_pandas_dataframe()
    
    y_actual = test_df[args.target_column]
    
    try:
        test_df = test_df.drop(columns=[args.target_column])
    except Exception as e:
        pass
    
    champion_preds = champion_model.predict(test_df)
    challenger_preds = challenger_model.predict(test_df)
    
    champion_r2_score = r2_score(y_actual, champion_preds)
    challenger_r2_score = r2_score(y_actual, challenger_preds)
    
    with open(os.path.join(args.comparison_metrics, 'comparison_metrics.txt'), 'w') as f:
        f.write('Champion r2: ' + str(champion_r2_score))
        f.write('Challenger r2: ' + str(challenger_r2_score))
    
    # TURNED OFF FOR DEVELOPMENT PURPOSES
    # UNCOMMENT LINES BELOW FOR CHAMPION VS. CHALLENGER EVALAUTION
    # if challenger_r2_score < champion_r2_score:
    #     # Cancel the run, go no further
    #     print('New model does not perform better than existing model. Cancel run.')
    #     current_run.parent.cancel()
    
    
    
# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)