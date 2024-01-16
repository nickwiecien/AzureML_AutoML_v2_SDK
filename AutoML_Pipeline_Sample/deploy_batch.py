import argparse
import json
import os
import time

from azureml.core import Run

from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.entities import BatchEndpoint, ModelBatchDeployment, ModelBatchDeploymentSettings, PipelineComponentBatchDeployment, Model, AmlCompute, Data, BatchRetrySettings, CodeConfiguration, Environment, Data
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# Based on example:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli
# which references
# https://github.com/Azure/azureml-examples/tree/main/cli/jobs/train/lightgbm/iris


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # # add arguments
    parser.add_argument(
        "--model_base_name", type=str, help="Name of the registered model"
    )
    parser.add_argument(
        "--endpoint_name", type=str, help="Name of the Azure ML endpoint"
    )
    parser.add_argument(
        "--registered_model_details", type=str, help="Details about the registered model"
    )
    
    # parse args
    args = parser.parse_args()

    return args


def main(args):
    """
    Update Managed Online Endpoint (Batch)
    """
    # Get pointer to workspace
    current_run = Run.get_context()
    current_experiment = current_run.experiment
    ws = current_experiment.workspace
    
    # Connect to ml_client
    ml_client = MLClient(credential=DefaultAzureCredential(), subscription_id=ws.subscription_id, resource_group_name=ws.resource_group, workspace_name=ws.name)
    
    
    # Load champion model
    model_name = args.model_base_name
    model = ml_client.models.get(model_name, label='latest')
    
    # Get pointer to endpoint
    batch_endpoint = ml_client.batch_endpoints.get(args.endpoint_name)
    
    from datetime import datetime

    timestamp = datetime.now().strftime("%m%d%H%M%f")

    batch_deployment_name = f'battery-batch-{timestamp}'

    deployment = ModelBatchDeployment(
        name=batch_deployment_name,
        description="Batch Endpoint for Battery Cycle Prediction model",
        endpoint_name=batch_endpoint.name,
        model=model,
        compute='cpu-cluster',
        settings=ModelBatchDeploymentSettings(
            instance_count=2,
            max_concurrency_per_instance=2,
            mini_batch_size=10,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name="predictions.csv",
            retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
            logging_level="info",
        ),
    )

    ml_client.batch_deployments.begin_create_or_update(deployment).result()
   
    
# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)