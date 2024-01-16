import argparse
import json
import os
import time


from azureml.core import Run

from azure.ai.ml import MLClient, Input, automl, Output
from azure.ai.ml import command, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandJob, CommandJobLimits, Data, AmlCompute, Model, ModelPackage, CodeConfiguration, AzureMLOnlineInferencingServer, ManagedOnlineEndpoint, ManagedOnlineDeployment, DataCollector, DeploymentCollection, Environment
from azure.ai.ml.constants import AssetTypes  
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
    Update Managed Online Endpoint (Real-Time)
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
    rt_endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    
    deployment_name = 'blue'

    if 'blue' in rt_endpoint.traffic.keys():
        if rt_endpoint.traffic['blue'] > 0:
            deployment_name = 'green'


    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=args.endpoint_name,
        model=model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        data_collector=DataCollector({'model_inputs': DeploymentCollection(enabled='true'), 
                                  'model_outputs': DeploymentCollection(enabled='true'), 
                                  'model_inputs_outputs':DeploymentCollection(enabled='true')}, rolling_rate= 'hour', sampling_rate=1.0)
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    rt_endpoint.traffic = {deployment_name: 100}
    ml_client.begin_create_or_update(rt_endpoint).result()

    

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
