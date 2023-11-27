# Azure Machine Learning - AutoML Example (v2 SDK_

#### Example notebook showcasing how to kick off an AutoML training job using the v2 SDK.

In this sample, we retrieve data from a publicly available source, register training & validation datasets to our AML workspace, then execute an AutoML training job. Upon completion of the AutoML run, we will retrieve the best performing model and run it locally using a `ManagedOnlineEndpoint`. Here, we can run our new model from a Docker container and can easily update our deployment to target cloud compute resources if we aim to make the model available to other users/applications/processes.

<i>Note:</i> This sample was adapted from the following examples:
- https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/automl-standalone-jobs/automl-regression-task-hardware-performance
- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python
