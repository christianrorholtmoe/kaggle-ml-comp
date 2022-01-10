#runs the experiment
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.script_run import ScriptRun

ws = Workspace.from_config() #get the workspace

#xgboost experiment
exp_xgboost = Experiment(workspace= ws, name="sales_xgboost")
config = ScriptRunConfig(source_directory= "./src", script= "xgmodel.py", compute_target="cluster1" ) #set up the configuration of the experiment
run = exp_xgboost.submit(config) #submit the experiment
