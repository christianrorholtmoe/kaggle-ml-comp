#runs the experiment
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.script_run import ScriptRun

ws = Workspace.from_config() #get the workspace

#xgboost experiment
exp_xgboost = Experiment(workspace= ws, name="sales_xgboost")
xgb_config = ScriptRunConfig(source_directory= "./src", script= "xgmodel.py", compute_target="cluster1" ) #set up the configuration of the experiment
run = exp_xgboost.submit(xgb_config) #submit the experiment
