#This script executes sets up on a Azure ML CPU cluster

#set up the required imports for Azure ML
from azureml.core import Workspace, workspace
from azureml.core import compute
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

#reference to the Azure ML workspace. 
ws = Workspace.from_config() #automatically looks for the .azureml folder

cpu_cluster_name = "cluster1"; #name of the compute cluster

#verify whether the cluster exisits. If not, create one
try:
    cpu_cluster = ComputeTarget(ws, cpu_cluster_name)
    print (cpu_cluster_name + " has already been provisioned at " + cpu_cluster.cluster_location + "- you are ready to start experimenting!")
except ComputeTargetException:
    print (cpu_cluster_name + " has NOT been provisioned - creating")
    compute_config = AmlCompute.provisioning_configuration(vm_size="Standard_D2_V2", vm_priority="lowpriority", max_nodes=2, idle_seconds_before_scaledown="60")
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

