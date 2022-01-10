

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

#xtreme gradient boosting
from xgboost import XGBClassifier 


#azure imports
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.script_run import ScriptRun
from azureml.core import Dataset

#workspace
ws = Workspace.from_config() #get the workspace

#get the training and test data
data = Dataset.get_by_name(ws, name='training data')
data.to_pandas_dataframe()


#divide data into train and evaluation set
x_train, x_test, y_train, y_test = train_test_split(data.drop_columns("Id", "row_id", "num_sold"), data["num_sold"], test_size=0.2)

evaluation_set = [(x_test, y_test)]


#set up the XGB model w initial hyperparameters. TODO: Add HyperDrive
xg_model = XGBClassifier(max_depth=10, learning_rate=0.02, n_estimators=1000, verbosity=1,
                         objective='multi:softmax', booster='gbtree', 
                         tree_method='auto', n_jobs=1, gamma=0, 
                         min_child_weight=1, max_delta_step=0, subsample=1,
                         colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                         base_score=0.5, random_state=0, missing=None,
                         num_parallel_tree=1, importance_type='gain')


#train the model
xg_model.fit(x_train,y_train, eval_metric="mlogloss", eval_set=evaluation_set, verbose=True, early_stopping_rounds=10)

print ("Finished training xgb model...")

#register the model


