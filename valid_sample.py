import pandas as pd
import numpy
import sklearn
import subprocess
import os
import csv
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import tspkg.code.tslm_v1 as tslm


# In[6]:


def find_repo_path(repo_name):
    # Open GitHub Desktop and navigate to the repository
    os.startfile('GitHub')
    os.startfile(f'x-github-client://openRepo/{repo_name}')

    # Wait for GitHub Desktop to open and retrieve the repository path
    while True:
        repo_path = os.path.expanduser(f'~/Documents/GitHub/{repo_name}')
        if os.path.exists(repo_path):
            break

    return repo_path


# In[86]:


def validate_tslm(data_nm="aus_production", features="Trend/Season", season_num=4, y="Beer", result="result"):
    repo_name = "tspkg"
    repo_path = find_repo_path(repo_name)
    data_path = repo_path + "/" + "Data/" + data_nm + ".csv"
    r_path = repo_path + "/" + "R/rlib_test.R"

    data = pd.read_csv(data_path)

    if result is False:

        train_mart = tslm.mk_train_mart(y=data[[y]], trend=True, freq=4)
        tslm_model = tslm.tslm(y=data[[y]], trend=True, freq=season_num)
        result = tslm.predict(fit_obj=tslm_model, train_mart=train_mart, trend=True, freq=4, h=len(train_mart))

    else:
        result_path = repo_path + "/" + "Result/" + result + ".csv"
        with open(result_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            result = []
            for row in csvreader:
                result.append(float(row[0]))

    info_dict = {'season_num': season_num,
                 'y': y,
                 'features': features}

    r = robjects.r
    pandas2ri.activate()
    r_data = pandas2ri.py2rpy(data)
    r_result = robjects.FloatVector(result)
    r_info = robjects.ListVector(dict(info_dict))
    robjects.globalenv['data'] = r_data
    robjects.globalenv['result'] = r_result
    robjects.globalenv['info'] = r_info

    r['source'](r_path)
