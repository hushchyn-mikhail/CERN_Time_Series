# -*- coding: utf-8 -*-


# <codecell>

import ipykee
ipykee.create_project(project_name="E._LossFunctionOptimization", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="E._LossFunctionOptimization")

# <codecell>

import ipykee
#ipykee.create_project(project_name="E._LossFunctionOptimization", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="E._LossFunctionOptimization")

# <codecell>

session.commit("Empty. First commit.")