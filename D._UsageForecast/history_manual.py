# -*- coding: utf-8 -*-


# <codecell>

import pandas as pd
res = pd.read_csv('Total report.csv')
res

# <codecell>

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

# <codecell>

session.commit("TOTAL REPORT 2.")