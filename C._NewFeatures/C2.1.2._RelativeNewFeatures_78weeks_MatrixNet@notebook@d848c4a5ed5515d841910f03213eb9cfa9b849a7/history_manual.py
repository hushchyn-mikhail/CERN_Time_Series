# -*- coding: utf-8 -*-


# <codecell>

import ipykee
#ipykee.create_project("C._NewFeatures", internal_path="C._NewFeatures", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="C._NewFeatures")

# <codecell>

session.commit("Upload by ipykee. First commit.")