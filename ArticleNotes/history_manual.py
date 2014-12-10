# -*- coding: utf-8 -*-


# <codecell>

import ipykee
ipykee.create_project("Articles", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="Articles")

# <codecell>

session.commit("First commit")