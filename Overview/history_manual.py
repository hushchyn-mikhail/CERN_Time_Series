# -*- coding: utf-8 -*-


# <codecell>

%pylab inline
from cern_utils import calc_util
import ipykee

# <codecell>

#get variables
keeper = ipykee.Keeper("A._TimeSeriesAnalysis")
session = keeper["A1.3.1._TSA_Cumulative_52weeks"]
vars_a131 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("A._TimeSeriesAnalysis")
session = keeper["A1.3.2._TSA_Cumulative_78weeks"]
vars_a132 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("A._TimeSeriesAnalysis")
session = keeper["A2.3.1._TSA_RollingMean26_52weeks"]
vars_a231 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("A._TimeSeriesAnalysis")
session = keeper["A2.3.2._TSA_RollingMean26_78weeks"]
vars_a232 = session.get_variables("master")
#variables.keys()

# <codecell>

#ROC - curve
figure(figsize=(15, 5))
subplot(121)
vars_a131['report.roc()'].plot()
vars_a132['report.roc()'].plot()
vars_a231['report.roc()'].plot()
vars_a232['report.roc()'].plot()
legend(['A1.3.1', 'A1.3.2', 'A2.3.1', 'A2.3.2'], loc='best')
subplot(122)
vars_a131['report2.roc()'].plot()
vars_a132['report2.roc()'].plot()
vars_a231['report2.roc()'].plot()
vars_a232['report2.roc()'].plot()
legend(['A1.3.1', 'A1.3.2', 'A2.3.1', 'A2.3.2'], loc='best')

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B1.1.1._Classifier_Cumulative"]
vars_b111 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B1.2.1._Classifier_and_TSA_Cumulative"]
vars_b121 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B1.2.2._Classifier_and_TSA_Cumulative_Grouped"]
vars_b122 = session.get_variables("master")
#variables.keys()

# <codecell>

#ROC - curve
figure(figsize=(8, 5))
vars_b111['report2.roc()'].plot()
vars_b121['report2.roc()'].plot()
vars_b122['report2.roc()'].plot()
vars_a131['report.roc()'].plot()
vars_a132['report.roc()'].plot()
legend(['B1.1.1','B1.2.1', 'B1.2.2', 'B1.3.1(A1.3.1)', 'B1.3.2(A1.3.2)'], loc='best')

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B2.1.1._Classifier_Rolling_Mean"]
vars_b211 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B2.2.1._Classifier_and_TSA_RollingMean"]
vars_b221 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B2.2.2._Classifier_and_TSA_RollinMean_Grouped"]
vars_b222 = session.get_variables("master")
#variables.keys()

# <codecell>

#ROC - curve
figure(figsize=(8, 5))
vars_b211['report2.roc()'].plot()
vars_b221['report2.roc()'].plot()
vars_b222['report2.roc()'].plot()
vars_a231['report.roc()'].plot()
vars_a232['report.roc()'].plot()
legend(['B2.1.1','B2.2.1', 'B2.2.2', 'B2.3.1(A2.3.1)', 'B2.3.2(A2.3.2)'], loc='best')

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B3.2.1._NewFeatures_52weeks"]
vars_b321 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("B._Classification")
session = keeper["B3.2.2._NewFeatures_78weeks"]
vars_b322 = session.get_variables("master")
#variables.keys()

# <codecell>

#ROC - curve
figure(figsize=(15, 5))
subplot(121)
vars_a131['report.roc()'].plot()
vars_a132['report.roc()'].plot()
vars_a231['report.roc()'].plot()
vars_a232['report.roc()'].plot()
vars_b321['report.roc()'].plot()
vars_b322['report.roc()'].plot()
legend(['A1.3.1(B1.3.1)', 'A1.3.2(B1.3.2)', 'A2.3.1(B2.3.2)', 'A2.3.2(B2.3.2)', 'B3.2.1', 'B3.2.2'], loc='best')
subplot(122)
vars_a131['report2.roc()'].plot()
vars_a132['report2.roc()'].plot()
vars_a231['report2.roc()'].plot()
vars_a232['report2.roc()'].plot()
vars_b321['report.roc()'].plot()
vars_b322['report2.roc()'].plot()
legend(['A1.3.1(B1.3.1)', 'A1.3.2(B1.3.2)', 'A2.3.1(B2.3.2)', 'A2.3.2(B2.3.2)', 'B3.2.1', 'B3.2.2'], loc='best')

# <codecell>

#get variables
keeper = ipykee.Keeper("C._NewFeatures")
session = keeper["C1.1._NewFeatures_78weeks"]
vars_c11 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("C._NewFeatures")
session = keeper["C2.1.1._RelativeNewFeatures_78weeks"]
vars_c21 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("C._NewFeatures")
session = keeper["C3.1.1._RelativeNewFeatures_NormedNbOfUsages_78weeks"]
vars_c311 = session.get_variables("master")
#variables.keys()

# <codecell>

#get variables
keeper = ipykee.Keeper("C._NewFeatures")
session = keeper["C3.1.2._NewFeatures_NormedNbOfUsages_78weeks"]
vars_c312 = session.get_variables("master")
#variables.keys()

# <codecell>

#ROC - curve
figure(figsize=(15, 5))
subplot(121)
vars_c11['report.roc()'].plot()
vars_c21['report.roc()'].plot()
vars_c311['report.roc()'].plot()
vars_c312['report.roc()'].plot()
legend(['C1.1(B3.2.2)', 'C2.1.1', 'C3.1.1', 'C3.1.2'], loc='best')
subplot(122)
vars_c11['report2.roc()'].plot()
vars_c21['report2.roc()'].plot()
vars_c311['report2.roc()'].plot()
vars_c312['report2.roc()'].plot()
legend(['C1.1(B3.2.2)', 'C2.1.1', 'C3.1.1', 'C3.1.2'], loc='best')

# <codecell>

import ipykee
ipykee.create_project("Overview", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="Overview")

# <codecell>

import ipykee
#ipykee.create_project("Overview", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="Overview")

# <codecell>

session.commit("First commit")