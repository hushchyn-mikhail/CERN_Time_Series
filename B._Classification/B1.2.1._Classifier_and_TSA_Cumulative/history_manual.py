# -*- coding: utf-8 -*-


# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

#Load original data
data_origin = pd.read_excel('../../popularity-728days_my.xls')
data_tsa = pd.read_csv('../../Cern_Time_Series/df_predict_labeled_1year_6month_cumulative_10_014.csv')
data = pd.concat([data_origin, data_tsa], axis = 1)
data.columns

# <codecell>

#Select data
selection = ((data['Now'] - data['Creation-week']) > 52)&((data['Now'] - data['FirstUsage']) > 52)&((data[52] - data[1]) != 0)
data_sel = data[selection].copy()

data_sel['slope_rel'] = data_sel.get(['slope'])/data_sel.get(['slope_std']).values
data_sel['y_0_rel'] = data_sel.get(['y_0'])/data_sel.get(['y_0_std']).values
data_sel['y_1_rel'] = data_sel.get(['y_1'])/data_sel.get(['y_1_std']).values
data_sel['y_f_rel'] = data_sel.get(['y_f'])/data_sel.get(['y_f_std']).values
#data_sel = data.copy()

# <codecell>

#Transform string features to digits
cols_str = ['Configuration', 'ProcessingPass', 'FileType', 'Storage']
df_str = data_sel.get(cols_str)

for col in cols_str:
    unique = np.unique(df_str[col])
    index = range(0, len(unique))
    mapping = dict(zip(unique, index))
    df_str = df_str.replace({col:mapping})

# <codecell>

#Get normed series
cols_series = range(1, 53)
df_series = data_sel.get(cols_series)
mins = df_series.min(axis = 1)
maxs = df_series.max(axis = 1)

for col in cols_series:
    df_series[col] = (df_series[col]-mins)/(maxs-mins+1)

    
#Add new features
first_usage = np.nan_to_num(df_series[df_series != 0].idxmin(axis = 1).values)
last_usage = np.nan_to_num(df_series[df_series > 0].idxmax(axis = 1).values)
interval_usage = last_usage - first_usage

df_series['First'] = first_usage
df_series['Last'] = last_usage
df_series['Interval'] = interval_usage

# <codecell>

#Get other features
cols_other = ['Type', 'Creation-week', 'NbLFN', 'LFNSize', 'NbDisk','DiskSize', 'NbTape', 'TapeSize',
              'NbArchived', 'ArchivedSize', 'Nb Replicas', 'Nb ArchReps','FirstUsage', 'LastUsage', 
              'Now','Neg_Prob', 'slope', 'slope_std', 'y_0', 'y_0_std', 'y_1', 'y_1_std',
                'y_f', 'y_f_std', 'LabelReal', 'slope_rel', 'y_0_rel', 'y_1_rel', 'y_f_rel']
df_other = data_sel.get(cols_other)

# <codecell>

#Label the data
labels = ((data_sel[104] - data_sel[52]) == 0)*1
labels.values

# <codecell>

#Concatenate all data sets
data_use = pd.concat([df_str,df_series,df_other], axis = 1)
data_sel.columns[70:]

# <codecell>

#Preparing signal and background data for classifier
data_sig = data_use[labels == 1]
data_bck = data_use[labels == 0]

#save signal and background data for classifier
data_sig.to_csv('../../Cern_Time_Series/Classification/data_sig_original_classifier_10_09.csv')
data_bck.to_csv('../../Cern_Time_Series/Classification/data_bck_original_classifier_10_09.csv')

# <codecell>

c = data_sig.columns
print c

# <codecell>

# Convert data to DataStorage
from cern_utils import converter_csv

#Load signal and background data
signal_data = converter_csv.load_from_csv('../../Cern_Time_Series/Classification/data_sig_original_classifier_10_09.csv', sep=',')
bck_data = converter_csv.load_from_csv('../../Cern_Time_Series/Classification/data_bck_original_classifier_10_09.csv', sep=',')

# <codecell>

# Get train and test data
signal_train, signal_test = signal_data.get_train_test(train_size=0.8)
bck_train, bck_test = bck_data.get_train_test(train_size=0.8)

# <codecell>

columns = signal_data.columns
print columns

#select variables for classifier
"""
variables = [u'Configuration', u'ProcessingPass', u'FileType', u'Storage', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', 
             u'10', u'11', u'12', u'13', u'14', u'15', u'16', u'17', u'18', u'19', u'20', u'21', u'22', u'23', u'24', u'25', 
             u'26', u'27', u'28', u'29', u'30', u'31', u'32', u'33', u'34', u'35', u'36', u'37', u'38', u'39', u'40', u'41', 
             u'42', u'43', u'44', u'45', u'46', u'47', u'48', u'49', u'50', u'51',u'52', u'Type', u'Creation-week', u'NbLFN', 
             u'LFNSize', u'NbDisk', u'DiskSize', u'NbTape', u'TapeSize', u'NbArchived', u'ArchivedSize', u'Nb Replicas', 
             u'Nb ArchReps', u'FirstUsage', u'LastUsage', u'Now']
"""
variables = [ u'Configuration', u'ProcessingPass', u'FileType', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', 
             u'10', u'11', u'12', u'13', u'14', u'15', u'16', u'17', u'18', u'19', u'20', u'21', u'22', u'23', u'24', u'25', 
             u'26', u'27', u'28', u'29', u'30', u'31', u'32', u'33', u'34', u'35', u'36', u'37', u'38', u'39', u'40', u'41', 
             u'42', u'43', u'44', u'45', u'46', u'47', u'48', u'49', u'50', u'51',u'52', 'First', 'Last', 'Interval','Type',
             u'Creation-week', u'NbLFN', u'LFNSize', u'NbDisk', u'DiskSize', u'NbTape', u'TapeSize', u'NbArchived', u'ArchivedSize',
             u'Nb Replicas', u'Nb ArchReps', u'FirstUsage', u'Neg_Prob', u'slope', u'slope_std', u'y_0', u'y_0_std', u'y_1',
             u'y_1_std',u'y_f', u'y_f_std', u'slope_rel', u'y_0_rel', u'y_1_rel', u'y_f_rel']

print variables

# <codecell>

from cern_utils import sklearn_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# configuring classifier
"""
classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                        min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                        max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
"""

classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini',
            max_depth=6, max_features=None, min_density=None,
            min_samples_leaf=10, min_samples_split=10, random_state=None,
            splitter='best'), n_estimators=200, learning_rate=0.1, algorithm='SAMME.R', random_state=None)


gbc = sklearn_classifier.ClassifierSklearn(base_classifier=GradientBoostingClassifier(n_estimators=1500, learning_rate=0.05,max_depth=6),
                                           directory='cern_time_series_classification_gbc/')

"""
gbc = sklearn_classifier.ClassifierSklearn(base_classifier=classifier,
                                           directory='cern_time_series_classification_gbc/')
"""

gbc.set_params(features=variables)

# training classifier
gbc.fit(signal_train, bck_train)

# <codecell>

# get prediction on data after classification
from cern_utils.predictions_report import PredictionsInfo
report = PredictionsInfo({'GBC': gbc}, signal_test, bck_test)

# <codecell>

#Plot importances of features according to trained model
importance = gbc.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def deviance(y_true, y_pred, sample_weight):
    return gbc.base_classifier.loss_(y_true, y_pred)

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred)  

def average_precision(y_true, y_pred, sample_weight):
    return average_precision_score(y_true, y_pred)  


report.learning_curve( { 'roc_auc':roc_auc, 'average_precision':average_precision}, steps=1).plot(figsize = (7,5))
plt.subplot(1,1,1)
report.learning_curve( {'devianse': deviance}, steps=1).plot(figsize = (7,5))
plt.subplot(1,1,1)

# <codecell>

#Plot learning curves on train
plt.plot(gbc.base_classifier.train_score_)
plt.title('Train Score')
plt.xlabel('Steps of boosting')
plt.ylabel('error')
plt.show()

# <codecell>

#Correlation matrix
report.features_correlation_matrix().plot(show_legend=False)

# <codecell>

#Features histogramms
hist_var = variables[:]
hist_var.remove(u'NbTape')
hist_var.remove(u'TapeSize')
report.features_pdf(features=hist_var, bins = 10).plot()

# <codecell>

#ROC - curve
report.roc().plot(xlim=(0, 1))

# <codecell>

# define metric functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


import numpy


def accuracy(s, b, t_s, t_b, s_NORM=1., b_NORM = 1.): 

    return (s + t_b - b)/(t_s + t_b)

def precision(s, b, t_s, t_b, s_NORM=1., b_NORM = 1.):
    return 1- b/t_b

report.metrics_vs_cut({'precision': precision, 'accuracy': accuracy}).plot(new_plot=True, figsize=(8, 4))

# <codecell>

report.prediction_pdf(bins = 20, normed = True, plot_type='bar').plot()

# <codecell>


def ReleaseMemory(cut = 0.9):

    signal_release = (report.prediction_sig['GBC'] >= cut)*1
    bck_release = (report.prediction_bck['GBC'] >= cut)*1

    released_memory = (signal_release*signal_test.get_data(['DiskSize']).values[:,0]).sum() + (bck_release*bck_test.get_data(['DiskSize']).values[:,0]).sum()
    good_memory = (signal_release*signal_test.get_data(['DiskSize']).values[:,0]).sum()

    part_of_good_memory = good_memory/released_memory*100
    
    return released_memory, good_memory, part_of_good_memory

def ReleaseMemoryPlot(mincut = 0.9, maxcut = 1, N = 100):
    step = (maxcut - mincut)/N
    cuts = [mincut + step*i for i in range(0, N+1)]
    
    released_memory = []
    good_memory = []
    part_of_good_memory = []
    
    all_memory = signal_test.get_data(['DiskSize']).values[:,0].sum() + bck_test.get_data(['DiskSize']).values[:,0].sum()
    memory_can_be_free = signal_test.get_data(['DiskSize']).values[:,0].sum()
    
    for i in cuts:
        rm, gm, pm = ReleaseMemory(cut = i)
        released_memory.append(rm)
        good_memory.append(gm)
        part_of_good_memory.append(pm)
    
    print 'all_memory = ', all_memory
    print 'memory_can_be_free = ', memory_can_be_free
    
    plt.subplot(1,1,1)
    plt.plot(cuts, released_memory, 'b', label = 'released memory')
    plt.plot(cuts, good_memory, 'r', label = 'good memory')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.subplot(1,1,1)
    plt.plot(cuts, part_of_good_memory, 'r', label = 'part of good memory')
    plt.legend(loc = 'best')
    plt.show()
       
        
ReleaseMemoryPlot(mincut = 0.5, maxcut = 1, N = 100)

# <codecell>

#Load original data
data_origin = pd.read_excel('../../popularity-728days_my.xls')
data_tsa2 = pd.read_csv('../../Cern_Time_Series/df_predict_labeled_6month_cumulative_10_08.csv')
data2 = pd.concat([data_origin, data_tsa2], axis = 1)
data.columns

# <codecell>

#Select data
selection2 = ((data2['Now'] - data2['Creation-week']) > 26)&((data2['Now'] - data2['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
data_sel2 = data2[selection2].copy()
#data_sel2 = data.copy()

data_sel2['slope_rel'] = data_sel2.get(['slope'])/data_sel2.get(['slope_std']).values
data_sel2['y_0_rel'] = data_sel2.get(['y_0'])/data_sel2.get(['y_0_std']).values
data_sel2['y_1_rel'] = data_sel2.get(['y_1'])/data_sel2.get(['y_1_std']).values
data_sel2['y_f_rel'] = data_sel2.get(['y_f'])/data_sel2.get(['y_f_std']).values

# <codecell>

#Transform string features to digits
df_str2 = data_sel2.get(cols_str)

for col in cols_str:
    unique = np.unique(df_str2[col])
    index = range(0, len(unique))
    mapping = dict(zip(unique, index))
    df_str2 = df_str2.replace({col:mapping})

# <codecell>

#Get normed series
cols_series2 = range(27, 79)
df_series2 = data_sel2.get(cols_series2)
mins2 = df_series2.min(axis = 1)
maxs2 = df_series2.max(axis = 1)

for col in cols_series2:
    df_series2[col] = (df_series2[col]-mins2)/(maxs2-mins2+1)
  
df_series2.columns = range(1,53)
df_series2.columns
    
#Add new features
first_usage2 = np.nan_to_num(df_series2[df_series2 != 0].idxmin(axis = 1).values)
last_usage2 = np.nan_to_num(df_series2[df_series2 > 0].idxmax(axis = 1).values)
interval_usage2 = last_usage2 - first_usage2

df_series2['First'] = first_usage2
df_series2['Last'] = last_usage2
df_series2['Interval'] = interval_usage2

# <codecell>

#df_series2.columns = range(1,53)
#df_series2.columns

# <codecell>

#Get other features
df_other2 = data_sel2.get(cols_other)

# <codecell>

#Label the data
labels2 = ((data_sel2[104] - data_sel2[78]) == 0)*1
labels2.values

# <codecell>

#Concatenate all data sets
data_use2 = pd.concat([df_str2,df_series2,df_other2], axis = 1)

# <codecell>

#Preparing signal and background data for classifier
data_sig2 = data_use2[labels2 == 1]
data_bck2 = data_use2[labels2 == 0]

#save signal and background data for classifier
data_sig2.to_csv('../../Cern_Time_Series/Classification/data_sig2_original_classifier_10_09.csv')
data_bck2.to_csv('../../Cern_Time_Series/Classification/data_bck2_original_classifier_10_09.csv')

# <codecell>

# Convert data to DataStorage
from cern_utils import converter_csv

#Load signal and background data
signal_data2 = converter_csv.load_from_csv('../../Cern_Time_Series/Classification/data_sig2_original_classifier_10_09.csv', sep=',')
bck_data2 = converter_csv.load_from_csv('../../Cern_Time_Series/Classification/data_bck2_original_classifier_10_09.csv', sep=',')

# <codecell>

# Get train and test data
signal_train2, signal_test2 = signal_data2.get_train_test(train_size=0.01)
bck_train2, bck_test2 = bck_data2.get_train_test(train_size=0.01)

# <codecell>

# get prediction on data after classification
from cern_utils.predictions_report import PredictionsInfo
report2 = PredictionsInfo({'GBC': gbc}, signal_test2, bck_test2)

# <codecell>

#Correlation matrix
report2.features_correlation_matrix().plot(show_legend=False)

# <codecell>

#Features histogramms
report2.features_pdf(features=hist_var, bins = 10).plot()

# <codecell>

#ROC - curve
report2.roc().plot(xlim=(0, 1))

# <codecell>

# define metric functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


report2.metrics_vs_cut({'precision': precision, 'accuracy': accuracy}).plot(new_plot=True, figsize=(8, 4))

# <codecell>

report2.prediction_pdf(bins = 20, normed = True, plot_type='bar').plot()

# <codecell>


def ReleaseMemory2(cut = 0.9):

    signal_release = (report2.prediction_sig['GBC'] >= cut)*1
    bck_release = (report2.prediction_bck['GBC'] >= cut)*1

    released_memory = (signal_release*signal_test2.get_data(['DiskSize']).values[:,0]).sum() + (bck_release*bck_test2.get_data(['DiskSize']).values[:,0]).sum()
    good_memory = (signal_release*signal_test2.get_data(['DiskSize']).values[:,0]).sum()

    part_of_good_memory = good_memory/released_memory*100
    
    return released_memory, good_memory, part_of_good_memory

def ReleaseMemoryPlot2(mincut = 0.9, maxcut = 1, N = 100):
    step = (maxcut - mincut)/N
    cuts = [mincut + step*i for i in range(0, N+1)]
    
    released_memory = []
    good_memory = []
    part_of_good_memory = []
    
    all_memory = signal_test2.get_data(['DiskSize']).values[:,0].sum() + bck_test2.get_data(['DiskSize']).values[:,0].sum()
    memory_can_be_free = signal_test2.get_data(['DiskSize']).values[:,0].sum()
    
    for i in cuts:
        rm, gm, pm = ReleaseMemory2(cut = i)
        released_memory.append(rm)
        good_memory.append(gm)
        part_of_good_memory.append(pm)
    
    print 'all_memory = ', all_memory
    print 'memory_can_be_free = ', memory_can_be_free
    
    plt.subplot(1,1,1)
    plt.plot(cuts, released_memory, 'b', label = 'released memory')
    plt.plot(cuts, good_memory, 'r', label = 'good memory')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.subplot(1,1,1)
    plt.plot(cuts, part_of_good_memory, 'r', label = 'part of good memory')
    plt.legend(loc = 'best')
    plt.show()
       
        
ReleaseMemoryPlot2(mincut = 0.5, maxcut = 1, N = 100)

# <codecell>

import ipykee
#ipykee.create_project("B._Classification", internal_path="B._Classification", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="B._Classification")

# <codecell>

session.add(report2.roc(), "report2.roc()")

session.add(report2.prediction_sig['GBC'], "report2.prediction_sig['GBC']")

session.add(report2.prediction_bck['GBC'], "report2.prediction_bck['GBC']")

session.add(report2.prediction_pdf(bins = 20, normed = True, plot_type='bar'), "report2.prediction_pdf()")

a=1
session.add(a, "test")

# <codecell>

session.add(report2.roc(), "report2.roc()")

session.add(report2.prediction_sig['GBC'], "report2.prediction_sig['GBC']")

session.add(report2.prediction_bck['GBC'], "report2.prediction_bck['GBC']")

session.add(report2.prediction_pdf(bins = 20, normed = True, plot_type='bar'), "report2.prediction_pdf()")

a=1
session.add(a, "test")

# <codecell>

session.commit("Variables added")