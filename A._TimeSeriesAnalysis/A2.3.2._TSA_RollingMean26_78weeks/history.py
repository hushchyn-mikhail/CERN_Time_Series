# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

#Load original data
data = pd.read_excel('../../popularity-728days_my.xls')
data.columns

# <codecell>

#Load results of forecast of rolling mean time series
data_res = pd.read_csv('../../Cern_Time_Series/df_predict_labeled_6month_rolling_mean_10_08.csv')
data_res.columns

# <codecell>

#Select data
selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
data_res_sel = data_res[selection].copy()
#data_sel = data.copy()

data_res_sel.shape

# <codecell>

#Data modification
data_res_sel['slope_rel'] = data_res_sel.get(['slope'])/data_res_sel.get(['slope_std']).values
data_res_sel['y_0_rel'] = data_res_sel.get(['y_0'])/data_res_sel.get(['y_0_std']).values
data_res_sel['y_1_rel'] = data_res_sel.get(['y_1'])/data_res_sel.get(['y_1_std']).values
data_res_sel['y_f_rel'] = data_res_sel.get(['y_f'])/data_res_sel.get(['y_f_std']).values
data_res_sel['DiskSize'] = data[selection].get('DiskSize').values
data_res_sel['LFNSize'] = data[selection].get('LFNSize').values
data_res_sel['Nb Replicas'] = data[selection].get('Nb Replicas').values
data_res_sel.columns

# <codecell>

#Preparing signal and background data for classifier
data_sig = data_res_sel[data_res_sel.LabelReal == 1]
data_bck = data_res_sel[data_res_sel.LabelReal == 0]

#save signal and background data for classifier
data_sig.to_csv('../../Cern_Time_Series/Classification/data_sig_res_1year_6month_cumulative_10_15.csv')
data_bck.to_csv('../../Cern_Time_Series/Classification/data_bck_res_1year_6month_cumulative_10_15.csv')

# <codecell>

#Histogramms of the TSA results
import matplotlib.pyplot as plt
#pd.options.display.mpl_style = 'default'

var_plot = [u'Neg_Prob', u'slope', u'slope_std', u'y_0', u'y_0_std', u'y_1', u'y_1_std',
            u'y_f', u'y_f_std', u'LabelReal', u'slope_rel', u'y_0_rel', u'y_1_rel', u'y_f_rel', u'DiskSize']
for i in var_plot:
    plt.subplot(1,1,1)
    
    a = np.percentile(data_res_sel[i].values, 0.5)
    if i == 'Neg_Prob' :
        b = np.percentile(data_res_sel[i].values, 100)
    else:
        b = np.percentile(data_res_sel[i].values, 90)
        
    plt.hist(data_sig[i].values, bins = 20, label='signal', alpha = 0.4, color = 'r',range= (a,b))
    plt.hist(data_bck[i].values, bins = 20, label='bck', alpha = 0.4, color = 'b', range= (a,b))
    plt.title(i)
    plt.legend(loc = 'best')
    plt.show()

# <codecell>

# Convert data to DataStorage
from cern_utils import converter_csv

#Load signal and background data
signal_data = converter_csv.load_from_csv('../../Cern_Time_Series/Classification/data_sig_res_1year_6month_cumulative_10_15.csv', sep=',')
bck_data = converter_csv.load_from_csv('../../Cern_Time_Series/Classification/data_bck_res_1year_6month_cumulative_10_15.csv', sep=',')

# <codecell>

# Get train and test data
signal_train, signal_test = signal_data.get_train_test(train_size=0.5)
bck_train, bck_test = bck_data.get_train_test(train_size=0.5)

# <codecell>

columns = signal_data.columns
print columns

#select variables for classifier
variables = [ 'Neg_Prob', 'slope', 'slope_std', 'y_0', 'y_0_std', 'y_1', 'y_1_std', 'y_f', 'y_f_std',
            'slope_rel', 'y_0_rel', 'y_1_rel', 'y_f_rel']
print variables

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from cern_utils import xgboost_classifier


# setup parameters for xgboost
param = {}
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = 1
param['eta'] = 0.1
param['max_depth'] = 2
param['eval_metric'] = 'map'
param['silent'] = 1
param['nthread'] = 16
param['min_child_weight'] = 1
param['subsample'] = 1
param['colsample_bytree'] = 1
param['base_score'] = 0.5
#param['num_feature'] = 10

# you can directly throw param in, though we want to watch multiple metrics here 
plst = list(param.items()) + [('eval_metric', 'map'), ('eval_metric', 'auc')]

xgboost = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
xgboost.set_params(features = variables, params = plst)
#setup additional parameters
xgboost.num_boost_round = 500
xgboost.watch = False

#trainig classifier
xgboost.fit(signal_train, bck_train)#,\
            #weight_sig=signal_train.get_data(['total_usage']).values,\
            #weight_bck=bck_train.get_data(['total_usage']).values)

# <codecell>

# get prediction on data after classification
from cern_utils.predictions_report import PredictionsInfo
report = PredictionsInfo({'xgboost': xgboost}, signal_test, bck_test)
report_train = PredictionsInfo({'xgboost': xgboost}, signal_train, bck_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
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


report.learning_curve( { 'roc_auc':roc_auc, 'average_precision':average_precision}, steps=10).plot(figsize = (7,5))

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
%pylab inline
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred)  

def average_precision(y_true, y_pred, sample_weight):
    return average_precision_score(y_true, y_pred)  

figure(figsize=(10, 6))
lc_test = report.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
av_test = report.learning_curve( {  'average_precision(test)':average_precision}, steps=10)
av_train = report_train.learning_curve( {  'average_precision(train)':average_precision}, steps=10)
av_test.plots[0].plot()
av_train.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
ll_test = report.learning_curve( { 'log_loss(test)':roc_auc}, steps=10)
ll_train = report_train.learning_curve( { 'log_loss(train)':roc_auc}, steps=10)
ll_test.plots[0].plot()
ll_train.plots[0].plot()
legend( loc='best')

# <codecell>

# get prediction on data after classification
figure(figsize=(10, 6))
report.prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train.prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')

# <codecell>

#Correlation matrix
#report.features_correlation_matrix().plot(show_legend=False, figsize=(15,5))

# <codecell>

#report.features_pdf(bins = 10).plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

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

#Normed signal
%pylab inline
from cern_utils import calc_util
iron = calc_util.classifier_flatten(report.prediction_sig['xgboost'])

_ = hist(iron(report.prediction_sig['xgboost']),  histtype='bar', bins=20, alpha=0.5, label='signal')
_ = hist(iron(report.prediction_bck['xgboost']),  histtype='bar', bins=20, alpha=0.5, label='bck')
legend(loc='best')

# <codecell>

from cern_utils import calc_util

def CondSize(report, signal_test, bck_test, classifier='xgboost', cut=0.6, peaks=5, imax=26):

    iron = calc_util.classifier_flatten(report.prediction_sig[classifier])
    
    cond_sig = (iron(report.prediction_sig[classifier]) < cut)\
    &(signal_test.get_data(['nb_peaks']).values<=peaks)[:,0]\
    &(signal_test.get_data(['inter_max']).values>=imax)[:,0]
    
    cond_bck = (iron(report.prediction_bck[classifier]) < cut)\
    &(bck_test.get_data(['nb_peaks']).values<=peaks)[:,0]\
    &(bck_test.get_data(['inter_max']).values>=imax)[:,0]

    nzrs = (signal_test.get_data(['Nb Replicas']).values >= 1)[:,0]
    nzrb = (bck_test.get_data(['Nb Replicas']).values >= 1)[:,0]

    sz_signal=signal_test.get_data(['DiskSize'])[(cond_sig)&nzrs].values.sum()\
    +bck_test.get_data(['DiskSize'])[(cond_bck)&nzrb].values.sum()\
    -signal_test.get_data(['LFNSize'])[(cond_sig)&nzrs].values.sum()\
    -bck_test.get_data(['LFNSize'])[(cond_bck)&nzrb].values.sum()

    return sz_signal

def RFiles(report, signal_test, bck_test, classifier='xgboost', mincut=0.01, maxcut=1, N=100, pq=95):
    print "Total number of the 'signal' files is ", signal_test.get_indices().shape[0]
    print "Total number of files is ", signal_test.get_indices().shape[0]+bck_test.get_indices().shape[0]
    
    step = (maxcut - mincut)/N
    cuts = [mincut + step*i for i in range(0, N+1)]
    
    iron = calc_util.classifier_flatten(report.prediction_sig[classifier])
    x=cuts
    
    nb_signals = []
    nb_true_signals = []
    nb_rels = []
    cut_pq = 1
    
    for i in cuts:
        nb_signal=((iron(report.prediction[classifier]) >= i)*1).sum()
        nb_true_signal=((iron(report.prediction_sig[classifier]) >= i)*1).sum()
        
        if nb_signal!=0:
            nb_rel=float(nb_true_signal)/float(nb_signal)*100
        else:
            nb_rel=100
        
        if cut_pq==1 and nb_rel>=pq:
            cut_pq=i
        
        nb_signals.append(nb_signal)
        nb_true_signals.append(nb_true_signal)
        nb_rels.append(nb_rel)

    
    plt.figure(figsize=(5, 3))
    plt.subplot(1,1,1)
    plt.plot(x, nb_signals, 'b', label = 'nb signal files')
    plt.plot(x, nb_true_signals, 'r', label = 'nb true signal files')
    plt.legend(loc = 'best')
    plt.show()
    
    plt.figure(figsize=(5, 3))
    plt.subplot(1,1,1)
    plt.plot(x, nb_rels, 'r', label = 'ration of the true signals to signals(%)')
    plt.legend(loc = 'best')
    plt.show()
    
    return cut_pq
    
def RSize(report, signal_test, bck_test, classifier='xgboost', mincut=0.01, maxcut=1, N=100, cond=0.9, Flag=False, pq=95):
    print "Total memory can be released is ", signal_test.get_data(['DiskSize']).values.sum()
    print "Total memory is ", signal_test.get_data(['DiskSize']).values.sum()+bck_test.get_data(['DiskSize']).values.sum()
    
    step = (maxcut - mincut)/N
    cuts = [mincut + step*i for i in range(0, N+1)]
    
    iron = calc_util.classifier_flatten(report.prediction_sig[classifier])
    x=cuts
    
    sz_signals = []
    sz_true_signals = []
    sz_rels = []
    cut_pq = 1
    
    nzrs = (signal_test.get_data(['Nb Replicas']).values >= 1)[:,0]
    nzrb = (bck_test.get_data(['Nb Replicas']).values >= 1)[:,0]
    
    for i in cuts:
        if i>=cond:
            sz_signal=signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= i)].values.sum()\
            +bck_test.get_data(['DiskSize'])[(iron(report.prediction_bck[classifier]) >= i)].values.sum()
            
            sz_true_signal=signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= i)].values.sum()
            
            if sz_signal!=0:
                sz_rel=float(sz_true_signal)/float(sz_signal)*100.
            else:
                sz_rel=100
                
            if cut_pq==1 and sz_rel>=pq:
                cut_pq=i
        else:
            sz_signal=signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= i)&nzrs].values.sum()\
            +bck_test.get_data(['DiskSize'])[(iron(report.prediction_bck[classifier]) >= i)&nzrb].values.sum()\
            -signal_test.get_data(['LFNSize'])[(iron(report.prediction_sig[classifier]) >= i)&nzrs].values.sum()\
            -bck_test.get_data(['LFNSize'])[(iron(report.prediction_bck[classifier]) >= i)&nzrb].values.sum()
            
            sz_true_signal=signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= i)&nzrs].values.sum()\
            -signal_test.get_data(['LFNSize'])[(iron(report.prediction_sig[classifier]) >= i)&nzrs].values.sum()
            
            if sz_signal!=0:
                sz_rel=float(sz_true_signal)/float(sz_signal)*100.
            else:
                sz_rel=100

            if cut_pq==1 and sz_rel>=pq:
                cut_pq=i

        sz_signals.append(sz_signal)
        sz_true_signals.append(sz_true_signal)
        sz_rels.append(sz_rel)

    
    if Flag==True:
        plt.figure(figsize=(5, 3))
        plt.subplot(1,1,1)
        plt.plot(x, sz_signals, 'b', label = 'signal files size')
        plt.plot(x, sz_true_signals, 'r', label = 'true signal files size')
        plt.legend(loc = 'best')
        plt.show()
    
        plt.figure(figsize=(5, 3))
        plt.subplot(1,1,1)
        plt.plot(x, sz_rels, 'r')
        plt.title('ratio of the true signals to signals(%)')
        plt.legend(loc = 'best')
        plt.show()
    else:
        plt.figure(figsize=(5, 3))
        plt.subplot(1,1,1)
        plt.plot(x, sz_signals, 'b', label = 'released memory')
        plt.legend(loc = 'best')
        plt.show()
        
    return cut_pq

# <codecell>

#CondSize(report, signal_test, bck_test, classifier='GBC', cut=0.6, peaks=5, imax=26)

# <codecell>

cut_pq1 = RSize(report, signal_test, bck_test, classifier='xgboost', mincut=0.1, maxcut=1, N=1000, cond=0, Flag=True)
print "cut_pq is ", cut_pq1

# <codecell>

RSize(report, signal_test, bck_test, classifier='xgboost', mincut=0.01, maxcut=1, N=1000, cond=1.1)

# <codecell>

cut_pq11 = RFiles(report, signal_test, bck_test, classifier='xgboost', mincut=0.001, maxcut=1, N=1000)
print "cut_pq11 is ", cut_pq11

# <codecell>

xgboost2 = xgboost_classifier.ClassifierXGBoost(directory='xgboost2/')
xgboost2.set_params(features = variables, params = plst)
#setup additional parameters
xgboost2.num_boost_round = 500
xgboost2.watch = False

#trainig classifier
xgboost2.fit(signal_test, bck_test)#,\
            #weight_sig=signal_train.get_data(['total_usage']).values,\
            #weight_bck=bck_train.get_data(['total_usage']).values)

# <codecell>

# get prediction on data after classification
from cern_utils.predictions_report import PredictionsInfo
report2 = PredictionsInfo({'xgboost2': xgboost2}, signal_train, bck_train)
report_train2 = PredictionsInfo({'xgboost2': xgboost2}, signal_test, bck_test)

# <codecell>

#Plot importances of features according to trained model
importance2 = xgboost2.get_feature_importance()
importance2.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred)  

def average_precision(y_true, y_pred, sample_weight):
    return average_precision_score(y_true, y_pred)  

figure(figsize=(10, 6))
lc_test2 = report2.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train2 = report_train2.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test2.plots[0].plot()
lc_train2.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
av_test2 = report2.learning_curve( {  'average_precision(test)':average_precision}, steps=10)
av_train2 = report_train2.learning_curve( {  'average_precision(train)':average_precision}, steps=10)
av_test2.plots[0].plot()
av_train2.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
ll_test2 = report2.learning_curve( { 'log_loss(test)':roc_auc}, steps=10)
ll_train2 = report_train2.learning_curve( { 'log_loss(train)':roc_auc}, steps=10)
ll_test2.plots[0].plot()
ll_train2.plots[0].plot()
legend( loc='best')

# <codecell>

# get prediction on data after classification
figure(figsize=(10, 6))
report2.prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train2.prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report2.roc().plot()
report_train2.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

# define metric functions
report2.metrics_vs_cut({'precision': precision, 'accuracy': accuracy}).plot(new_plot=True, figsize=(8, 4))

# <codecell>

figure(figsize=(10, 6))
report2.prediction_pdf(bins = 20, normed = True, plot_type='bar').plot()

# <codecell>

#Normed signal
%pylab inline
from cern_utils import calc_util
iron2 = calc_util.classifier_flatten(report2.prediction_sig['xgboost2'])

_ = hist(iron2(report2.prediction_sig['xgboost2']),  histtype='bar', bins=20, alpha=0.5, label='signal')
_ = hist(iron2(report2.prediction_bck['xgboost2']),  histtype='bar', bins=20, alpha=0.5, label='bck')
legend(loc='best')

# <codecell>

cut_pq2 = RSize(report2, signal_train, bck_train, classifier='xgboost2', mincut=0.2, maxcut=1, N=1000, cond=0, Flag=True)
print "cut_pq is ", cut_pq2

# <codecell>

RSize(report2, signal_train, bck_train, classifier='xgboost2', mincut=0.01, maxcut=1, N=1000, cond=1.1)

# <codecell>

cut_pq22 = RFiles(report2, signal_train, bck_train, classifier='xgboost2', mincut=0.001, maxcut=1, N=1000)
print "cut_pq is ", cut_pq22

# <codecell>

#Selection
sel = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) == 0)
check = ((data[sel][104] - data[sel][52]) == 0).values*1
memory1 = data[sel].get('DiskSize').values.sum()
memory_check = (data[sel].get('DiskSize').values*check).sum()
print check.sum()-check.shape[0]
print memory1
print memory_check

# <codecell>

#Classification. Brave
iron2 = calc_util.classifier_flatten(report2.prediction_sig['xgboost2'])

memory21 = signal_train.get_data(['DiskSize'])[iron2(report2.prediction_sig['xgboost2']) >= cut_pq2].values.sum()\
+bck_train.get_data(['DiskSize'])[iron2(report2.prediction_bck['xgboost2']) >= cut_pq2].values.sum()\
+signal_test.get_data(['DiskSize'])[iron(report.prediction_sig['xgboost']) >= cut_pq1].values.sum()\
+bck_test.get_data(['DiskSize'])[iron(report.prediction_bck['xgboost']) >= cut_pq1].values.sum()

print memory21

# <codecell>

#Classification. Safe

nzrs = (signal_test.get_data(['Nb Replicas']).values >= 1)[:,0]
nzrb = (bck_test.get_data(['Nb Replicas']).values >= 1)[:,0]
nzrs2 = (signal_train.get_data(['Nb Replicas']).values >= 1)[:,0]
nzrb2 = (bck_train.get_data(['Nb Replicas']).values >= 1)[:,0]
s_cut = 0.7

memory22 = signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig['xgboost']) >= s_cut)&nzrs].values.sum()\
+bck_test.get_data(['DiskSize'])[(iron(report.prediction_bck['xgboost']) >= s_cut)&nzrb].values.sum()\
-signal_test.get_data(['LFNSize'])[(iron(report.prediction_sig['xgboost']) >= s_cut)&nzrs].values.sum()\
-bck_test.get_data(['LFNSize'])[(iron(report.prediction_bck['xgboost']) >= s_cut)&nzrb].values.sum()\
+signal_train.get_data(['DiskSize'])[(iron2(report2.prediction_sig['xgboost2']) >= s_cut)&nzrs2].values.sum()\
+bck_train.get_data(['DiskSize'])[(iron2(report2.prediction_bck['xgboost2']) >= s_cut)&nzrb2].values.sum()\
-signal_train.get_data(['LFNSize'])[(iron2(report2.prediction_sig['xgboost2']) >= s_cut)&nzrs2].values.sum()\
-bck_train.get_data(['LFNSize'])[(iron2(report2.prediction_bck['xgboost2']) >= s_cut)&nzrb2].values.sum()

print memory22

# <codecell>

#Classification. Combination

memory231 = signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig['xgboost']) >= s_cut)&nzrs].values.sum()\
+bck_test.get_data(['DiskSize'])[(iron(report.prediction_bck['xgboost']) >= s_cut)&nzrb].values.sum()\
-signal_test.get_data(['LFNSize'])[(iron(report.prediction_sig['xgboost']) >= s_cut)&nzrs].values.sum()\
-bck_test.get_data(['LFNSize'])[(iron(report.prediction_bck['xgboost']) >= s_cut)&nzrb].values.sum()\
+signal_train.get_data(['DiskSize'])[(iron2(report2.prediction_sig['xgboost2']) >= s_cut)&nzrs2].values.sum()\
+bck_train.get_data(['DiskSize'])[(iron2(report2.prediction_bck['xgboost2']) >= s_cut)&nzrb2].values.sum()\
-signal_train.get_data(['LFNSize'])[(iron2(report2.prediction_sig['xgboost2']) >= s_cut)&nzrs2].values.sum()\
-bck_train.get_data(['LFNSize'])[(iron2(report2.prediction_bck['xgboost2']) >= s_cut)&nzrb2].values.sum()

memory232 = signal_test.get_data(['DiskSize'])[(iron(report.prediction_sig['xgboost']) >= cut_pq1)&nzrs].values.sum()\
+bck_test.get_data(['DiskSize'])[(iron(report.prediction_bck['xgboost']) >= cut_pq1)&nzrb].values.sum()\
-signal_test.get_data(['LFNSize'])[(iron(report.prediction_sig['xgboost']) >= cut_pq1)&nzrs].values.sum()\
-bck_test.get_data(['LFNSize'])[(iron(report.prediction_bck['xgboost']) >= cut_pq1)&nzrb].values.sum()\
+signal_train.get_data(['DiskSize'])[(iron2(report2.prediction_sig['xgboost2']) >= cut_pq2)&nzrs2].values.sum()\
+bck_train.get_data(['DiskSize'])[(iron2(report2.prediction_bck['xgboost2']) >= cut_pq2)&nzrb2].values.sum()\
-signal_train.get_data(['LFNSize'])[(iron2(report2.prediction_sig['xgboost2']) >= cut_pq2)&nzrs2].values.sum()\
-bck_train.get_data(['LFNSize'])[(iron2(report2.prediction_bck['xgboost2']) >= cut_pq2)&nzrb2].values.sum()

memory233 = signal_test.get_data(['DiskSize'])[iron(report.prediction_sig['xgboost']) >= cut_pq1].values.sum()\
+bck_test.get_data(['DiskSize'])[iron(report.prediction_bck['xgboost']) >= cut_pq1].values.sum()\
+signal_train.get_data(['DiskSize'])[iron2(report2.prediction_sig['xgboost2']) >= cut_pq2].values.sum()\
+bck_train.get_data(['DiskSize'])[iron2(report2.prediction_bck['xgboost2']) >= cut_pq2].values.sum()

memory23 = memory231-memory232+memory233
print memory23

# <codecell>

#Rare usage
memory3 = 0#CondSize(report2, signal_train, bck_train, classifier='xgboost2', cut=s_cut, peaks=3, imax=26)\
#+CondSize(report, signal_test, bck_test, classifier='xgboost', cut=s_cut, peaks=3, imax=26)
print memory3

# <codecell>

#Total released memory
memory = memory1+memory22+memory3
total = signal_train.get_data(['DiskSize']).values.sum()+bck_train.get_data(['DiskSize']).values.sum()+memory1\
+signal_test.get_data(['DiskSize']).values.sum()+bck_test.get_data(['DiskSize']).values.sum()
can_released = signal_train.get_data(['DiskSize']).values.sum()+signal_test.get_data(['DiskSize']).values.sum()+memory1

print "memory is ", memory
print "total memory is ", total
print "memory can be released is ", can_released
print "Ratio is ", float(memory)/float(total)*100

# <codecell>

#Total released memory
memory = memory1+memory21+memory3

print "memory is ", memory
print "total memory is ", total
print "memory can be released is ", can_released
print "Ratio is ", float(memory)/float(total)*100

# <codecell>

#Total released memory
memory = memory1+memory23+memory3

print "memory is ", memory
print "total memory is ", total
print "memory can be released is ", can_released
print "Ratio is ", float(memory)/float(total)*100

# <codecell>

import ipykee
#ipykee.create_project("A._TimeSeriesAnalysis", internal_path="A._TimeSeriesAnalysis", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="A._TimeSeriesAnalysis")

# <codecell>

session.add(report.roc(), "report.roc()")
session.add(report2.roc(), "report2.roc()")
session.add(report_train.roc(), "report_train.roc()")
session.add(report_train2.roc(), "report_train2.roc()")

session.add(report.prediction_sig['xgboost'], "report.prediction_sig['xgboost2']")
session.add(report2.prediction_sig['xgboost2'], "report2.prediction_sig['xgboost2']")

session.add(report.prediction_bck['xgboost'], "report.prediction_bck['xgboost2']")
session.add(report2.prediction_bck['xgboost2'], "report2.prediction_bck['xgboost2']")

session.add(report.prediction_pdf(bins = 20, normed = True, plot_type='bar'), "report.prediction_pdf()")
session.add(report2.prediction_pdf(bins = 20, normed = True, plot_type='bar'), "report2.prediction_pdf()")

a=1
session.add(a, "test")

