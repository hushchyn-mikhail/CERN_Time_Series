# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
%pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load original data
data = pd.read_excel('popularity-728days_my.xls')
data.columns

# <codecell>

#Select data
selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)&(data['FirstUsage']!=0)
data_sel = data[selection].copy()
#data_sel = data.copy()
print data_sel.shape

# <codecell>

periods = range(1,79)

#------------------------------------------------------
#Get maximum intervals and last weeks with zeros usages
def InterMax(data_sel, periods):
    #Get binary vector representation of the selected data
    data_bv = data_sel.copy()
    #Get week's usages
    for i in periods:
        if i!=1:
            data_bv[i] = data_sel[i] - data_sel[i-1]
            
    #Get binary representation
    data_bv[periods] = (data_bv[periods] != 0)*1
    
    inter_max = []
    last_zeros = []
    nb_peaks = []
    inter_mean = []
    inter_std = []
    inter_rel = []
    
    for i in range(0,data_bv.shape[0]):
        ds = data_bv[periods].irow(i)
        nz = ds.nonzero()[0]
        inter = []
        
        nb_peaks.append(len(nz))
        if len(nz)==0:
            nz = [0]
        if len(nz)<2:
            inter = [0]
            #nz = [0]
        else:
            for k in range(0, len(nz)-1):
                val = nz[k+1]-nz[k]
                inter.append(val)
        
        inter = np.array(inter)
        inter_mean.append(inter.mean())
        inter_std.append(inter.std())
        if inter.mean()!=0:
            inter_rel.append(inter.std()/inter.mean())
        else:
            inter_rel.append(0)
                
        last_zeros.append(nz[-1]+data_bv['Now'].values[i]-104-data_bv['Creation-week'].values[i])
        inter_max.append(max(inter))
    
    return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)
#------------------------------------------------------
def MassCenter(data_sel, periods):
    data_bv = data_sel.copy()
    p = np.array(periods)
    #Get week's usages
    for i in periods:
        if i!=1:
            data_bv[i] = data_sel[i] - data_sel[i-1]
    max_values = data_bv[periods].max(axis=1)
    for i in periods:
        data_bv[i] = (data_bv[i]/max_values).values
    
    mass_center = []
    mass_center2 = []
    mass_moment = []
    r_moment = []
            
    for i in range(0,data_bv.shape[0]):
        center = (data_bv[periods].irow(i).values*(p+data_bv['Now'].values[i]-104-data_bv['Creation-week'].values[i])).sum()/data_bv[periods].irow(i).values.sum()
        center2 = (data_bv[periods].irow(i).values*np.square((p+data_bv['Now'].values[i]-104-data_bv['Creation-week'].values[i]))).sum()
        moment = (data_bv[periods].irow(i).values*np.square((p+data_bv['Now'].values[i]-104-data_bv['Creation-week'].values[i])-center)).sum()
        r_m = moment/data_bv[periods].irow(i).values.sum()
        mass_center.append(center)
        mass_center2.append(center2)
        mass_moment.append(moment)
        r_moment.append(r_m)
    
    print data_bv.shape
    print data_sel.shape
        
    return np.array(mass_center), np.array(mass_moment), np.array(r_moment), np.array(mass_center2)
#------------------------------------------------------
def Binary(data_sel, periods):
    data_bv = data_sel.copy()
    p = np.array(periods) - periods[0]+1
    #Get week's usages
    for i in periods:
        if i!=1:
            data_bv[i] = data_sel[i] - data_sel[i-1]
    #Get binary representation
    data_bv[p] = (data_bv[periods] != 0)*1
    
    return data_bv[p]

# <codecell>

#Create a Data Frame for Classifier
df = pd.DataFrame()

#Add features
inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = InterMax(data_sel, periods)
df['last-zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel

mass_center, mass_moment, r_moment, mass_center2 = MassCenter(data_sel, periods)
df['mass_center'] = mass_center
df['mass_center_sqr'] = mass_center2
df['mass_moment'] = mass_moment
df['r_moment'] = r_moment

# <codecell>

df['DiskSize'] = data_sel['DiskSize'].values
df['LFNSize'] = data_sel['LFNSize'].values
df['Nb Replicas'] = data_sel['Nb Replicas'].values

df['LogDiskSize'] = np.log(data_sel['DiskSize'].values+0.00001)
df['total_usage'] = data_sel[periods[-1]].values
df['mean_usage'] = df['total_usage'].values/(df['nb_peaks'].values+1)

df['log_total_usage'] = np.log(data_sel[periods[-1]].values+1)
df['log_mean_usage'] = df['total_usage'].values - np.log(df['nb_peaks'].values+1)

"""
df['log_mass_center'] = np.log(mass_center+1)
df['log_mass_moment'] = np.log(mass_moment+1)
df['log_r_moment'] = np.log(r_moment+1)
df['log_mass_center_sqr'] = np.log(mass_center2+1)

df['log_last-zeros'] = np.log(last_zeros+1)
df['log_inter_max'] = np.log(inter_max+1)
df['log_nb_peaks'] = np.log(nb_peaks+1)
df['log_inter_mean'] = np.log(inter_mean+1)
df['log_inter_std'] = np.log(inter_std+1)
df['log_inter_rel'] = np.log(inter_rel+1)

#df['FileType'] = data_sel['FileType']
#df['Configuration'] = data_sel['Configuration']
#df['ProcessingPass'] = data_sel['ProcessingPass']
"""

# <codecell>

#Transform string features to digits
cols_str = ['Configuration', 'ProcessingPass', 'FileType', 'Storage']
df_str = data_sel.get(cols_str)

for col in cols_str:
    unique = np.unique(df_str[col])
    index = range(0, len(unique))
    mapping = dict(zip(unique, index))
    df_str = df_str.replace({col:mapping})
    
df['FileType'] = df_str['FileType'].values
df['Configuration'] = df_str['Configuration'].values
df['ProcessingPass'] = df_str['ProcessingPass'].values

other_vars = [u'Type', u'Creation-week', u'NbLFN', u'LFNSize', u'NbDisk', u'DiskSize', u'NbTape', u'TapeSize',
              u'NbArchived', u'ArchivedSize', u'Nb Replicas', u'Nb ArchReps', u'FirstUsage']
for i in other_vars:
    df[i] = data_sel[i].values
df['silence'] = df['FirstUsage']-df[u'Creation-week']

# <codecell>

#Add Binary Vector
"""
bv = Binary(data_sel, periods)
p = np.array(periods) - periods[0]+1
for i in p:
    df[i]=bv[i].values
"""

# <codecell>

#get noremd week's usages
data_b = data_sel.copy()
for i in periods:
    if i!=1:
        data_b[i] = data_sel[i] - data_sel[i-1]
max_values = data_b[periods].max(axis=1)

#add weekly usages transformed to [0,1] range of values
for i in periods:
    df[str(i)] = (data_b[i]/max_values).values
#add periods in string form    
periods_txt = []
for i in periods:
    periods_txt.append(str(i))

#all weeks were divided into several bins.
bins = []
for i in range(0, periods[-1]//13):
    cur_bin = data_b[range(i*13+1, (i+1)*13+1)]
    df["bin"+str(i)] = cur_bin.mean(axis=1).values
    bins.append("bin"+str(i))

# <codecell>

y_true = ((data_sel[104] - data_sel[78]) == 0).values*1

# <codecell>

for i in df.columns:
    plt.subplot(1,1,1)
    plt.hist(df[i][y_true == 1].values, label='signal', color='b', alpha=0.5, bins = 20)
    plt.hist(df[i][y_true == 0].values, label='bck', color='r', alpha=0.5, bins = 20)
    plt.ylabel('Nb of data sets')
    plt.legend(loc = 'best')
    plt.title(i)
    plt.show()

# <codecell>

#After additional selection
"""
for i in df_s.columns:
    plt.subplot(1,1,1)
    plt.hist(df_s[i][y_true_s == 1].values, label='1', color='b', alpha=0.5, bins = 20)
    plt.hist(df_s[i][y_true_s == 0].values, label='0', color='r', alpha=0.5, bins = 20)
    plt.legend(loc = 'best')
    plt.title(i)
    plt.show()
"""

# <codecell>

#
"""
df = df_s
y_true = y_true_s
df.shape
"""

# <codecell>

#new names of the columns
"""
cols_txt = list(df.columns[0:32])
periods_txt=[]
for i in df.columns[32:]:
    periods_txt.append(str(i))
new_cols = cols_txt + periods_txt
df_new = pd.DataFrame(data=df.values, columns=new_cols)
df_new.columns
"""

# <codecell>

#Preparing signal and background data for classifier
from cern_utils import data_storage

#Load signal and background data
signal_data = data_storage.DataStorageDF(df[y_true == 1])
bck_data = data_storage.DataStorageDF(df[y_true == 0])
# Get train and test data
signal_train, signal_test = signal_data.get_train_test(train_size=0.5)
bck_train, bck_test = bck_data.get_train_test(train_size=0.5)

# <codecell>

#select variables for classifier
columns = signal_data.columns
print columns
print '****************************************************'

variables = ['last-zeros', 'mass_center', 'inter_max', 'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
             u'mass_moment', u'r_moment',u'FileType',
             u'Configuration', u'ProcessingPass']#, u'Nb Replicas', ]
    
variables = [u'last-zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', u'mass_center',
             u'mass_center_sqr', u'mass_moment', u'r_moment', u'DiskSize', u'LogDiskSize', u'total_usage', u'mean_usage',
             u'FileType', u'Configuration', u'ProcessingPass', u'log_total_usage', u'log_mean_usage']+other_vars+bins

#variables = signal_data.columns

print variables
print len(variables)

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from cern_utils import xgboost_classifier


# setup parameters for xgboost
param = {}
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = 1
param['eta'] = 0.05
param['max_depth'] = 6
param['eval_metric'] = 'map'
param['silent'] = 1
param['nthread'] = 16
param['min_child_weight'] = 1
param['subsample'] = 0.8
param['colsample_bytree'] = 1
param['base_score'] = 0.5
#param['num_feature'] = 10

# you can directly throw param in, though we want to watch multiple metrics here 
plst = list(param.items()) + [('eval_metric', 'map'), ('eval_metric', 'auc')]

xgboost = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
xgboost.set_params(features = variables, params = plst)
#setup additional parameters
xgboost.num_boost_round = 2500
xgboost.watch = False
xgboost.missing = None

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

def log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred)  

def average_precision(y_true, y_pred, sample_weight):
    return average_precision_score(y_true, y_pred)  

figure(figsize=(10, 6))
lc_test = report.learning_curve( { 'roc_auc(test)':roc_auc}, steps=100)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=100)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
av_test = report.learning_curve( {  'average_precision(test)':average_precision}, steps=100)
av_train = report_train.learning_curve( {  'average_precision(train)':average_precision}, steps=100)
av_test.plots[0].plot()
av_train.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
ll_test = report.learning_curve( { 'log_loss(test)':roc_auc}, steps=100)
ll_train = report_train.learning_curve( { 'log_loss(train)':roc_auc}, steps=100)
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
#report.features_correlation_matrix().plot(show_legend=False)

# <codecell>

#Features histogramms
#hist_var = variables[:]
#hist_var.remove(u'NbTape')
#hist_var.remove(u'TapeSize')
#report.features_pdf(features=hist_var, bins = 10).plot()

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

figure(figsize=(10, 6))
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

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[periods_txt]
series = series[iron(report.prediction_sig['xgboost']) > 0.95]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(1,78)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[periods_txt]
series = series[iron(report.prediction_sig['xgboost']) > 0.85]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(1,78)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[periods_txt]
series = series[(iron(report.prediction_sig['xgboost']) > 0.4)&(iron(report.prediction_sig['xgboost']) < 0.8)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.02)
plt.xlim(1,78)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

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
    plt.plot(x, nb_rels, 'r', label = 'ratio of the true signals to the signals(%)')
    plt.legend(loc = 'best')
    plt.show()
    
    return cut_pq
    
def RSize(report, signal_test, bck_test, classifier='xgboost', mincut=0.01, maxcut=1, N=100, cond=0.9, Flag=False, pq=95, s_pq=90):
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
    s_cut_pq = 1
    
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
            if s_cut_pq==1 and sz_rel>=s_pq:
                s_cut_pq=i
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
            if s_cut_pq==1 and sz_rel>=s_pq:
                s_cut_pq=i

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
        plt.title('Ratio(%)')
        plt.legend(loc = 'best')
        plt.show()
    else:
        plt.figure(figsize=(5, 3))
        plt.subplot(1,1,1)
        plt.plot(x, sz_signals, 'b', label = 'released memory')
        plt.legend(loc = 'best')
        plt.show()
        
    return cut_pq, s_cut_pq

# <codecell>

CondSize(report, signal_test, bck_test, classifier='xgboost', cut=0.6, peaks=5, imax=26)

# <codecell>

cut_1, s_cut_1 = RSize(report, signal_test, bck_test, classifier='xgboost', mincut=0.1, maxcut=1, N=100, cond=0, Flag=True)
print cut_1, s_cut_1

# <codecell>

RSize(report, signal_test, bck_test, classifier='xgboost', mincut=0.01, maxcut=1, N=1000, cond=1.1)

# <codecell>

cut_pq11 = RFiles(report, signal_test, bck_test, classifier='xgboost', mincut=0.001, maxcut=1, N=1000)
print "cut_pq11 is ", cut_pq11

# <codecell>

xgboost2 = xgboost_classifier.ClassifierXGBoost(directory='xgboost2/')
xgboost2.set_params(features = variables, params = plst)
#setup additional parameters
xgboost2.num_boost_round = 1500
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
lc_test2 = report2.learning_curve( { 'roc_auc(test)':roc_auc}, steps=100)
lc_train2 = report_train2.learning_curve( { 'roc_auc(train)':roc_auc}, steps=100)
lc_test2.plots[0].plot()
lc_train2.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
av_test2 = report2.learning_curve( {  'average_precision(test)':average_precision}, steps=100)
av_train2 = report_train2.learning_curve( {  'average_precision(train)':average_precision}, steps=100)
av_test2.plots[0].plot()
av_train2.plots[0].plot()

# <codecell>

figure(figsize=(10, 6))
ll_test2 = report2.learning_curve( { 'log_loss(test)':roc_auc}, steps=100)
ll_train2 = report_train2.learning_curve( { 'log_loss(train)':roc_auc}, steps=100)
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
report2.prediction_pdf(bins = 10, normed = False, plot_type='bar').plot()

# <codecell>

#Normed signal
%pylab inline
from cern_utils import calc_util
iron2 = calc_util.classifier_flatten(report2.prediction_sig['xgboost2'])

_ = hist(iron2(report2.prediction_sig['xgboost2']),  histtype='bar', bins=10, alpha=0.5, label='signal', color='b')
_ = hist(iron2(report2.prediction_bck['xgboost2']),  histtype='bar', bins=10, alpha=0.5, label='bck', color='r')
legend(loc='best', fontsize='xx-large')

# <codecell>

#Normed signal
%pylab inline
from cern_utils import calc_util
iron2 = calc_util.classifier_flatten(report2.prediction_sig['xgboost2'])

_ = hist(report2.prediction_sig['xgboost2'],  histtype='bar', bins=10, alpha=0.5, label='signal', color='b')
_ = hist(report2.prediction_bck['xgboost2'],  histtype='bar', bins=10, alpha=0.5, label='bck', color='r')
legend(loc='best',fontsize='xx-large')

# <codecell>

cut_2, s_cut_2 = RSize(report2, signal_train, bck_train, classifier='xgboost2', mincut=0.01, maxcut=1, N=100, cond=0, Flag=True)
print cut_2, s_cut_2

# <codecell>

CondSize(report2, signal_train, bck_train, classifier='xgboost2', cut=0.6, peaks=5, imax=26)

# <codecell>

RSize(report2, signal_train, bck_train, classifier='xgboost2', mincut=0.01, maxcut=1, N=1000, cond=1.1)

# <codecell>

cut_pq22 = RFiles(report2, signal_train, bck_train, classifier='xgboost2', mincut=0.001, maxcut=1, N=1000)
print "cut_pq is ", cut_pq22

# <codecell>

def BraveSt(report, classifier, cut, signal, bck):
    
    iron = calc_util.classifier_flatten(report.prediction_sig[classifier])
    memory = signal.get_data(['DiskSize'])[iron(report.prediction_sig[classifier]) >= cut].values.sum()\
    +bck.get_data(['DiskSize'])[iron(report.prediction_bck[classifier]) >= cut].values.sum()
    
    return memory

def SafeSt(report, classifier, cut, signal, bck):
    
    iron = calc_util.classifier_flatten(report.prediction_sig[classifier])
    
    nzrs = (signal.get_data(['Nb Replicas']).values >= 1)[:,0]
    nzrb = (bck.get_data(['Nb Replicas']).values >= 1)[:,0]

    memory = signal.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= cut)&nzrs].values.sum()\
    +bck.get_data(['DiskSize'])[(iron(report.prediction_bck[classifier]) >= cut)&nzrb].values.sum()\
    -signal.get_data(['LFNSize'])[(iron(report.prediction_sig[classifier]) >= cut)&nzrs].values.sum()\
    -bck.get_data(['LFNSize'])[(iron(report.prediction_bck[classifier]) >= cut)&nzrb].values.sum()
    
    return memory

def CombineSt(report, classifier, s_cut, cut, signal, bck):
    
    iron = calc_util.classifier_flatten(report.prediction_sig[classifier])
    
    nzrs = (signal.get_data(['Nb Replicas']).values >= 1)[:,0]
    nzrb = (bck.get_data(['Nb Replicas']).values >= 1)[:,0]
    
    memory231 = signal.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= s_cut)&nzrs].values.sum()\
    +bck.get_data(['DiskSize'])[(iron(report.prediction_bck[classifier]) >= s_cut)&nzrb].values.sum()\
    -signal.get_data(['LFNSize'])[(iron(report.prediction_sig[classifier]) >= s_cut)&nzrs].values.sum()\
    -bck.get_data(['LFNSize'])[(iron(report.prediction_bck[classifier]) >= s_cut)&nzrb].values.sum()

    memory232 = signal.get_data(['DiskSize'])[(iron(report.prediction_sig[classifier]) >= cut)&nzrs].values.sum()\
    +bck.get_data(['DiskSize'])[(iron(report.prediction_bck[classifier]) >= cut)&nzrb].values.sum()\
    -signal.get_data(['LFNSize'])[(iron(report.prediction_sig[classifier]) >= cut)&nzrs].values.sum()\
    -bck.get_data(['LFNSize'])[(iron(report.prediction_bck[classifier]) >= cut)&nzrb].values.sum()

    memory233 = signal.get_data(['DiskSize'])[iron(report.prediction_sig[classifier]) >= cut].values.sum()\
    +bck.get_data(['DiskSize'])[iron(report.prediction_bck[classifier]) >= cut].values.sum()

    memory23 = memory231-memory232+memory233
    
    return memory23


#Selection
sel = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) == 0)
memory1 = data[sel].get('DiskSize').values.sum()
print memory1

def Totals(signal, bck):
    total = signal.get_data(['DiskSize']).values.sum()+bck.get_data(['DiskSize']).values.sum()
    return total
    
def CanRel(signal, bck):
    can_released = signal.get_data(['DiskSize']).values.sum()
    return can_released

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

#Rare usage
memory3 = CondSize(report2, signal_train, bck_train, classifier='xgboost2', cut=s_cut_1, peaks=3, imax=26)\
+CondSize(report, signal_test, bck_test, classifier='xgboost', cut=s_cut_2, peaks=3, imax=26)
print memory3

# <codecell>

#Using functions
memory_brave = memory1+BraveSt(report, 'xgboost', cut_1, signal_test, bck_test)\
+BraveSt(report2, 'xgboost2', cut_2, signal_train, bck_train)

memory_safe = memory1+SafeSt(report, 'xgboost', s_cut_1, signal_test, bck_test)\
+SafeSt(report2, 'xgboost2', s_cut_2, signal_train, bck_train)

memory_combine = memory1+CombineSt(report, 'xgboost', s_cut_1, cut_1, signal_test, bck_test)\
+CombineSt(report2, 'xgboost2', s_cut_2, cut_2, signal_train, bck_train)

print memory_brave
print memory_safe
print memory_combine

# <codecell>

import ipykee
session = ipykee.Session(project_name="C._NewFeatures")

# <codecell>

#session.commit("Classifier was trained. Bins of Nb os usages were added. Not optimized.")

# <codecell>

session.commit("Optimized.")

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[iron(report.prediction_sig['xgboost']) > 0.95]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(1,78)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[iron(report.prediction_sig['xgboost']) > 0.95]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[iron(report.prediction_sig['xgboost']) > 0.85]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[periods_txt]
series = series[(iron(report.prediction_sig['xgboost']) > 0.)&(iron(report.prediction_sig['xgboost']) < 0.2)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(1,78)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[periods_txt]
series = series[iron(report.prediction_sig['xgboost']) > 0.85]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(1,78)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.)&(iron(report.prediction_sig['xgboost']) < 0.2)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.)&(iron(report.prediction_sig['xgboost']) < 0.05)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.)&(iron(report.prediction_sig['xgboost']) < 0.99)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.)&(iron(report.prediction_sig['xgboost']) < 0.2)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.4)&(iron(report.prediction_sig['xgboost']) < 0.8)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.9)&(iron(report.prediction_sig['xgboost']) < 1)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
series = signal_test.get_data()[bins]
series = series[(iron(report.prediction_sig['xgboost']) > 0.8)&(iron(report.prediction_sig['xgboost']) < 1)]
print "Number of series is ", series.shape[0]
for i in range(0, series.shape[0]):
    cur_serie = series.irow(i)
    plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
plt.xlim(0,6)
plt.xlabel('Weeks')
plt.ylabel('Nb of usages')
plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[periods_txt]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[periods_txt]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

periods

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[periods_txt]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,78)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,10):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/10.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/10.)]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/10., ", ", (i+1)/10., "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
intervals = range(0,20)
for i in intervals:
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/(float)len(intervals))&(iron(report.prediction_sig['xgboost']) <= (i+1)/(float)len(intervals))]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/10., ", ", (i+1)/10., "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
intervals = range(0,20)
for i in intervals:
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/float(len(intervals)))&(iron(report.prediction_sig['xgboost']) <= (i+1)/float(len(intervals)))]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/10., ", ", (i+1)/10., "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
intervals = range(0,20)
for i in intervals:
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/float(len(intervals)))&(iron(report.prediction_sig['xgboost']) <= (i+1)/float(len(intervals)))]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/float(len(intervals)), ", ", (i+1)/float(len(intervals)), "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
print "These plots show the time series distribution along the antipopularity values."
print "All time series were splited into 6 bins with 13 weeks in each one."
intervals = range(0,20)
for i in intervals:
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/float(len(intervals)))&(iron(report.prediction_sig['xgboost']) <= (i+1)/float(len(intervals)))]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/float(len(intervals)), ", ", (i+1)/float(len(intervals)), "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('log10(Nb of usages)')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
print "These plots show the time series distribution along the antipopularity values."
print "All time series were splited into 6 bins with 13 weeks in each one."
intervals = range(0,20)
for i in intervals:
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/float(len(intervals)))&(iron(report.prediction_sig['xgboost']) <= (i+1)/float(len(intervals)))]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/float(len(intervals)), ", ", (i+1)/float(len(intervals)), "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('log10(Nb of usages)')
    plt.show()

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
for i in range(0,20):
    plt.subplot(1,1,1)
    series = signal_test.get_data()[periods_txt]
    series = series[(iron(report.prediction_sig['xgboost']) > i/20.)&(iron(report.prediction_sig['xgboost']) <= (i+1)/20.)]
    print "Number of series is ", series.shape[0]
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(periods, cur_serie.values, width=1, bottom=0, color='b', edgecolor='b', alpha=0.1)
    plt.xlim(0,78)
    plt.xlabel('Weeks')
    plt.ylabel('Nb of usages')
    plt.show()

# <codecell>

session.commit("Added plots that show the time series distribution along the antipopularity values.")

# <codecell>

#Plot signal_test series for an interval of antipopularity values using BINS
print "These plots show the time series distribution along the antipopularity values."
print "All time series were splited into 6 bins with 13 weeks in each one."
intervals = range(0,20)
for i in intervals:
    plt.subplot(1,1,1)
    series = signal_test.get_data()[bins]
    series = series[(iron(report.prediction_sig['xgboost']) > i/float(len(intervals)))&(iron(report.prediction_sig['xgboost']) <= (i+1)/float(len(intervals)))]
    print "Number of series is ", series.shape[0]
    print "Antipopularity is (", i/float(len(intervals)), ", ", (i+1)/float(len(intervals)), "]"
    for i in range(0, series.shape[0]):
        cur_serie = series.irow(i)
        plt.bar(range(0, len(bins)), np.log10(cur_serie.values+1), width=1, bottom=0, color='b', edgecolor='b', alpha=0.04)
    plt.xlim(0,6)
    plt.xlabel('Weeks')
    plt.ylabel('log10(Nb of usages)')
    plt.show()

