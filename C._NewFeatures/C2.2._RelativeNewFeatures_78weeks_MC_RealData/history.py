# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
%pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load original data
data = pd.read_excel('popularity-728days_my_origin.xls')
data.columns

# <codecell>

#Select data
selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
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
bv = Binary(data_sel, periods)
p = np.array(periods) - periods[0]+1
for i in p:
    df[i]=bv[i].values

# <codecell>

y_true = ((data_sel[104] - data_sel[78]) == 0).values*1

# <codecell>

for i in df.columns:
    plt.subplot(1,1,1)
    plt.hist(df[i][y_true == 1].values, label='signal', color='b', alpha=0.5, bins = 10)
    plt.hist(df[i][y_true == 0].values, label='bck', color='r', alpha=0.5, bins = 10)
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

from cern_utils import data_storage
"""
#whole data
one_signal_train, one_signal_test = data_storage.DataStorageDF(df[y_true == 1]).get_train_test(train_size=0.5)
one_bck_train, one_bck_test = data_storage.DataStorageDF(df[y_true == 0]).get_train_test(train_size=0.5)
"""
#Split Datasets to the MC and Real Data
groups = np.unique(data_sel.Type.values)

#list of the datastorages
signal_data = [0,0]
bck_data = [0,0]
signal_train = [0,0]
signal_test = [0,0]
bck_train = [0,0]
bck_test = [0,0]

#MC: data_sel.Type=0
#Real Data: data_sel.Type=1

for i in range(0, len(groups)):
    signal_data[i] = data_storage.DataStorageDF(df[(y_true == 1)&(df.Type == groups[i])])
    bck_data[i] = data_storage.DataStorageDF(df[(y_true == 0)&(df.Type == groups[i])])
    
    signal_train[i], signal_test[i] = signal_data[i].get_train_test(train_size=0.5)
    bck_train[i], bck_test[i] = bck_data[i].get_train_test(train_size=0.5)

#whole data
one_signal_train = signal_train[0]
one_signal_test = signal_test[0]
one_bck_train = bck_train[0]
one_bck_test = bck_test[0]
for i in range(1, len(groups)):
    one_signal_train = one_signal_train.union((one_signal_train, signal_train[i]))
    one_signal_test = one_signal_test.union((one_signal_test, signal_test[i]))
    one_bck_train = one_bck_train.union((one_bck_train, bck_train[i]))
    one_bck_test = one_bck_test.union((one_bck_test, bck_test[i]))

# <codecell>

#select variables for classifier
columns = signal_data[0].columns
print columns
print '****************************************************'

variables = ['last-zeros', 'mass_center', 'inter_max', 'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
             u'mass_moment', u'r_moment',u'FileType',
             u'Configuration', u'ProcessingPass']#, u'Nb Replicas', ]
    
variables = [u'last-zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', u'mass_center',
             u'mass_center_sqr', u'mass_moment', u'r_moment', u'DiskSize', u'LogDiskSize', u'total_usage', u'mean_usage',
             u'FileType', u'Configuration', u'ProcessingPass', u'log_total_usage', u'log_mean_usage']+other_vars

#variables = signal_data.columns

print variables
print len(variables)

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred)  

def average_precision(y_true, y_pred, sample_weight):
    return average_precision_score(y_true, y_pred) 

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from cern_utils import xgboost_classifier
from cern_utils.predictions_report import PredictionsInfo
%pylab inline
from cern_utils import calc_util

#list of the classifiers and reports
classifier = [0 for i in range(0, len(groups))]
report = [0 for i in range(0, len(groups))]
report_train = [0 for i in range(0, len(groups))]

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

#group ID
gid = 0

# setup parameters for xgboost
param = {}
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = 1
param['eta'] = 0.02
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

classifier_one = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
classifier_one.set_params(features = variables, params = plst)
#setup additional parameters
classifier_one.num_boost_round = 2500
classifier_one.watch = False
classifier_one.missing = None

#trainig classifier
classifier_one.fit(one_signal_train, one_bck_train)

print 'Classifier Results'
# get prediction on data after classification
report_one = PredictionsInfo({'classifier': classifier_one}, one_signal_test, one_bck_test)
report_train_one = PredictionsInfo({'classifier': classifier_one}, one_signal_train, one_bck_train)
#plotting learning curves
figure(figsize=(10, 6))
lc_test = report_one.learning_curve( { 'roc_auc(test)':roc_auc}, steps=100)
lc_train = report_train_one.learning_curve( { 'roc_auc(train)':roc_auc}, steps=100)
lc_test.plots[0].plot()
lc_train.plots[0].plot()
# get prediction on data after classification
figure(figsize=(10, 6))
report_one.prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train_one.prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')
#ROC - curve
figure(figsize=(5, 3))
report_one.roc().plot()
report_train_one.roc().plot()
legend(['test', 'train'], loc='best')
#Plot importances of features according to trained model
importance = classifier_one.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#transformed prediction
iron = calc_util.classifier_flatten(report_one.prediction_sig['classifier'])
_ = hist(iron(report_one.prediction_sig['classifier']),  histtype='bar', bins=20, alpha=0.5, label='signal')
_ = hist(iron(report_one.prediction_bck['classifier']),  histtype='bar', bins=20, alpha=0.5, label='bck')
legend(loc='best')

#Storage Space Saving
cut_one, s_cut_one = RSize(report_one, one_signal_test, one_bck_test, classifier='classifier', mincut=0.1, maxcut=1, N=100, cond=0, Flag=True)
print cut_one, s_cut_one

# <codecell>

#group ID
gid = 0

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

classifier[gid] = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
classifier[gid].set_params(features = variables, params = plst)
#setup additional parameters
classifier[gid].num_boost_round = 1000
classifier[gid].watch = False
classifier[gid].missing = None

#trainig classifier
classifier[gid].fit(signal_train[gid], bck_train[gid])

print 'Classifier Results'
# get prediction on data after classification
report[gid] = PredictionsInfo({'classifier': classifier[gid]}, signal_test[gid], bck_test[gid])
report_train[gid] = PredictionsInfo({'classifier': classifier[gid]}, signal_train[gid], bck_train[gid])
#plotting learning curves
figure(figsize=(10, 6))
lc_test = report[gid].learning_curve( { 'roc_auc(test)':roc_auc}, steps=100)
lc_train = report_train[gid].learning_curve( { 'roc_auc(train)':roc_auc}, steps=100)
lc_test.plots[0].plot()
lc_train.plots[0].plot()
# get prediction on data after classification
figure(figsize=(10, 6))
report[gid].prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train[gid].prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')
#ROC - curve
figure(figsize=(5, 3))
report[gid].roc().plot()
report_train[gid].roc().plot()
legend(['test', 'train'], loc='best')
#Plot importances of features according to trained model
importance = classifier[gid].get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#transformed prediction
gid = 0
iron = calc_util.classifier_flatten(report[gid].prediction_sig['classifier'])
_ = hist(iron(report[gid].prediction_sig['classifier']),  histtype='bar', bins=20, alpha=0.5, label='signal')
_ = hist(iron(report[gid].prediction_bck['classifier']),  histtype='bar', bins=20, alpha=0.5, label='bck')
legend(loc='best')

#Storage Space Saving
cut_1, s_cut_1 = RSize(report[gid], signal_test[gid], bck_test[gid], classifier='classifier', mincut=0.1, maxcut=1, N=100, cond=0, Flag=True)
print cut_1, s_cut_1

# <codecell>

#group ID
gid = 1

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

classifier[gid] = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
classifier[gid].set_params(features = variables, params = plst)
#setup additional parameters
classifier[gid].num_boost_round = 200
classifier[gid].watch = False
classifier[gid].missing = None

#trainig classifier
classifier[gid].fit(signal_train[gid], bck_train[gid])

print 'Classifier Results'
# get prediction on data after classification
report[gid] = PredictionsInfo({'classifier': classifier[gid]}, signal_test[gid], bck_test[gid])
report_train[gid] = PredictionsInfo({'classifier': classifier[gid]}, signal_train[gid], bck_train[gid])
#plotting learning curves
figure(figsize=(10, 6))
lc_test = report[gid].learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train[gid].learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()
# get prediction on data after classification
figure(figsize=(10, 6))
report[gid].prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train[gid].prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')
#ROC - curve
figure(figsize=(5, 3))
report[gid].roc().plot()
report_train[gid].roc().plot()
legend(['test', 'train'], loc='best')
#Plot importances of features according to trained model
importance = classifier[gid].get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#transformed prediction
gid = 1
iron = calc_util.classifier_flatten(report[gid].prediction_sig['classifier'])
_ = hist(iron(report[gid].prediction_sig['classifier']),  histtype='bar', bins=20, alpha=0.5, label='signal')
_ = hist(iron(report[gid].prediction_bck['classifier']),  histtype='bar', bins=20, alpha=0.5, label='bck')
legend(loc='best')

#Storage Space Saving
cut_2, s_cut_2 = RSize(report[gid], signal_test[gid], bck_test[gid], classifier='classifier', mincut=0.1, maxcut=1, N=100, cond=0, Flag=True)
print cut_2, s_cut_2

# <codecell>

#test datasets
gdf_signal_test = pd.DataFrame()
gdf_bck_test = pd.DataFrame()

for i in range(0, len(report)):
    
    len_sig = report[i].prediction_sig['classifier'].shape[0]
    len_bck = report[i].prediction_bck['classifier'].shape[0]
    
    df_temp_sig = pd.DataFrame()
    df_temp_bck = pd.DataFrame()
    
    for k in range(0, len(report)):
        df_temp_sig[str(k)] = -999*np.ones(report[i].prediction_sig['classifier'].shape[0])
        df_temp_bck[str(k)] = -999*np.ones(report[i].prediction_bck['classifier'].shape[0])
    
    df_temp_sig[str(i)] = report[i].prediction_sig['classifier']
    df_temp_bck[str(i)] = report[i].prediction_bck['classifier']
    
    gdf_signal_test = pd.concat([gdf_signal_test, df_temp_sig], axis=0)
    gdf_bck_test = pd.concat([gdf_bck_test, df_temp_bck], axis=0)
    
for i in ['DiskSize', 'Nb Replicas', 'LFNSize']: 
    gdf_signal_test[i] = np.concatenate((signal_test[0].get_data([i]).values[:,0],\
                                  signal_test[1].get_data([i]).values[:,0]), axis=0)
    gdf_bck_test[i] = np.concatenate((bck_test[0].get_data([i]).values[:,0],\
                                  bck_test[1].get_data([i]).values[:,0]), axis=0)

gds_signal_test = data_storage.DataStorageDF(gdf_signal_test)
gds_bck_test = data_storage.DataStorageDF(gdf_bck_test)


#train datasets
gdf_signal_train = pd.DataFrame()
gdf_bck_train = pd.DataFrame()

for i in range(0, len(report)):
    
    len_sig = report_train[i].prediction_sig['classifier'].shape[0]
    len_bck = report_train[i].prediction_bck['classifier'].shape[0]
    
    df_temp_sig = pd.DataFrame()
    df_temp_bck = pd.DataFrame()
    
    for k in range(0, len(report)):
        df_temp_sig[str(k)] = -999*np.ones(report_train[i].prediction_sig['classifier'].shape[0])
        df_temp_bck[str(k)] = -999*np.ones(report_train[i].prediction_bck['classifier'].shape[0])
    
    df_temp_sig[str(i)] = report_train[i].prediction_sig['classifier']
    df_temp_bck[str(i)] = report_train[i].prediction_bck['classifier']
    
    gdf_signal_train = pd.concat([gdf_signal_train, df_temp_sig], axis=0)
    gdf_bck_train = pd.concat([gdf_bck_train, df_temp_bck], axis=0)
    
for i in ['DiskSize', 'Nb Replicas', 'LFNSize']: 
    gdf_signal_train[i] = np.concatenate((signal_train[0].get_data([i]).values[:,0],\
                                  signal_train[1].get_data([i]).values[:,0]), axis=0)
    gdf_bck_train[i] = np.concatenate((bck_train[0].get_data([i]).values[:,0],\
                                  bck_train[1].get_data([i]).values[:,0]), axis=0)
    
gds_signal_train = data_storage.DataStorageDF(gdf_signal_train)
gds_bck_train = data_storage.DataStorageDF(gdf_bck_train)

# <codecell>

#All groups

# setup parameters for xgboost
param = {}
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = 1
param['eta'] = 0.05
param['max_depth'] = 2
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

classifier_all = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
classifier_all.set_params( params = plst)
#setup additional parameters
classifier_all.num_boost_round = 1000
classifier_all.watch = False
classifier_all.missing = None

#trainig classifier
classifier_all.fit(gds_signal_train, gds_bck_train)

print 'Classifier Results'
# get prediction on data after classification
report_all = PredictionsInfo({'classifier': classifier_all}, gds_signal_test, gds_bck_test)
report_train_all = PredictionsInfo({'classifier': classifier_all}, gds_signal_train, gds_bck_train)
#plotting learning curves
figure(figsize=(10, 6))
lc_test = report_all.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train_all.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()
# get prediction on data after classification
figure(figsize=(10, 6))
report_all.prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train_all.prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')
#ROC - curve
figure(figsize=(5, 3))
report_all.roc().plot()
report_train_all.roc().plot()
legend(['test', 'train'], loc='best')
#Plot importances of features according to trained model
importance = classifier_all.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#transformed prediction
iron = calc_util.classifier_flatten(report_all.prediction_sig['classifier'])
_ = hist(iron(report_all.prediction_sig['classifier']),  histtype='bar', bins=20, alpha=0.5, label='signal')
_ = hist(iron(report_all.prediction_bck['classifier']),  histtype='bar', bins=20, alpha=0.5, label='bck')
legend(loc='best')

#Storage Space Saving
cut_all, s_cut_all = RSize(report_all, gds_signal_test, gds_bck_test, classifier='classifier', mincut=0.1, maxcut=1, N=100, cond=0, Flag=True)
print cut_all, s_cut_all

# <codecell>

new_signal_train, new_signal_test = gds_signal_test.get_train_test(train_size=0.5)
new_bck_train, new_bck_test = gds_bck_test.get_train_test(train_size=0.5)

# <codecell>

#All groups

# setup parameters for xgboost
param = {}
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = 1
param['eta'] = 0.025
param['max_depth'] = 1
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

classifier_all = xgboost_classifier.ClassifierXGBoost(directory='xgboost/')
classifier_all.set_params( params = plst)
#setup additional parameters
classifier_all.num_boost_round = 1000
classifier_all.watch = False
classifier_all.missing = None

#trainig classifier
classifier_all.fit(new_signal_train, new_bck_train)

print 'Classifier Results'
# get prediction on data after classification
report_all = PredictionsInfo({'classifier': classifier_all}, new_signal_test, new_bck_test)
report_train_all = PredictionsInfo({'classifier': classifier_all}, new_signal_train, new_bck_train)
#plotting learning curves
figure(figsize=(10, 6))
lc_test = report_all.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train_all.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()
# get prediction on data after classification
figure(figsize=(10, 6))
report_all.prediction_pdf(bins = 20, normed = True, plot_type='bar', class_type='both').plot()
report_train_all.prediction_pdf(bins = 20, normed = True, class_type='both').plot()
xlim(0, 1)
legend(['bck(test)', 'sig(test)', 'bck(train)', 'sig(train)'], loc='best')
#ROC - curve
figure(figsize=(5, 3))
report_all.roc().plot()
report_train_all.roc().plot()
legend(['test', 'train'], loc='best')
#Plot importances of features according to trained model
importance = classifier_all.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#transformed prediction
iron = calc_util.classifier_flatten(report_all.prediction_sig['classifier'])
_ = hist(iron(report_all.prediction_sig['classifier']),  histtype='bar', bins=10, alpha=0.5, label='signal')
_ = hist(iron(report_all.prediction_bck['classifier']),  histtype='bar', bins=10, alpha=0.5, label='bck')
legend(loc='best')

#Storage Space Saving
cut_all, s_cut_all = RSize(report_all, new_signal_test, new_bck_test, classifier='classifier', mincut=0.1, maxcut=1, N=100, cond=0, Flag=True)
print cut_all, s_cut_all

# <codecell>

%matplotlib inline
#Compare ROC curves
from sklearn.metrics import roc_curve, auc

gr_names = ['MC','Real Data']
#Compare Whole-MC and Whole_Real Data ROC curves.
for i in range(0, len(groups)):
    y_true_wh = np.concatenate((np.ones((report_one.prediction_sig['classifier'][(one_signal_test.get_data(['Type']).values[:,0]==groups[i])]).shape[0]),\
                                np.zeros((report_one.prediction_bck['classifier'][(one_bck_test.get_data(['Type']).values[:,0]==groups[i])]).shape[0])),\
                               axis=0)
    y_score_wh = np.concatenate((report_one.prediction_sig['classifier'][(one_signal_test.get_data(['Type']).values[:,0]==groups[i])],\
                                report_one.prediction_bck['classifier'][(one_bck_test.get_data(['Type']).values[:,0]==groups[i])]),\
                               axis=0)
    fpr_wh, tpr_wh, _ = roc_curve(y_true_wh, y_score_wh)
    
    y_true_q = np.concatenate((np.ones((report[i].prediction_sig['classifier']).shape[0]),\
                                np.zeros((report[i].prediction_bck['classifier']).shape[0])),\
                               axis=0)
    y_score_q = np.concatenate((report[i].prediction_sig['classifier'],\
                                report[i].prediction_bck['classifier']),\
                               axis=0)
    fpr_q, tpr_q, _ = roc_curve(y_true_q, y_score_q)
    
    plt.subplot(1,1,1)
    print 'Square under the curve for the Whole Data is ', auc(fpr_wh, tpr_wh)
    print 'Square under the curve for the ',gr_names[i], 'is ', auc(fpr_q, tpr_q)
    plt.plot(fpr_wh, tpr_wh, 'r', label='Whole Data')
    plt.plot(fpr_q, tpr_q, 'b', label=gr_names[i])
    plt.legend(loc='best')
    plt.title('ROC')
    plt.show()
    
#Compare Whole-All_groups ROC curve
y_true_wh = np.concatenate((np.ones((report_one.prediction_sig['classifier']).shape[0]),\
                            np.zeros((report_one.prediction_bck['classifier']).shape[0])),\
                            axis=0)
y_score_wh = np.concatenate((report_one.prediction_sig['classifier'],\
                            report_one.prediction_bck['classifier']),\
                            axis=0)
fpr_wh, tpr_wh, _ = roc_curve(y_true_wh, y_score_wh)
    
y_true_all = np.concatenate((np.ones((report_all.prediction_sig['classifier']).shape[0]),\
                            np.zeros((report_all.prediction_bck['classifier']).shape[0])),\
                            axis=0)
y_score_all = np.concatenate((report_all.prediction_sig['classifier'],\
                            report_all.prediction_bck['classifier']),\
                            axis=0)
fpr_all, tpr_all, _ = roc_curve(y_true_all, y_score_all)
    
plt.subplot(1,1,1)
print 'Square under the curve for the Whole Data is ', auc(fpr_wh, tpr_wh)
print 'Square under the curve for the all groups is ', auc(fpr_all, tpr_all)
plt.plot(fpr_wh, tpr_wh, 'r', label='Whole Data')
plt.plot(fpr_all, tpr_all, 'b', label='All groups')
plt.legend(loc='best')
plt.title('ROC')
plt.show()

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

print 'Total Size of the Datasets is ', Totals(one_signal_test, one_bck_test)
#print 'Total Storage Space could be saved is ', CanRel(one_signal_test, one_bck_test)
print 'Memory 1 is ', memory1

# <codecell>

memory_one = CombineSt(report_one, 'classifier', s_cut_one, cut_one, one_signal_test, one_bck_test)
memory_gr = CombineSt(report[0], 'classifier', s_cut_1, cut_1, signal_test[0], bck_test[0])\
+CombineSt(report[1], 'classifier', s_cut_2, cut_2, signal_test[1], bck_test[1])
memory_all = 2*CombineSt(report_all, 'classifier', s_cut_all, cut_all, new_signal_test, new_bck_test)#\
#+CombineSt(report_train_all, 'classifier', s_cut_all, cut_all, new_signal_train, new_bck_train)

print memory_one
print memory_gr
print memory_all

# <codecell>

memory_one = BraveSt(report_one, 'classifier', cut_one, one_signal_test, one_bck_test)
memory_gr = BraveSt(report[0], 'classifier', cut_1, signal_test[0], bck_test[0])\
+BraveSt(report[1], 'classifier', cut_2, signal_test[1], bck_test[1])
memory_all = BraveSt(report_all, 'classifier', cut_all, new_signal_test, new_bck_test)\
+BraveSt(report_train_all, 'classifier', cut_all, new_signal_train, new_bck_train)

print memory_one
print memory_gr
print memory_all

# <codecell>

memory_one = SafeSt(report_one, 'classifier', s_cut_one, one_signal_test, one_bck_test)
memory_gr = SafeSt(report[0], 'classifier', s_cut_1, signal_test[0], bck_test[0])\
+SafeSt(report[1], 'classifier', s_cut_2, signal_test[1], bck_test[1])
memory_all = SafeSt(report_all, 'classifier', s_cut_all, new_signal_test, new_bck_test)\
+SafeSt(report_train_all, 'classifier', s_cut_all, new_signal_train, new_bck_train)

print memory_one
print memory_gr
print memory_all

# <codecell>

import ipykee
session = ipykee.Session(project_name="C._NewFeatures")

# <codecell>

session.commit("Upload by ipykee. NOT READY.")

# <codecell>

session.commit("Upload by ipykee. XGBoost classifier was trained on RelativeNewFeatures.")

