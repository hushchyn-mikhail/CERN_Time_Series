# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%%px
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load original data
data = pd.read_csv('popularity-728days_my.csv')

head = list(data.columns[:21]) + range(1,105)
data = pd.DataFrame(columns=head, data=data.values)

# <codecell>

#%%px
#Select data
selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
data_sel = data[selection].copy()
#data_sel = data.copy()
print data_sel.shape

# <codecell>

#%%px
periods = range(1,87) #train + valid size

#------------------------------------------------------
#Get maximum intervals and last weeks with zeros usages
def InterMax(data_sel, periods):
    #Get binary vector representation of the selected data
    data_bv = data_sel[periods].copy()
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
                
        last_zeros.append(periods[-1] - nz[-1] + 1)
        inter_max.append(max(inter))
    
    return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)

# <codecell>

#%%px
#Add features
inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = InterMax(data_sel, periods)
data_sel['last-zeros'] = last_zeros
data_sel['inter_max'] = inter_max
data_sel['nb_peaks'] = nb_peaks
data_sel['inter_mean'] = inter_mean
data_sel['inter_std'] = inter_std
data_sel['inter_rel'] = inter_rel

# <codecell>

#%%px
data = data_sel[data_sel['nb_peaks']>=0]

# <codecell>

#%%px
data_weeks = data[range(1,105)]

# <codecell>

#%%px
df_time_series = data_weeks.copy()
for i in range(1,105):
    if i!=1:
        df_time_series[i] = data_weeks[i]-data_weeks[i-1]

# <codecell>

max_accesses = df_time_series.max(axis=1)
df_ts_states = df_time_series.copy()
for col in df_ts_states.columns:
    df_ts_states[col] = 0 + (df_time_series[col]>0)*(df_time_series[col]<=1.*max_accesses)*1

# <codecell>

#Example
row = 1
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    row_sums = counts.sum(axis=1, dtype=float).reshape(n,1) + (counts.sum(axis=1, dtype=float).reshape(n,1)==0)*1
    counts = counts/row_sums
    return np.mat(counts)

def StatDist(matrix):
    return np.array((matrix**100)[0,:])[0]

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

%%time
dict_matrixes = {}
stat_dists = []
test_sum = []
valid_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][87:]
    valid = df_ts_states.irow([row]).values[0][70:87]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    stat_dists.append(stationary_dist)
    
    test_sum.append(((test>0)*1).sum())
    valid_sum.append(((valid>0)*1).sum())
    
stat_dists_t = np.array(stat_dists)
test_sum_t = np.array(test_sum)
valid_sum_t = np.array(valid_sum)

# <codecell>

data.columns[100:]

# <codecell>

select = (data['inter_max']>=0)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

bt=test.shape[0]
bv=valid.shape[0]

subplot(341)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

bt=test.shape[0]
bv=valid.shape[0]

subplot(341)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

bt=test.shape[0]
bv=valid.shape[0]

subplot(241)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(242)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(243)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(244)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(245)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(2,4,6)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(2,4,7)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,4,8)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))

bt=test.shape[0]
bv=valid.shape[0]

subplot(241)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(242)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(243)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(244)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(245)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(2,4,6)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(2,4,7)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,4,8)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

def GetCoord(xedges, yedges, x, y):
    for i in range(0,len(xedges)):
        if x<xedges[i]:
            break
            
    for j in range(0,len(yedges)):
        if y<yedges[j]:
            break
    
    return i-1,j-1

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

plt.hist(data['inter_max'])
plt.title("'inter_max' distribution for test+valid")
plt.show()

# <codecell>

data.columns[100:]

# <codecell>

select = (data['inter_max']<=10)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

# #get variables
# import ipykee
# keeper = ipykee.Keeper("C._NewFeatures")
# session = keeper["C2.1.1._RelativeNewFeatures_78weeks"]
# vars_c21 = session.get_variables("master")
# #variables.keys()

# <codecell>

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

# <codecell>

session.commit("Advanced Markov Chains. 2-states Markov chains.")

# <codecell>

#%%px
#Select data
#selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
data_sel = data[selection].copy()
data_sel = data.copy()
print data_sel.shape

# <codecell>

#%%px
#Select data
#selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
#data_sel = data[selection].copy()
data_sel = data.copy()
print data_sel.shape

# <codecell>

#%%px
#Select data
#selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
#data_sel = data[selection].copy()
data_sel = data.copy()
print data_sel.shape

# <codecell>

#%%px
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load original data
data = pd.read_csv('popularity-728days_my.csv')

head = list(data.columns[:21]) + range(1,105)
data = pd.DataFrame(columns=head, data=data.values)

# <codecell>

#%%px
#Select data
#selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
#data_sel = data[selection].copy()
data_sel = data.copy()
print data_sel.shape

# <codecell>

#%%px
periods = range(1,87) #train + valid size

#------------------------------------------------------
#Get maximum intervals and last weeks with zeros usages
def InterMax(data_sel, periods):
    #Get binary vector representation of the selected data
    data_bv = data_sel[periods].copy()
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
                
        last_zeros.append(periods[-1] - nz[-1] + 1)
        inter_max.append(max(inter))
    
    return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)

# <codecell>

#%%px
#Add features
inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = InterMax(data_sel, periods)
data_sel['last-zeros'] = last_zeros
data_sel['inter_max'] = inter_max
data_sel['nb_peaks'] = nb_peaks
data_sel['inter_mean'] = inter_mean
data_sel['inter_std'] = inter_std
data_sel['inter_rel'] = inter_rel

# <codecell>

#%%px
periods = range(1,87) #train + valid size

#------------------------------------------------------
#Get maximum intervals and last weeks with zeros usages
def InterMax(data_sel, periods):
    #Get binary vector representation of the selected data
    data_bv = data_sel[periods].copy()
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
                
        last_zeros.append(periods[-1] - nz[-1] + 1)
        inter_max.append(max(inter))
    
    return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)

# <codecell>

#%%px
#Add features
inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = InterMax(data_sel, periods)
data_sel['last-zeros'] = last_zeros
data_sel['inter_max'] = inter_max
data_sel['nb_peaks'] = nb_peaks
data_sel['inter_mean'] = inter_mean
data_sel['inter_std'] = inter_std
data_sel['inter_rel'] = inter_rel

# <codecell>

#%%px
data = data_sel[data_sel['nb_peaks']>=0]

# <codecell>

#%%px
data_weeks = data[range(1,105)]

# <codecell>

#%%px
df_time_series = data_weeks.copy()
for i in range(1,105):
    if i!=1:
        df_time_series[i] = data_weeks[i]-data_weeks[i-1]

# <codecell>

max_accesses = df_time_series.max(axis=1)
df_ts_states = df_time_series.copy()
for col in df_ts_states.columns:
    df_ts_states[col] = 0 + (df_time_series[col]>0)*(df_time_series[col]<=1.*max_accesses)*1

# <codecell>

#Example
row = 1
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    row_sums = counts.sum(axis=1, dtype=float).reshape(n,1) + (counts.sum(axis=1, dtype=float).reshape(n,1)==0)*1
    counts = counts/row_sums
    return np.mat(counts)

def StatDist(matrix):
    return np.array((matrix**100)[0,:])[0]

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

%%time
dict_matrixes = {}
stat_dists = []
test_sum = []
valid_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][87:]
    valid = df_ts_states.irow([row]).values[0][70:87]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    stat_dists.append(stationary_dist)
    
    test_sum.append(((test>0)*1).sum())
    valid_sum.append(((valid>0)*1).sum())
    
stat_dists_t = np.array(stat_dists)
test_sum_t = np.array(test_sum)
valid_sum_t = np.array(valid_sum)

# <codecell>

select = (data['inter_max']>=0)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))

bt=test.shape[0]
bv=valid.shape[0]

subplot(241)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(242)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(243)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(244)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(245)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(2,4,6)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(2,4,7)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,4,8)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

def GetCoord(xedges, yedges, x, y):
    for i in range(0,len(xedges)):
        if x<xedges[i]:
            break
            
    for j in range(0,len(yedges)):
        if y<yedges[j]:
            break
    
    return i-1,j-1

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

plt.hist(data['inter_max'])
plt.title("'inter_max' distribution for test+valid")
plt.show()

# <codecell>

data.columns[100:]

# <codecell>

select = (data['inter_max']<=10)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

data

# <codecell>

data.columns

# <codecell>

df = pd.DataFrame()

# <codecell>

df.columns

# <codecell>

df = pd.DataFrame()

df['last-zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel

features = ['']

labels = ((data['104'] - data['78']) == 0).values*1

# <codecell>

df = pd.DataFrame()

df['last-zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel

features = ['']

labels = ((data[104] - data[78]) == 0).values*1

# <codecell>

df.columns

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels, train_size=0.5)

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']

labels = ((data[104] - data[78]) == 0).values*1

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']

labels = ((data[104] - data[78]) == 0).values*1

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels, train_size=0.5)

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'test_mc', 'train'], loc='best')

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']

labels_valid = ((data[87] - data[70]) == 0).values*1

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

labels_valid = ((data[87] - data[70]) == 0).values*1

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

#%%px
periods = range(1,70) #train + valid size

#------------------------------------------------------
#Get maximum intervals and last weeks with zeros usages
def InterMax(data_sel, periods):
    #Get binary vector representation of the selected data
    data_bv = data_sel[periods].copy()
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
                
        last_zeros.append(periods[-1] - nz[-1] + 1)
        inter_max.append(max(inter))
    
    return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)

# <codecell>

#%%px
#Add features
inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = InterMax(data_sel, periods)
data_sel['last-zeros'] = last_zeros
data_sel['inter_max'] = inter_max
data_sel['nb_peaks'] = nb_peaks
data_sel['inter_mean'] = inter_mean
data_sel['inter_std'] = inter_std
data_sel['inter_rel'] = inter_rel

# <codecell>

#%%px
data = data_sel[data_sel['nb_peaks']>=0]

# <codecell>

#%%px
data_weeks = data[range(1,105)]

# <codecell>

#%%px
df_time_series = data_weeks.copy()
for i in range(1,105):
    if i!=1:
        df_time_series[i] = data_weeks[i]-data_weeks[i-1]

# <codecell>

max_accesses = df_time_series.max(axis=1)
df_ts_states = df_time_series.copy()
for col in df_ts_states.columns:
    df_ts_states[col] = 0 + (df_time_series[col]>0)*(df_time_series[col]<=1.*max_accesses)*1

# <codecell>

#Example
row = 1
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    row_sums = counts.sum(axis=1, dtype=float).reshape(n,1) + (counts.sum(axis=1, dtype=float).reshape(n,1)==0)*1
    counts = counts/row_sums
    return np.mat(counts)

def StatDist(matrix):
    return np.array((matrix**100)[0,:])[0]

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

%%time
dict_matrixes = {}
stat_dists = []
test_sum = []
valid_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][87:]
    valid = df_ts_states.irow([row]).values[0][70:87]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    stat_dists.append(stationary_dist)
    
    test_sum.append(((test>0)*1).sum())
    valid_sum.append(((valid>0)*1).sum())
    
stat_dists_t = np.array(stat_dists)
test_sum_t = np.array(test_sum)
valid_sum_t = np.array(valid_sum)

# <codecell>

data.columns

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

labels_valid = ((data[87] - data[70]) == 0).values*1

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

data.columns[100:]

# <codecell>

select = (data['inter_max']>=0)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))

bt=test.shape[0]
bv=valid.shape[0]

subplot(241)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(242)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(243)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(244)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(245)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(2,4,6)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(2,4,7)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,4,8)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

def GetCoord(xedges, yedges, x, y):
    for i in range(0,len(xedges)):
        if x<xedges[i]:
            break
            
    for j in range(0,len(yedges)):
        if y<yedges[j]:
            break
    
    return i-1,j-1

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1']
features = ['State_0', 'State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1']
features = ['State_0', 'State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1']
features = ['State_0', 'State_1', 'y_score']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

#labels_valid = ((data[87] - data[70]) == 0).values*1
labels_valid = ((data[104] - data[84]) == 0).values*1

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1']
features = ['State_0', 'State_1', 'y_score']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1']
features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_1', 'y_score', 'inter_max']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'Score_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

labels_valid = ((data[87] - data[70]) == 0).values*1
#labels_valid = ((data[104] - data[84]) == 0).values*1

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

#labels_valid = ((data[87] - data[70]) == 0).values*1
labels_valid = ((data[104] - data[84]) == 0).values*1

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

labels_valid = ((data[87] - data[70]) == 0).values*1
#labels_valid = ((data[104] - data[84]) == 0).values*1

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

plt.hist2d(valid_sum, df['State_0'].values, alpha=1, bins=bv, norm=LogNorm())

# <codecell>

_=plt.hist2d(valid_sum, df['State_0'].values, alpha=1, bins=bv, norm=LogNorm())

# <codecell>

_=plt.hist2d(valid_sum, df['State_1'].values, alpha=1, bins=bv, norm=LogNorm())

# <codecell>

df.columns

# <codecell>

_=plt.hist2d(valid_sum, df['last_zeros'].values, alpha=1, bins=bv, norm=LogNorm())

# <codecell>

_=plt.hist2d(valid_sum, df['last_zeros'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

_=plt.hist2d(valid_sum, df['inter_max'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df['inter_max'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df['inter_max'].values)
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df['inter_max'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df['inter_max'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

_=plt.hist2d(valid_sum, df['nb_peaks'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df['nb_peaks'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df['nb_peaks'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df['nb_peaks'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

_=plt.hist2d(valid_sum, df['inter_mean'].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

col = 'inter_mean'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df[col].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df[col].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

col = 'inter_std'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df[col].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df[col].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

col = 'inter_rel'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df[col].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df[col].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

col = 'State_0'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

col = 'State_1'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df[col].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], df[col].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

#%%px
#Select data
selection = ((data['Now'] - data['Creation-week']) > 26)&((data['Now'] - data['FirstUsage']) > 26)&((data[78] - data[1]) != 0)
data_sel = data[selection].copy()
#data_sel = data.copy()
print data_sel.shape

# <codecell>

#%%px
periods = range(1,70) #train + valid size

#------------------------------------------------------
#Get maximum intervals and last weeks with zeros usages
def InterMax(data_sel, periods):
    #Get binary vector representation of the selected data
    data_bv = data_sel[periods].copy()
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
                
        last_zeros.append(periods[-1] - nz[-1] + 1)
        inter_max.append(max(inter))
    
    return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)

# <codecell>

#%%px
#Add features
inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = InterMax(data_sel, periods)
data_sel['last-zeros'] = last_zeros
data_sel['inter_max'] = inter_max
data_sel['nb_peaks'] = nb_peaks
data_sel['inter_mean'] = inter_mean
data_sel['inter_std'] = inter_std
data_sel['inter_rel'] = inter_rel

# <codecell>

#%%px
data = data_sel[data_sel['nb_peaks']>=0]

# <codecell>

#%%px
data_weeks = data[range(1,105)]

# <codecell>

#%%px
df_time_series = data_weeks.copy()
for i in range(1,105):
    if i!=1:
        df_time_series[i] = data_weeks[i]-data_weeks[i-1]

# <codecell>

max_accesses = df_time_series.max(axis=1)
df_ts_states = df_time_series.copy()
for col in df_ts_states.columns:
    df_ts_states[col] = 0 + (df_time_series[col]>0)*(df_time_series[col]<=1.*max_accesses)*1

# <codecell>

#Example
row = 1
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    row_sums = counts.sum(axis=1, dtype=float).reshape(n,1) + (counts.sum(axis=1, dtype=float).reshape(n,1)==0)*1
    counts = counts/row_sums
    return np.mat(counts)

def StatDist(matrix):
    return np.array((matrix**100)[0,:])[0]

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

%%time
dict_matrixes = {}
stat_dists = []
test_sum = []
valid_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][87:]
    valid = df_ts_states.irow([row]).values[0][70:87]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    stat_dists.append(stationary_dist)
    
    test_sum.append(((test>0)*1).sum())
    valid_sum.append(((valid>0)*1).sum())
    
stat_dists_t = np.array(stat_dists)
test_sum_t = np.array(test_sum)
valid_sum_t = np.array(valid_sum)

# <codecell>

data.columns

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

labels_valid = ((data[87] - data[70]) == 0).values*1
#labels_valid = ((data[104] - data[84]) == 0).values*1

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, 1-stat_dists_t[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum_t)):
    x,y = GetCoord(xedges, yedges, valid_sum_t[i], 1-stat_dists_t[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

col = 'State_1'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

col = 'inter_max'
_=plt.hist2d(valid_sum, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Your column')
plt.title('LogNormed histogram for valid')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = df[col].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

y_true

# <codecell>

test_sum

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum_t>0)*1
y_score = df[col].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, df[col].values, alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum_t[i], df[col].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum_t>0)*1
#y_score = df['inter_max'].values
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

#labels_valid = ((data[87] - data[70]) == 0).values*1
labels_valid = ((data[104] - data[84]) == 0).values*1

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[84]) == 0).values*1

#labels_valid = ((data[87] - data[70]) == 0).values*1
labels_valid = ((data[104] - data[87]) == 0).values*1

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[87]) == 0).values*1

#labels_valid = ((data[87] - data[70]) == 0).values*1
labels_valid = ((data[104] - data[87]) == 0).values*1

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, 1-stat_dists_t[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum_t)):
    x,y = GetCoord(xedges, yedges, valid_sum_t[i], 1-stat_dists_t[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

df['y_score'] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
#features = ['State_0', 'State_1', 'y_score', 'inter_max']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[87]) == 0).values*1

labels_valid = ((data[87] - data[70]) == 0).values*1
#labels_valid = ((data[104] - data[87]) == 0).values*1

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]

variables = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

for num in range(0,len(variables)):
    subplot(3,3,num+1)
    (counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, 1-stat_dists_t[:,0], alpha=1, bins=bv, norm=LogNorm())
    colorbar()
    plt.xlabel('Number of the non zero values in valid')
    plt.ylabel(variables[num])
    plt.title('LogNormed histogram for valid')

#     counts_std = counts/counts.max()
#     y_score = []
#     for i in range(0, len(test_sum_t)):
#         x,y = GetCoord(xedges, yedges, valid_sum_t[i], 1-stat_dists_t[i,0])
#         y_score.append(1-counts_std[x,y])
#     y_score = np.array(y_score)

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]

variables = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

figure(figsize=(20, 10))
for num in range(0,len(variables)):
    subplot(2,4,num+1)
    (counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, 1-stat_dists_t[:,0], alpha=1, bins=bv, norm=LogNorm())
    colorbar()
    plt.xlabel('Number of the non zero values in valid')
    plt.ylabel(variables[num])
    plt.title('LogNormed histogram for valid')

#     counts_std = counts/counts.max()
#     y_score = []
#     for i in range(0, len(test_sum_t)):
#         x,y = GetCoord(xedges, yedges, valid_sum_t[i], 1-stat_dists_t[i,0])
#         y_score.append(1-counts_std[x,y])
#     y_score = np.array(y_score)

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]

variables = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

figure(figsize=(20, 10))
for num in range(0,len(variables)):
    subplot(2,4,num+1)
    (counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, df[variables[num]].values, alpha=1, bins=bv, norm=LogNorm())
    colorbar()
    plt.xlabel('Number of the non zero values in valid')
    plt.ylabel(variables[num])
    plt.title('LogNormed histogram for valid')

#     counts_std = counts/counts.max()
#     y_score = []
#     for i in range(0, len(test_sum_t)):
#         x,y = GetCoord(xedges, yedges, valid_sum_t[i], 1-stat_dists_t[i,0])
#         y_score.append(1-counts_std[x,y])
#     y_score = np.array(y_score)

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]

variables = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

figure(figsize=(20, 10))
for num in range(0,len(variables)):
    subplot(2,4,num+1)
    (counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, df[variables[num]].values, alpha=1, bins=bv, norm=LogNorm())
    colorbar()
    plt.xlabel('Number of the non zero values in valid')
    plt.ylabel(variables[num])
    plt.title('LogNormed histogram for valid')

    counts_std = counts/counts.max()
    y_score = []
    for i in range(0, len(test_sum_t)):
        x,y = GetCoord(xedges, yedges, valid_sum_t[i], df[variables[num]].values[i])
        y_score.append(1-counts_std[x,y])
    y_score = np.array(y_score)
    df['y_score_'+variables[num]] = y_score

# <codecell>

df

# <codecell>

df = pd.DataFrame()

df['last_zeros'] = last_zeros
df['inter_max'] = inter_max
df['nb_peaks'] = nb_peaks
df['inter_mean'] = inter_mean
df['inter_std'] = inter_std
df['inter_rel'] = inter_rel
df['State_0'] = stat_dists_t[:,0]
df['State_1'] = stat_dists_t[:,1]
df['labels_test'] = ((data[104] - data[87]) == 0).values*1

#labels_valid = ((data[87] - data[70]) == 0).values*1
labels_valid = ((data[104] - data[87]) == 0).values*1

# <codecell>

from matplotlib.colors import LogNorm
bv=valid.shape[0]

variables = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel', 'State_0', 'State_1']

figure(figsize=(20, 10))
for num in range(0,len(variables)):
    subplot(2,4,num+1)
    (counts, xedges, yedges, Image) = plt.hist2d(valid_sum_t, df[variables[num]].values, alpha=1, bins=bv, norm=LogNorm())
    colorbar()
    plt.xlabel('Number of the non zero values in valid')
    plt.ylabel(variables[num])
    plt.title('LogNormed histogram for valid')

    counts_std = counts/counts.max()
    y_score = []
    for i in range(0, len(test_sum_t)):
        x,y = GetCoord(xedges, yedges, valid_sum_t[i], df[variables[num]].values[i])
        y_score.append(1-counts_std[x,y])
    y_score = np.array(y_score)
    df['y_score_'+variables[num]] = y_score

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

train_data, test_data, train_labels, test_labels = train_test_split(df, labels_valid, train_size=0.5)

# <codecell>

df.columns

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
            u'State_0', u'State_1', u'labels_test', u'y_score_last_zeros', u'y_score_inter_max',
            u'y_score_nb_peaks', u'y_score_inter_mean', u'y_score_inter_std', u'y_score_inter_rel',
            u'y_score_State_0', u'y_score_State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
            u'State_0', u'State_1', u'y_score_last_zeros', u'y_score_inter_max',
            u'y_score_nb_peaks', u'y_score_inter_mean', u'y_score_inter_std', u'y_score_inter_rel',
            u'y_score_State_0', u'y_score_State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
features = [ u'y_score_last_zeros', u'y_score_inter_max',
            u'y_score_nb_peaks', u'y_score_inter_mean', u'y_score_inter_std', u'y_score_inter_rel',
            u'y_score_State_0', u'y_score_State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
            u'State_0', u'State_1', u'y_score_last_zeros', u'y_score_inter_max',
            u'y_score_nb_peaks', u'y_score_inter_mean', u'y_score_inter_std', u'y_score_inter_rel',
            u'y_score_State_0', u'y_score_State_1']

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
            u'State_0', u'State_1', u'y_score_last_zeros', u'y_score_inter_max',
            u'y_score_nb_peaks', u'y_score_inter_mean', u'y_score_inter_std', u'y_score_inter_rel',
             u'y_score_State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel']#, 'State_0', 'State_1', 'y_score']
features = [u'last_zeros', u'inter_max', u'nb_peaks', u'inter_mean', u'inter_std', u'inter_rel',
            u'State_0', u'State_1', u'y_score_last_zeros', u'y_score_inter_max',
            u'y_score_nb_peaks', u'y_score_inter_mean', u'y_score_inter_std', u'y_score_inter_rel',
            u'y_score_State_0', u'y_score_State_1']

# <codecell>

import inspect
import os
import sys

code_path = os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost-master/wrapper")
sys.path.append(code_path)

import xgboost as xgb
from rep.classifiers import XGBoostClassifier

xgboost = XGBoostClassifier(objective='binary:logitraw', eta=0.02, max_depth=6, subsample=0.8, features=features, n_estimators=2500)
xgboost.fit(train_data, train_labels)

# <codecell>

from rep.report import ClassificationReport
from rep.data.storage import DataStorageDF, LabeledDataStorage

lds_test = LabeledDataStorage(DataStorageDF(test_data[features]), test_labels)
#report_test_mc = ClassificationReport({'xgboost':xgboost}, lds_test)
report_test = ClassificationReport({'xgboost':xgboost}, lds_test)

lds_train = LabeledDataStorage(DataStorageDF(train_data[features]), train_labels)
report_train = ClassificationReport({'xgboost':xgboost}, lds_train)

# <codecell>

#Plot importances of features according to trained model
importance = xgboost.get_feature_importance()
importance.sort(['effect'], ascending=False)[['effect']].plot(figsize=(13,3), kind='bar')

# <codecell>

#Plot learning curves to see possible overfitting of trained classifier
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

def roc_auc(y_true, y_pred, sample_weight):
    return roc_auc_score(y_true, y_pred[:,1])  

figure(figsize=(10, 6))
lc_test = report_test.learning_curve( { 'roc_auc(test)':roc_auc}, steps=10)
lc_train = report_train.learning_curve( { 'roc_auc(train)':roc_auc}, steps=10)
lc_test.plots[0].plot()
lc_train.plots[0].plot()

# <codecell>

#ROC - curve
figure(figsize=(5, 3))
report_test.roc().plot()
#report_test_mc.roc().plot()
report_train.roc().plot()
legend(['test', 'train'], loc='best')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = test_data['labels_test'].values
y_score = xgboost.predict_proba(test_data)[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

data.columns[100:]

# <codecell>

select = (data['inter_max']>=0)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))

bt=test.shape[0]
bv=valid.shape[0]

subplot(241)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(242)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(243)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(244)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(245)
plt.hist(test_sum, bins=bt)
plt.title('Number of the non zero values in test')

subplot(2,4,6)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(2,4,7)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,4,8)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

#General ROC-curve
from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

def GetCoord(xedges, yedges, x, y):
    for i in range(0,len(xedges)):
        if x<xedges[i]:
            break
            
    for j in range(0,len(yedges)):
        if y<yedges[j]:
            break
    
    return i-1,j-1

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=bt, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=bv, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

