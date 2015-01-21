# -*- coding: utf-8 -*-


# <codecell>

# from IPython import parallel
# clients = parallel.Client(profile='ssh-ipy2.0')
# clients.block = True  # use synchronous computations
# print clients.ids

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
    df_ts_states[col] = 0 + (df_time_series[col]>0)*(df_time_series[col]<=0.5*max_accesses)*1 + \
    (df_time_series[col]>0.5*max_accesses)*(df_time_series[col]<=max_accesses)*2

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
    distinct = set([0,1,2])
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

data.columns[100:]

# <codecell>

select = (data['inter_max']<=20)*(data['last_zeros']<=10)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

select = (data['inter_max']<=20)*(data['last-zeros']<=10)
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

select = (data['inter_max']<=20)*(data['last-zeros']>=10)
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

plt.hist(data['inter_mean'])
plt.show()

# <codecell>

select = (data['inter_max']<=20)*(data['inter_mean']<=5)
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

select = (data['inter_max']<=20)*(data['inter_mean']<=3)
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

select = (data['inter_max']<=20)*(data['inter_mean']>=3)
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

select = (data['inter_max']<=20)*(data['inter_mean']>=8)
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

plt.hist(data['inter_std'])
plt.show()

# <codecell>

select = (data['inter_max']<=20)*(data['inter_std']<=5)
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

select = (data['inter_max']<=20)*(data['inter_std']<=2)
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

plt.hist(data['inter_rel'])
plt.show()

# <codecell>

select = (data['inter_max']<=20)*(data['inter_rel']<=0.5)
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

select = (data['inter_max']<=20)*(data['inter_rel']>=0.5)
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

select = (data['inter_max']<=20)*(data['inter_rel']>=0.9)
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

plt.hist(data['nb_peaks'])
plt.show()

# <codecell>

select = (data['inter_max']<=20)*(data['nb_peaks']<=10)
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

select = (data['inter_max']<=20)*(data['nb_peaks']<=5)
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

select = (data['inter_max']<=20)*(data['nb_peaks']>=5)
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

select = (data['inter_max']<=20)*(data['nb_peaks']>=2)
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

select = (data['inter_max']<=20)*(data['nb_peaks']>=10)
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

select = (data['inter_max']<=20)
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

plt.hist(data['inter_max'])
plt.show()

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

select = (data['inter_max']<=5)
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

select = (data['inter_max']<=8)
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

select = (data['inter_max']<=2)
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

plt.hist(data['inter_max'])
plt.title("'inter_max' distribution")
plt.show()

# <codecell>

plt.hist(data['inter_max'])
plt.title("'inter_max' distribution for test+valid")
plt.show()

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=bt, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=bv, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,1])
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

session.commit("Markov Chains. Report 1.")