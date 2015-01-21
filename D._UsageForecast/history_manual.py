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
periods = range(1,105)

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

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    stat_dists.append(stationary_dist)
    
    test_sum.append(((test>0)*1).sum())
    
stat_dists = np.array(stat_dists)
test_sum = np.array(test_sum)

# <codecell>

#Exxamples
for i in range(0,60):
    figure(figsize=(15, 5))
    subplot(121)
    plt.plot(df_time_series.irow([i]).values[0])
    plt.title(str(i))
    subplot(122)
    plt.plot(df_ts_states.irow([i]).values[0])

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(334)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(335)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(336)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(337)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(338)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(339)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>1)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>2)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>3)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>4)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>5)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>6)*1
#y_score = non_nan_res['66'].values
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
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

plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35)
plt.show()

# <codecell>

plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35)
plt.colorbar()
plt.show()

# <codecell>

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[test_sum==0,1], bins=10)
plt.hist(stat_dists[test_sum!=0,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[test_sum==0,1], bins=50)
plt.hist(stat_dists[test_sum!=0,1], bins=50)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[test_sum==0,1], bins=50)
#plt.hist(stat_dists[test_sum!=0,1], bins=50)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
#plt.hist(stat_dists[test_sum==0,1], bins=50)
plt.hist(stat_dists[test_sum!=0,1], bins=50)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[test_sum==0,1], bins=50)
#plt.hist(stat_dists[test_sum!=0,1], bins=50)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

%%time
dict_matrixes = {}
stat_dists = []
test_sum = []
valid_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][85:]
    valid = df_ts_states.irow([row]).values[0][70:85]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    stat_dists.append(stationary_dist)
    
    test_sum.append(((test>0)*1).sum())
    valid_sum.append(((valid>0)*1).sum())
    
stat_dists = np.array(stat_dists)
test_sum = np.array(test_sum)
valid_sum = np.array(valid_sum)

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(334)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(335)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(336)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(337)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(338)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(339)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(331)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(334)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(335)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(336)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(337)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(338)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(339)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(331)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(334)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(335)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(336)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(337)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(338)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(339)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(331)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(334)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(335)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(336)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(337)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(338)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(339)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(331)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in valid')

subplot(332)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(333)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(334)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in valid')

subplot(335)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(336)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(337)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in valid')

subplot(338)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(339)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in valid')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram')

subplot(344)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in valid')

subplot(345)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(346)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram')

subplot(347)
plt.hist(valid_sum, bins=10)
plt.title('Number of the non zero values in valid')

subplot(348)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(349)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum), axis=1)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum), axis=0)

# <codecell>

stat_dists

# <codecell>

valid_sum

# <codecell>

valid_sum.reshape(len(valid_sum), 1)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, vvalid_sum.reshape(len(valid_sum), 1)), axis=1)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

# <codecell>

X

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,\ 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None,\ 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y)
y_score = gbc.predict_proba(X)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

y_score

# <codecell>

y_true

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = ((test_sum>0)*1).reshape(len(test_sum)
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = ((test_sum>0)*1).reshape(len(test_sum))
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)

# <codecell>

y_true

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = ((test_sum>0)*1).reshape(len(test_sum),1)
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)

# <codecell>

y_true

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)

# <codecell>

y_true

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X).reshape(len(valid_sum))

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X).reshape(len(valid_sum),)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)#.reshape(len(valid_sum),)

# <codecell>

y_score

# <codecell>

y_score.reshape((len(valid_sum),))

# <codecell>

y_score

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=1000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

y_score.shape

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum[:3000]>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X[:3000,:],y_true)
y_score = gbc.predict_proba(X[3000:,:])[:,1].reshape(len(valid_sum[3000:]),)

# <codecell>

y_score.shape

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum[3000:]>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum[:3000]>0)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=3000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X[:3000,:],y_true)
y_score = gbc.predict_proba(X[3000:,:])[:,1].reshape(len(valid_sum[3000:]),)

# <codecell>

y_score.shape

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum[3000:]>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>1)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score.shape

# <codecell>

from sklearn.metrics import roc_curve, auc

#y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>1)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=2000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score.shape

# <codecell>

from sklearn.metrics import roc_curve, auc

#y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>2)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=2000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score.shape

# <codecell>

from sklearn.metrics import roc_curve, auc

#y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.ensemble import GradientBoostingClassifier

y_true = (test_sum>3)*1
X = np.concatenate((stat_dists, valid_sum.reshape(len(valid_sum), 1)), axis=1)

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=2000, subsample=1.0, 
                                 min_samples_split=2, min_samples_leaf=1, max_depth=6, init=None, random_state=None, 
                                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)

gbc.fit(X,y_true)
y_score = gbc.predict_proba(X)[:,1].reshape(len(valid_sum),)

# <codecell>

y_score.shape

# <codecell>

from sklearn.metrics import roc_curve, auc

#y_true = (test_sum>0)*1
#y_score = non_nan_res['66'].values
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum[valid_sum<5]>0)*1
y_score = stat_dists[valid_sum<5,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum[valid_sum<2]>0)*1
y_score = stat_dists[valid_sum<2,1]
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

from sklearn.metrics import roc_curve, auc
sel = valid_sum<2
y_true = (test_sum[sel]>0)*1
y_score = stat_dists[sel,1]
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

from sklearn.metrics import roc_curve, auc
sel = stat_dists[:,1] < 0.2
y_true = (test_sum[sel]>0)*1
y_score = stat_dists[sel,1]
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

from sklearn.metrics import roc_curve, auc
sel = stat_dists[:,1] < 0.1
y_true = (test_sum[sel]>0)*1
y_score = stat_dists[sel,1]
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

from sklearn.metrics import roc_curve, auc
sel = stat_dists[:,1] < 0.01
y_true = (test_sum[sel]>0)*1
y_score = stat_dists[sel,1]
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

from sklearn.metrics import roc_curve, auc
sel = stat_dists[:,1] < 0.05
y_true = (test_sum[sel]>0)*1
y_score = stat_dists[sel,1]
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

from sklearn.metrics import roc_curve, auc
y_true = (test_sum>0)*1
y_score = stat_dists[:,2]
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

from sklearn.metrics import roc_curve, auc
y_true = (test_sum>0)*1
y_score = 1-stat_dists[:,0]
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

plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
plt.show()

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
plt.show()

# <codecell>

Image

# <codecell>

Image.show()

# <codecell>

counts

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

xedges

# <codecell>

yedges

# <codecell>

counts

# <codecell>

counts[0,:]

# <codecell>

counts.max()

# <codecell>

counts_std = counts/counts.max()
counts_std

# <codecell>

counts_std = counts/counts.max()

# <codecell>

def GetCoord(xedges, yedges, x, y):
    for i in range(0,len(xedges)):
        if x>=xedges[i]:
            break
            
    for j in range(0,len(yedges)):
        if y>=yedges[j]:
            break
    
    return i,j

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, test_sum[i], stat_dists[i,1])
    y_score.append(counts_std[x,y])

# <codecell>

y_score

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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, test_sum[i], stat_dists[i,1])
    y_score.append(counts_std[x,y])

# <codecell>

y_score

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
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

plt.hist(y_score)
plt.show()

# <codecell>

plt.hist(y_score[y_true==0])
plt.hist(y_score[y_true!=0])
plt.show()

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, test_sum[i], stat_dists[i,1])
    y_score.append(counts_std[x,y])

y_score = np.array(y_score)

# <codecell>

plt.hist(y_score[y_true==0])
plt.hist(y_score[y_true!=0])
plt.show()

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.show()

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.legend(loc='best')
plt.show()

# <codecell>

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, 1-y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, test_sum[i], stat_dists[i,1])
    y_score.append(1-counts_std[x,y])

y_score = np.array(y_score)

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.legend(loc='best')
plt.show()

# <codecell>

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
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

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,1])
    y_score.append(1-counts_std[x,y])

y_score = np.array(y_score)

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.legend(loc='best')
plt.show()

# <codecell>

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
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

plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (valid_sum>0)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
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

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>1)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>2)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>3)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>4)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>5)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>10)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>1)*1
#y_score = stat_dists[:,1]
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

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,2])
    y_score.append(1-counts_std[x,y])

y_score = np.array(y_score)

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.legend(loc='best')
plt.show()

# <codecell>

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>1)*1
#y_score = stat_dists[:,1]
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

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], 1-stat_dists[i,0])
    y_score.append(1-counts_std[x,y])

y_score = np.array(y_score)

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.legend(loc='best')
plt.show()

# <codecell>

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>1)*1
#y_score = stat_dists[:,1]
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

plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (valid_sum>1)*1
#y_score = stat_dists[:,1]
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

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>1)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>2)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>3)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>5)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>15)*1
#y_score = stat_dists[:,1]
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

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
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

(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,1])
    y_score.append(1-counts_std[x,y])

y_score = np.array(y_score)

# <codecell>

plt.hist(y_score[y_true==0], label='y_true==0')
plt.hist(y_score[y_true!=0], label = 'y_true!=0')
plt.legend(loc='best')
plt.show()

# <codecell>

plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,1])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.show()

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")
plt.show()

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")
plt.show()

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.show()

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
ply.title('LogNormed histogram for test')

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,2,3)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,2,3)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,2,4)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,2,3)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,2,4)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,2,3)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,2,4)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,1])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

figure(figsize=(15, 10))
subplot(2,2,1)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,2,2)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,2,3)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,2,4)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

figure(figsize=(15, 10))

subplot(231)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true[y_score<0.5], y_score[y_score<0.5], pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

y_score[y_score<0.5]

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true[valid_sum<5], y_score[valid_sum<0.5], pos_label=None, sample_weight=None)
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true[valid_sum<5], y_score[valid_sum<5], pos_label=None, sample_weight=None)
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true[valid_sum<1], y_score[valid_sum<1], pos_label=None, sample_weight=None)
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true[valid_sum>1], y_score[valid_sum>1], pos_label=None, sample_weight=None)
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')

from sklearn.metrics import roc_curve, auc
#y_true = (test_sum>0)*1
#y_score = stat_dists[:,1]
fpr, tpr, _ = roc_curve(y_true[valid_sum>5], y_score[valid_sum>5], pos_label=None, sample_weight=None)
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

select = (data['nb_peaks']>=20)
select.shape

# <codecell>

select = (data['nb_peaks']>=20)
select.sum()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum[select], bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[select,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum[select], stat_dists[select,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum[select], stat_dists[select,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

select

# <codecell>

select.values

# <codecell>

test_sum(select.values)

# <codecell>

test_sum[select.values]

# <codecell>

select = (data['nb_peaks']>=20)
select = select.values
select.sum()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum[select], bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[select,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum[select], stat_dists[select,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum[select], stat_dists[select,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

# <codecell>

%%time
dict_matrixes = {}
stat_dists = []
test_sum = []
valid_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][85:]
    valid = df_ts_states.irow([row]).values[0][70:85]

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

select = (data['nb_peaks']>=20)
select = select.values

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
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

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_sum[i], stat_dists[i,1])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(2,3,4)
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
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
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

select = (data['nb_peaks']<=10)
select = select.values

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
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
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

data.columns

# <codecell>

data.columns[30:]

# <codecell>

data.columns[50:]

# <codecell>

data.columns[100:]

# <codecell>

select = (data['inter_max']<=10)
select = select.values

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
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
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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
print selec.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

select = (data['inter_max']<=10)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

select = (data['inter_max']<=5)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
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
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

select = (data['inter_max']>=0)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
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
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

# <codecell>

session.commit("Markov Chains. y_score added. Distributions of the y_score added.")

# <codecell>

select = (data['inter_max']<=5)
select = select.values
print select.sum()

stat_dists = stat_dists_t[select]
test_sum = test_sum_t[select]
valid_sum = valid_sum_t[select]

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 15))

subplot(341)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(342)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary probability of the state 1')

subplot(343)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(344)
plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(345)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(346)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary probability of the state 2')

subplot(347)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for test')

subplot(348)
plt.hist2d(valid_sum, stat_dists[:,2], alpha=1, bins=15, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in valid')
plt.ylabel('Stationary probability of the state 2')
plt.title('LogNormed histogram for valid')

subplot(349)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(3,4,10)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('1-Stationary probability of the state 1')

subplot(3,4,11)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('1-Stationary probability of the state 1')
plt.title('LogNormed histogram for valid')

subplot(3,4,12)
plt.hist2d(valid_sum, 1-stat_dists[:,0], alpha=1, bins=19, norm=LogNorm())
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
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=19, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary probability of the state 1')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_sum, stat_dists[:,1], alpha=1, bins=15, norm=LogNorm())
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
plt.hist2d(test_sum, y_score, alpha=1, bins=19, norm=LogNorm())
plt.colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')

subplot(2,3,5)
plt.hist2d(valid_sum, y_score, alpha=1, bins=19, norm=LogNorm())
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

session.commit("Markov Chains. Selection was made. ROC aus = 0.99.")