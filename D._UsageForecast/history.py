# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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
row = 48
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = ts_train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
    return np.mat(counts)

def StatDist(matrix):
    return np.array((matrix**100)[0,:])[0]

# <codecell>

#Example
row = 49
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
    return np.mat(counts)

def StatDist(matrix):
    return np.array((matrix**100)[0,:])[0]

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 41
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 42
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 33
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 10
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
row = 11
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
row = 12
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
row = 13
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
row = 14
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
row = 15
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

for i in range(0,60):
    figure(figsize=(15, 5))
    subplot(121)
    plt.plot(df_time_series.irow([row]).values[0])
    plt.title(str(i))
    subplot(122)
    plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

for i in range(0,60):
    figure(figsize=(15, 5))
    subplot(121)
    plt.plot(df_time_series.irow([i]).values[0])
    plt.title(str(i))
    subplot(122)
    plt.plot(df_ts_states.irow([i]).values[0])

# <codecell>

#Example
row = 32
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 40
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 44
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 53
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 54
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 56
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 59
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 58
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

#Example
row = 32
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

train1 = df_ts_states.irow([row]).values[0][:70]
test1 = df_ts_states.irow([row]).values[0][70:]

transition_matrix1 = TransitionMatrix(train1)
stationary_dist1 = StatDist(transition_matrix1)

print 'Transition matrix:\n', transition_matrix1
print 'Stationary distribution:\n', stationary_dist1

# <codecell>

train = df_ts_states.irow([row]).values[0][:70]
test = df_ts_states.irow([row]).values[0][70:]

transition_matrix = TransitionMatrix(train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

c = np.array(transition_matrix)
c

# <codecell>

c = np.array(transition_matrix).reshape(transition_matrix.shape[0]*transition_matrix.shape[1],)
c

# <codecell>

d = {1:2, 2:3}
d

# <codecell>

d[1]

# <codecell>

print 3**100

# <codecell>

%time print 3**100

# <codecell>

%%time 
print 3**100

# <codecell>

%time 
print 3**100

# <codecell>

print 60//10

# <codecell>

print 61//10

# <codecell>

print 61%10

# <codecell>

print 60%10

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}

for row in range(0, df_ts_states):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    if row%100==0:
        print row

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    if row%100==0:
        print row

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist

# <codecell>

dict_dists

# <codecell>

dict_matrixes

# <codecell>

#Example
row = 2
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

#Example
row = 2
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

#Example
row = 3
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print n
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print distinct
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

ts_train

# <codecell>

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n) - np.ones((n,n))
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n) - np.ones((n,n))
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n) - 1
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = 1 * coded_data[:-1] + n * coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train, n_states=3):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = n_states
    pairs = n * coded_data[:-1] + coded_data[1:]
    mask_data = np.array(range(0,n*n))
    #pairs = np.concatenate((pairs, mask_data))
    print pairs
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

ts_train

# <codecell>

#Example
row = 48
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

#Example
row = 4
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    print distinct
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    #distinct = []
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    #distinct = []
    print distinct
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    #distinct = []
    print type(distinct)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    distinct = set([0,1,2])
    print type(distinct)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    distinct = set(data)
    distinct = set([0,1,2])
    print type(distinct)
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1,2])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    print counts.sum(axis=1, dtype=float).reshape(n,1)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1,2])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    print counts.sum(axis=1, dtype=float).reshape(n,1) + counts.sum(axis=1, dtype=float).reshape(n,1)
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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

def TransitionMatrix(train):
    data = train
    #distinct = set(data)
    distinct = set([0,1,2])
    coding = {j:i for i, j in enumerate(distinct)}
    coded_data = np.fromiter((coding[i] for i in data), dtype=np.uint8)
    n = len(distinct)
    pairs = n * coded_data[:-1] + coded_data[1:]
    counts = np.bincount(pairs, minlength=n*n).reshape(n, n)
    print counts.sum(axis=1, dtype=float).reshape(n,1) + (counts.sum(axis=1, dtype=float).reshape(n,1)==0)*1
    #counts = counts/counts.sum(axis=1, dtype=float).reshape(n,1)
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
dict_dists = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist

# <codecell>

#Example
row = 1
figure(figsize=(15, 5))
subplot(121)
plt.plot(df_time_series.irow([row]).values[0])
subplot(122)
plt.plot(df_ts_states.irow([row]).values[0])

# <codecell>

#Example
ts_train = df_ts_states.irow([row]).values[0]
transition_matrix = TransitionMatrix(ts_train)
stationary_dist = StatDist(transition_matrix)

print 'Transition matrix:\n', transition_matrix
print 'Stationary distribution:\n', stationary_dist

# <codecell>

test.sum(axis=0)

# <codecell>

test.sum(axis=1)

# <codecell>

test.sum()

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    dict_test_sum = test.sum(axis=0)

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(dict_test_sum.itemes())

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(dict_test_sum.items())

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    dict_test_sum[row] = test.sum(axis=0)

# <codecell>

test

# <codecell>

test.sum()

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    dict_test_sum[row] = test.sum()

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}
dict_test_sum = {}

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    dict_test_sum[row] = test.sum()

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(dict_test_sum.items())

# <codecell>

test = df_ts_states.irow([row]).values[0][70:]
test

# <codecell>

test = df_ts_states.irow([row]).values[0][70:]
test.sum()

# <codecell>

test = df_ts_states.irow([0]).values[0][70:]
test.sum()

# <codecell>

test = df_ts_states.irow([0]).values[0][70:]
test.sum()

# <codecell>

test = df_ts_states.irow([0]).values[0][70:]
test.sum()

# <codecell>

dict_test_sum[1]

# <codecell>

dict_test_sum[0]

# <codecell>

dict_test_sum

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(dict_test_sum.items())

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}
test_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    test_sum.append(test.sum())

# <codecell>

test_sum

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum)

# <codecell>

%%time
dict_matrixes = {}
dict_dists = {}
test_sum = []

for row in range(0, df_ts_states.shape[0]):
    train = df_ts_states.irow([row]).values[0][:70]
    test = df_ts_states.irow([row]).values[0][70:]

    transition_matrix = TransitionMatrix(train)
    stationary_dist = StatDist(transition_matrix)
    
    dict_matrixes[row] = transition_matrix
    dict_dists[row] = stationary_dist
    
    test_sum.append(((test>0)*1).sum())

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum)

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum, bins=20)

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum, bins=10)

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

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

stat_dists

# <codecell>

stat_dists[1,:]

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(121)
plt.hist(stat_dists[1,:], bins=10)
plt.title('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(122)
plt.hist(stat_dists[1,:], bins=10)
plt.title('Stationary distribution of the state 1')

# <codecell>

stat_dists[1,:]

# <codecell>

figure(figsize=(15, 5))
subplot(121)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(122)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 5))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1])

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1])

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.1)

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.02)

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.02)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,2], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,2], alpha=0.02)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,0], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,0], alpha=0.02)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, 1-stat_dists[:,0], alpha=0.02)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, 1-stat_dists[:,0], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, 1-stat_dists[:,0], alpha=0.005)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(1-stat_dists[:,0], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, 1-stat_dists[:,0], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=100)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=100)
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35)
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,1], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,2], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(15, 10))
subplot(221)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(222)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(223)
plt.scatter(test_sum, stat_dists[:,1], alpha=0.01)
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')

subplot(224)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(15, 10))
subplot(231)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(232)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(233)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')
plt.title('LogNormed histogram')

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist(test_sum, bins=10)
plt.title('Number of the non zero values in test')

subplot(232)
plt.hist(stat_dists[:,1], bins=10)
plt.title('Stationary distribution of the state 1')

subplot(233)
plt.hist2d(test_sum, 1-stat_dists[:,0], alpha=1, bins=35, norm=LogNorm())
colorbar()
plt.xlabel('Number of the non zero values in test')
plt.ylabel('Stationary distribution of the state 1')
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

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

