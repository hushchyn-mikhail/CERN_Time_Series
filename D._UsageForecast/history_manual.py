# -*- coding: utf-8 -*-


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

df_time_series

# <codecell>

data_weeks

# <codecell>

ts_data = df_time_series[:-26]
ts_data

# <codecell>

ts_data = df_time_series[range(1, 78)]
ts_data

# <codecell>

ts_data = df_time_series[range(1, 79)]
ts_data

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
ts_data

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = (data_weeks[104]>=0)*1
y_true

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = (data_weeks[104]>=0)*1
y_true.sum()

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = ((data_weeks[104]-data_weeks[78])>=0)*1
y_true.sum()

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = ((data_weeks[104]-data_weeks[78])>0)*1
y_true.sum()

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = ((data_weeks[104]-data_weeks[78])>0)*1

# <codecell>

def smoothing(time_serie):
    serie = time_serie
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    return serie

# <codecell>

ts_data.shape

# <codecell>

ts_rs_data = pd.rolling_sum(ts_data, window=52,axis=1)[range(52,131)]
ts_rs_data

# <codecell>

ts_rs_data = pd.rolling_sum(ts_data, window=52,axis=1)[range(52,131)]

# <codecell>

time_serie = ts_rs_data.irow([0])

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()])
plt.show()

# <codecell>

time_serie.max()

# <codecell>

time_serie

# <codecell>

time_serie.max(axis=1)

# <codecell>

time_serie.values.max(axis=1)

# <codecell>

time_serie.values[].max(axis=1)

# <codecell>

time_serie.values[0].max(axis=1)

# <codecell>

time_serie = ts_rs_data.irow([0]).values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()])
plt.show()

# <codecell>

time_serie = ts_rs_data.irow([0]).values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='d')
plt.show()

# <codecell>

time_serie = ts_rs_data.irow([0]).values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = [time_serie[0:27]]
right_data = [time_serie[27:53]]

# <codecell>

left_data = [(range(0,27), time_serie[0:27])]
right_data = [(range(27,53), time_serie[27:53])]

# <codecell>

from class sklearn.linear_model import class LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

from sklearn.linear_model import class LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

x = (range(0,27), time_serie[0:27])
x

# <codecell>

x = (range(0,27), time_serie[0:27])
x[0]

# <codecell>

x = (range(0,27), time_serie[0:27])
x[1]

# <codecell>

x = (range(0,27), time_serie[0:27])
x[1].shape

# <codecell>

x = (range(0,27), time_serie[0:27])
len(x[0])

# <codecell>

left_data = [(np.array(range(0,27)), time_serie[0:27])]
right_data = [(np.array(range(27,53)), time_serie[27:53])]

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

for x in left_data:
    x

# <codecell>

for x in left_data:
    print x

# <codecell>

for x in left_data:
    print x[0]

# <codecell>

for x in left_data:
    print x[1]

# <codecell>

np.array(range(0,27))

# <codecell>

np.array(range(0,27)).reshape((27,1))

# <codecell>

np.array(range(0,27)).reshape(27,1)

# <codecell>

left_data = [(np.array(range(0,27)).reshape(27,1), time_serie[0:27].reshape(27,1))]
right_data = [(np.array(range(27,53)).reshape(27,1), time_serie[27:53].reshape(27,1))]

# <codecell>

np.array(range(27,53)).shape

# <codecell>

left_data = [(np.array(range(0,27)).reshape(27,1), time_serie[0:27].reshape(27,1))]
right_data = [(np.array(range(27,53)).reshape(26,1), time_serie[27:53].reshape(26,1))]

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_data:
    plt.plot(X[0], X[1], color='r')
for X in right_data:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

left_data = []
for i in range(11,27):
    fit_data = (np.array(range(0,i)).reshape(i,1), time_serie[0:i].reshape(i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = [(np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))]

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

left_data = []
for i in range(11,27):
    fit_data = (np.array(range(0,i)).reshape(i,1), time_serie[0:i].reshape(i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = [(np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))]
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

left_data = []
for i in range(11,27):
    fit_data = (np.array(range(0,i)).reshape(i,1), time_serie[0:i].reshape(i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

time_serie = smoothing(ts_rs_data.irow([0]).values[0])

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
time_serie = ts_rs_data.irow([0]).values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    print lr.coef_
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

left_predict = []
for X in left_data:
    lr.fit(X[0], X[1])
    print lr.coef_[0,0]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
for X in right_data:
    lr.fit(X[0], X[1])
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = ts_rs_data.irow([0]).index[0]
results['Label'] = y_true[0]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = lr.coef_[0,0]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = lr.coef_[0,0]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_rs_data.irow([0]).index[0]]
results['Label'] = [y_true[0]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([3])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[0]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([3])
time_serie = smoothing(ts_train.values[0])

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[0]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([4])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([17])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[0]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([20])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[0]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([20])
time_serie = smoothing(ts_train.values[0])

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[0]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[20]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([20])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,17):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(10,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[20]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[20]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = ts.values[0]
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([row]), y_true[row])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report.append(new_row)
    if row%500:
        print row

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report.append(new_row)
    if row%500:
        print row

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([3])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

y_true

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = (((data_weeks[104]-data_weeks[78])>0)*1).values

# <codecell>

y_true

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([3])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = ts.values[0]
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

new_row

# <codecell>

report.append(new_row)

# <codecell>

report.append(new_row)
report

# <codecell>

report  = report.append(new_row)

# <codecell>

report  = report.append(new_row)
report

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report = report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

figure(figsize=(15,5))
subplot(121)
plt.hist(report['left_1'].values[report['Label'].values==0])

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0])

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0])
_=plt.hist(report['left_1'].values[report['Label'].values==1])

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5)
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5)

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(121)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(122)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

# <codecell>

ts_train.max(axis=1)

# <codecell>

ts_train.max(axis=1).values

# <codecell>

ts_train.max(axis=1).values[0]

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([3])
time_serie = ts_train.values[0]/ts_train.max(axis=1).values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([3])
time_serie = ts_train.values[0]/(ts_train.max(axis=1).values[0]+1)

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = ts.values[0]/(ts.max(axis=1).values[0]+1)
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report = report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

figure(figsize=(15,5))
subplot(121)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(122)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

# <codecell>

figure(figsize=(15,5))
subplot(131)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(132)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

# <codecell>

figure(figsize=(15,5))
subplot(131)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(132)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(133)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

# <codecell>

figure(figsize=(15,5))
subplot(131)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(132)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(133)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

# <codecell>

figure(figsize=(15,5))
subplot(131)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(132)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(133)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = report['Label'].values
y_score = diff
#y_score = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
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

figure(figsize=(15,5))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_22'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_1'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

left-cols = ['left_'+str(i) for i in range(1,23)]
report.columns

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
report.columns

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,23)]

left_avg = report[left_cols].mean(axis=1)
left_avg

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,23)]

left_avg = report[left_cols].mean(axis=1)
right_avg = report[right_cols].mean(axis=1)

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(right_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = left_avg - right_avg
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_avg hist')

subplot(222)
_=plt.hist(right_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_avg hist')

subplot(223)
diff = left_avg - right_avg
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for i in range(0, len(left_cols)):
    pass

# <codecell>

score = 0
for i in left_cols:
    score = score + (report[right_clas]>report[i])*1

# <codecell>

score = 0
for i in left_cols:
    score = score + (report[right_cols]>report[i])*1

# <codecell>

((report[right_cols]>report['left_1'])*1).sum(axis=1)

# <codecell>

((report[right_cols].values>report['left_1'].values)*1).sum(axis=1)

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]>report[l])*1).sum(axis=1)

# <codecell>

((report['right_1'].values>report['left_1'].values)*1).sum(axis=1)

# <codecell>

((report['right_1'].values>report['left_1'].values)*1)

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]>report[l])*1)

# <codecell>

score

# <codecell>

score

# <codecell>

score.max()

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]>report[l])*1)
score = score/484

# <codecell>

score.max()

# <codecell>

figure(figsize=(15,10))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]<report[l])*1)
score = score/484

# <codecell>

score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]<report[l])*1)
score = score/score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([46])
time_serie = ts_train.values[0]/(ts_train.max(axis=1).values[0]+1)

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = smoothing(ts.values[0]/(ts.max(axis=1).values[0]+1))
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report = report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_1'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,23)]

left_avg = report[left_cols].mean(axis=1)
right_avg = report[right_cols].mean(axis=1)

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_avg hist')

subplot(222)
_=plt.hist(right_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_avg hist')

subplot(223)
diff = left_avg - right_avg
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]<report[l])*1)
score = score/score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([46])
time_serie = smoothing(ts_train.values[0]/(ts_train.max(axis=1).values[0]+1))

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([46])
time_serie = ts_train.values[0]/(ts_train.max(axis=1).values[0]+1)

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([44])
time_serie = ts_train.values[0]/(ts_train.max(axis=1).values[0]+1)

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([46])
time_serie = ts_train.values[0]/(ts_train.max(axis=1).values[0]+1)

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([46])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

plt.plot(ts_data.irow([46]))
plt.plot()

# <codecell>

plt.plot(ts_data.irow([46]))
plt.show()

# <codecell>

plt.plot(ts_data.irow([46]).values[0])
plt.show()

# <codecell>

plt.plot(ts_data.irow([45]).values[0])
plt.show()

# <codecell>

plt.plot(ts_data.irow([44]).values[0])
plt.show()

# <codecell>

plt.plot(ts_data.irow([43]).values[0])
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([43])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([42])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

plt.plot(ts_data.irow([42]).values[0])
plt.show()

# <codecell>

plt.plot(ts_data.irow([41]).values[0])
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([41])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[3]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[41]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

plt.plot(df_time_series.irow([41]).values[0])
plt.show()

# <codecell>

ts_rs_data = pd.rolling_sum(ts_data, window=26,axis=1)[range(26,131)]

# <codecell>

ts_data = df_time_series[range(1, 53)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = (((data_weeks[104]-data_weeks[52])>0)*1).values

# <codecell>

ts_data = df_time_series[range(1, 53)]
for i in range(53, 53+52):
    ts_data[i] = 0
y_true = (((data_weeks[104]-data_weeks[52])>0)*1).values

# <codecell>

ts_data

# <codecell>

y_true

# <codecell>

def smoothing(time_serie):
    serie = time_serie
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    return serie

# <codecell>

ts_data.shape

# <codecell>

ts_rs_data = pd.rolling_sum(ts_data, window=26,axis=1)[range(26,105)]

# <codecell>

ts_train.max(axis=1).values[0]

# <codecell>

plt.plot(df_time_series.irow([41]).values[0])
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([41])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[41]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = smoothing(ts.values[0]/(ts.max(axis=1).values[0]+1))
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report = report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_1'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,23)]

left_avg = report[left_cols].mean(axis=1)
right_avg = report[right_cols].mean(axis=1)

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_avg hist')

subplot(222)
_=plt.hist(right_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_avg hist')

subplot(223)
diff = left_avg - right_avg
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]<report[l])*1)
score = score/score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = ts.values[0]/(ts.max(axis=1).values[0]+1)
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report = report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_1'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,23)]

left_avg = report[left_cols].mean(axis=1)
right_avg = report[right_cols].mean(axis=1)

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_avg hist')

subplot(222)
_=plt.hist(right_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_avg hist')

subplot(223)
diff = left_avg - right_avg
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]<report[l])*1)
score = score/score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

ts_data = df_time_series[range(1, 79)]
for i in range(79, 79+52):
    ts_data[i] = 0
y_true = (((data_weeks[104]-data_weeks[78])>0)*1).values

# <codecell>

y_true

# <codecell>

def smoothing(time_serie):
    serie = time_serie
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie, com=1)
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    serie = pd.ewma(serie[::-1], com=1)[::-1]
    return serie

# <codecell>

ts_data.shape

# <codecell>

ts_rs_data = pd.rolling_sum(ts_data, window=52,axis=1)[range(52,131)]

# <codecell>

ts_train.max(axis=1).values[0]

# <codecell>

plt.plot(df_time_series.irow([41]).values[0])
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([41])
time_serie = ts_train.values[0]

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
plt.show()

# <codecell>

left_data = []
for i in range(0,22):
    fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
    left_data.append(fit_data)
right_data = []
for i in range(5,27):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[41]]
lr = LinearRegression()

left_predict = []
num = 0
for X in left_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['left_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    left_predict.append((X[0], predict))
right_predict = []
num=0
for X in right_data:
    lr.fit(X[0], X[1])
    num = num + 1
    results['right_'+str(num)] = [lr.coef_[0,0]]
    predict = lr.predict(X[0])
    right_predict.append((X[0], predict))

# <codecell>

plt.plot(time_serie, color='b')
plt.plot([26,26],[0, time_serie.max()], color='black')
for X in left_predict:
    plt.plot(X[0], X[1], color='r')
for X in right_predict:
    plt.plot(X[0], X[1], color='r')
plt.show()

# <codecell>

results

# <codecell>

from sklearn.linear_model import LinearRegression

def GetTrends(ts, label):
    time_serie = ts.values[0]/(ts.max(axis=1).values[0]+1)
    
    left_data = []
    for i in range(0,22):
        fit_data = (np.array(range(i,27)).reshape(27-i,1), time_serie[i:27].reshape(27-i,1))
        left_data.append(fit_data)
    right_data = []
    for i in range(5,27):
        fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
        right_data.append(fit_data)
        
    results = pd.DataFrame()
    results['Index'] = [ts.index[0]]
    results['Label'] = [label]
    lr = LinearRegression()

    left_predict = []
    num = 0
    for X in left_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['left_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        left_predict.append((X[0], predict))
    right_predict = []
    num=0
    for X in right_data:
        lr.fit(X[0], X[1])
        num = num + 1
        results['right_'+str(num)] = [lr.coef_[0,0]]
        predict = lr.predict(X[0])
        right_predict.append((X[0], predict))
        
    return results

# <codecell>

report = GetTrends(ts_rs_data.irow([0]), y_true[0])

for row in range(1, ts_rs_data.shape[0]):
    new_row = GetTrends(ts_rs_data.irow([row]), y_true[row])
    report = report.append(new_row)
    if row%500==0:
        print row

# <codecell>

report

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_22']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_22'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_22 hist')

subplot(222)
_=plt.hist(report['right_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_1 hist')

subplot(223)
diff = report['left_1'] - report['right_1']
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,23)]

left_avg = report[left_cols].mean(axis=1)
right_avg = report[right_cols].mean(axis=1)

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_avg hist')

subplot(222)
_=plt.hist(right_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_avg hist')

subplot(223)
diff = left_avg - right_avg
_=plt.hist(diff.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((report[r]<report[l])*1)
score = score/score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = score
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

# <codecell>

session.commit("Trend Analysis. First finished version.")

# <codecell>

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

# <codecell>

session.commit("Trend Analysis. First finished version.")