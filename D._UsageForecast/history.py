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
data = data_sel[data_sel['inter_max']>=0]

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

plt.plot(df_time_series.irow([69]).values[0])
plt.show()

# <codecell>

#time_serie = smoothing(ts_rs_data.irow([0]).values[0])
ts_train = ts_rs_data.irow([69])
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
for i in range(5,50):
    fit_data = (np.array(range(27,27+i)).reshape(i,1), time_serie[27:27+i].reshape(i,1))
    right_data.append(fit_data)

# <codecell>

from sklearn.linear_model import LinearRegression

results = pd.DataFrame()
results['Index'] = [ts_train.index[0]]
results['Label'] = [y_true[37]]
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
    for i in range(5,50):
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

figure(figsize=(15,10))
subplot(221)
_=plt.hist(report['left_1'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['left_1'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(report['right_45'].values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(report['right_45'].values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_22 hist')

subplot(223)
diff = report['left_1'] - report['right_45']
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
_=plt.hist(report['left_22'].values[report['Label'].values==0], color='r', alpha=0.5, label='0', bins=10)
_=plt.hist(report['left_22'].values[report['Label'].values==1], color='b', alpha=0.5, label='1', bins=10)
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
plt.title('Left_1 hist')

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

report.columns

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,45)]

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
diff = left_avg / (right_avg+0.1)
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

diff_avg = 0
num=0.
for l in left_cols:
    for r in right_cols:
        diff_avg = diff_avg + report[l]-report[r]
        num = num+1.
diff_avg = diff_avg/num

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(diff_avg.values[report['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff_avg.values[report['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = report['Label'].values
y_score = diff_avg
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc

# <codecell>

rep = report[(data_sel['inter_max'].values<=25)]
rep.shape

# <codecell>

score = 0
for l in left_cols:
    for r in right_cols:
        score = score + ((rep[r]<rep[l])*1)
score = score/score.max()

# <codecell>

figure(figsize=(15,5))

subplot(121)
_=plt.hist(score.values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(score.values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

subplot(122)
from sklearn.metrics import roc_curve, auc
y_true = rep['Label'].values
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

from rep.utils import Flattener
iron = Flattener(score.values[rep['Label'].values==0])

figure(figsize=(15,5))

subplot(121)
_=plt.hist(iron(score.values[rep['Label'].values==0]), color='r', alpha=0.5, label='0')
_=plt.hist(iron(score.values[rep['Label'].values==1]), color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Score value')
plt.ylabel('Counts')
plt.title('Score hist')

# <codecell>

left_cols = ['left_'+str(i) for i in range(1,23)]
right_cols = ['right_'+str(i) for i in range(1,45)]

left_avg = rep[left_cols].mean(axis=1)
right_avg = rep[right_cols].mean(axis=1)

# <codecell>

figure(figsize=(15,10))
subplot(221)
_=plt.hist(left_avg .values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(left_avg .values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_avg hist')

subplot(222)
_=plt.hist(right_avg.values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(right_avg.values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_avg hist')

subplot(223)
diff = left_avg / (right_avg+0.06)
_=plt.hist(diff.values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = rep['Label'].values
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
_=plt.hist(rep['left_1'].values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(rep['left_1'].values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Left_1 hist')

subplot(222)
_=plt.hist(rep['right_45'].values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(rep['right_45'].values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Right_45 hist')

subplot(223)
diff = rep['left_1'] / (rep['right_45']+0.35)
_=plt.hist(diff.values[rep['Label'].values==0], color='r', alpha=0.5, label='0')
_=plt.hist(diff.values[rep['Label'].values==1], color='b', alpha=0.5, label='1')
plt.legend(loc='best')
plt.xlabel('Slope Coef. value')
plt.ylabel('Counts')
plt.title('Diff hist')

subplot(224)
from sklearn.metrics import roc_curve, auc
y_true = rep['Label'].values
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

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

# <codecell>

#session.commit("Trend Analysis. Report 1.")

