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
periods = range(1,78)

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

#%%px
param1 = 1
df_ts_rolling_sum = pd.rolling_sum(df_time_series, window=param1,axis=1)[range(param1,105)]

# <codecell>

df_ts_rolling_sum

# <codecell>

param3 = 105-param1

# <codecell>

from kernel_regression import KernelRegression

def KernelRegression_and_RollingMean(X, y, window):
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    y_kr = kr.fit(X, y).predict(X)
    y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
    return y_kr, y_rm

# <codecell>

cd kernel_regression-master/

# <codecell>

!python setup.py install

# <codecell>

from kernel_regression import KernelRegression

def KernelRegression_and_RollingMean(X, y, window):
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    y_kr = kr.fit(X, y).predict(X)
    y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
    return y_kr, y_rm

# <codecell>

# %%px
results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+range(0,param3))
# results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv')

# <codecell>

from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR

poly = PolynomialFeatures(degree=1)
lr = LinearRegression()
#lr = SVR()
#lr = Ridge(alpha=0.5)
#lr = Lasso(alpha=0.02)
#lr = OrthogonalMatchingPursuit()
#lr = Lars()
#lr = PassiveAggressiveRegressor()
#lr = IsotonicRegression(increasing='auto')

# <codecell>

print [1]*10

# <codecell>

t=10
v=16
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([45]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

subplot(232)
plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(test_sum)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0001)*1
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

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

# test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=10)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=100)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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
periods = range(1,60)

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

#%%px
param1 = 1
df_ts_rolling_sum = pd.rolling_sum(df_time_series, window=param1,axis=1)[range(param1,105)]

# <codecell>

df_ts_rolling_sum

# <codecell>

param3 = 105-param1

# <codecell>

from kernel_regression import KernelRegression

def KernelRegression_and_RollingMean(X, y, window):
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    y_kr = kr.fit(X, y).predict(X)
    y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
    return y_kr, y_rm

# <codecell>

# %%px
results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+range(0,param3))
# results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv')

# <codecell>

from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR

poly = PolynomialFeatures(degree=1)
lr = LinearRegression()
#lr = SVR()
#lr = Ridge(alpha=0.5)
#lr = Lasso(alpha=0.02)
#lr = OrthogonalMatchingPursuit()
#lr = Lars()
#lr = PassiveAggressiveRegressor()
#lr = IsotonicRegression(increasing='auto')

# <codecell>

print [1]*10

# <codecell>

t=20
v=23
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([45]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 43
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

# test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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
periods = range(1,78)

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

#%%px
param1 = 1
df_ts_rolling_sum = pd.rolling_sum(df_time_series, window=param1,axis=1)[range(param1,105)]

# <codecell>

df_ts_rolling_sum

# <codecell>

param3 = 105-param1

# <codecell>

from kernel_regression import KernelRegression

def KernelRegression_and_RollingMean(X, y, window):
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    y_kr = kr.fit(X, y).predict(X)
    y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
    return y_kr, y_rm

# <codecell>

# %%px
results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+range(0,param3))
# results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv')

# <codecell>

from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR

poly = PolynomialFeatures(degree=1)
lr = LinearRegression()
#lr = SVR()
#lr = Ridge(alpha=0.5)
#lr = Lasso(alpha=0.02)
#lr = OrthogonalMatchingPursuit()
#lr = Lars()
#lr = PassiveAggressiveRegressor()
#lr = IsotonicRegression(increasing='auto')

# <codecell>

print [1]*10

# <codecell>

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([45]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 43
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

# test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([45]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 43
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

# test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

# test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=20)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=20)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([row]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([row]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))+1
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([row]).values[0]+1
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))+1
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([row]).values[0]
        if p==1:
            p=p+1
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))+1
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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
periods = range(1,88)

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

#%%px
param1 = 1
df_ts_rolling_sum = pd.rolling_sum(df_time_series, window=param1,axis=1)[range(param1,105)]

# <codecell>

df_ts_rolling_sum

# <codecell>

param3 = 105-param1

# <codecell>

from kernel_regression import KernelRegression

def KernelRegression_and_RollingMean(X, y, window):
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    y_kr = kr.fit(X, y).predict(X)
    y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
    return y_kr, y_rm

# <codecell>

# %%px
results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+range(0,param3))
# results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv')

# <codecell>

from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR

poly = PolynomialFeatures(degree=1)
lr = LinearRegression()
#lr = SVR()
#lr = Ridge(alpha=0.5)
#lr = Lasso(alpha=0.02)
#lr = OrthogonalMatchingPursuit()
#lr = Lars()
#lr = PassiveAggressiveRegressor()
#lr = IsotonicRegression(increasing='auto')

# <codecell>

print [1]*10

# <codecell>

t=5
v=10
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([row]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))+1
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 15
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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
periods = range(1,78)

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

#%%px
param1 = 1
df_ts_rolling_sum = pd.rolling_sum(df_time_series, window=param1,axis=1)[range(param1,105)]

# <codecell>

df_ts_rolling_sum

# <codecell>

param3 = 105-param1

# <codecell>

from kernel_regression import KernelRegression

def KernelRegression_and_RollingMean(X, y, window):
    kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    y_kr = kr.fit(X, y).predict(X)
    y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
    return y_kr, y_rm

# <codecell>

# %%px
results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+range(0,param3))
# results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv')

# <codecell>

from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, Perceptron
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR

poly = PolynomialFeatures(degree=1)
lr = LinearRegression()
#lr = SVR()
#lr = Ridge(alpha=0.5)
#lr = Lasso(alpha=0.02)
#lr = OrthogonalMatchingPursuit()
#lr = Lars()
#lr = PassiveAggressiveRegressor()
#lr = IsotonicRegression(increasing='auto')

# <codecell>

print [1]*10

# <codecell>

t=10
v=15
param4 = v+t

def ANN(rows_range):
    
    keys = [i for i in range(1,param3+1)]
    results = []

    for row in rows_range:
        if row%500==0:
            print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        index = ts_train.index[0]
        max_value = ts_train.max(axis=1).values[0]
        ts_train = ts_train/(1.0*max_value)
        x = np.array([[i] for i in range(0, 105)])
        #Get train data
        x_train = x[range(param1, 105-param4)]
        y_train = ts_train[range(param1, 105-param4)].values[0]
        #y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = x[range(105-param4, 105-param4+v)]
        y_valid = ts_train[range(105-param4, 105-param4+v)].values[0]
        #y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = x[range(105-param4+v, 105)]
        y_test = ts_train[range(105-param4+v, 105)].values[0]
        #y_test = y_test.reshape(len(y_test),1)
        #Add new features
        
        # Create network with 2 layers and random initialized
#         lr.fit(poly.fit_transform(x_train), y_train)

#         # Simulate network
#         out_train = lr.predict(poly.fit_transform(x_train))
#         out_valid = lr.predict(poly.fit_transform(x_valid))
#         out_test = lr.predict(poly.fit_transform(x_test))
        p = data['nb_peaks'].irow([row]).values[0]
        w = data['inter_max'][data['nb_peaks']==p]
        window = int(w.quantile(0.9))+1
        y_kr, y_rm = KernelRegression_and_RollingMean(x_train, y_train, window)

        # Simulate network
        out_train = y_rm
        out_valid = [y_rm[-1]]*len(y_valid)
        out_test = [y_rm[-1]]*len(y_test)

#         plt.subplot(1,1,1)
#         plt.plot(y_kr, color='g')
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0), color='b')
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0), color='r')
#         plt.show()

        res = {'Index':index, 'y':np.concatenate((y_train,y_valid, y_test),axis=0),
               'y_out':np.concatenate((out_train,out_valid,out_test),axis=0), 'y_kr':y_kr}
        results.append(res)
        #Get results
        #index = ts_train.index[0]
#         error_train = mean_absolute_error(y_train, out_train)
#         error_valid = mean_absolute_error(y_valid, out_valid)
#         error_test = mean_absolute_error(y_test, out_test)
#         values = list(np.concatenate((out_train,out_valid,out_test)))
#         values = np.reshape(values,(len(values),))
#         data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
#         for i in range(1,param3+1):
#             data_dict[i] = [values[i-1]]
#         new_row = pd.DataFrame(data=data_dict)
#         results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

#%%px
#!easy_install neurolab

# <codecell>

rows = range(0,5704)#5704

# <codecell>

%%time
results = ANN(rows)

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# <codecell>

a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    res = results[row]
    plt.plot(res['y_kr'], color='g')
    plt.plot(res['y'], color='b')
    plt.plot(res['y_out'], color='r')
    plt.plot([105-t,105-t], [-1,1], color='black')
    plt.plot([105-v-t,105-v-t], [-1,1], color='black')
    plt.title('Index is '+str(res['Index']))
    #plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

a = [1,2,3,4,5]
print a[-2:-1]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

#print error hists
test_mean = []
valid_mean = []
t_mean = []
predict_mean = []
for res in results:
    #window = data['inter_max'].irow([row]).values[0]+1
    window = 25
    predict_mean.append(res['y_out'][-1])
    test_mean.append(res['y'][-window:].mean())
    t_mean.append(res['y'][-window+t:].mean())
    valid_mean.append(res['y'][-window:-window+t].mean())
    
test_mean = np.array(test_mean)
t_mean = np.array(t_mean)
valid_mean = np.array(valid_mean)
predict_mean = np.array(predict_mean)

# test_mean = test_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# t_mean = t_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# valid_mean = valid_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0) ]
# predict_mean = predict_mean[(data['inter_max'].values<25)*(data['nb_peaks'].values>0)]

# <codecell>

test_mean.shape

# <codecell>

#print error hists


figure(figsize=(15, 10))
subplot(221)
plt.hist(test_mean, color='r', bins=20, label='test', alpha=0.5)
plt.hist(predict_mean, color='b', bins=20, label='train', alpha=0.5)
plt.legend(loc='best')

subplot(222)
plt.hist(predict_mean[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(predict_mean[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(223)
diff = predict_mean-test_mean
plt.hist(diff[test_mean==0], color='r', bins=20, label='0', alpha=0.5)
plt.hist(diff[test_mean!=0], color='b', bins=20, label='>0', alpha=0.5)
plt.legend(loc='best')

subplot(224)
from matplotlib.colors import LogNorm
plt.hist2d(predict_mean,test_mean, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

#plt.show()
#plt.show()

# <codecell>

plt.scatter(test_mean, predict_mean)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm
plt.hist2d(data['inter_max'],data['nb_peaks'], norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

p = data['nb_peaks'].irow([45]).values[0]
w = data['inter_max'][data['nb_peaks']==p]
print int(w.quantile(0.9))
plt.hist(w.values)
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (test_mean>0.0001)*1
#y_score = non_nan_res[str(param3)].values
y_score = predict_mean
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

from matplotlib.colors import LogNorm

figure(figsize=(20, 10))
subplot(231)
plt.hist2d(t_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(valid_mean,predict_mean, norm=LogNorm(), bins=50)
plt.colorbar()

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(valid_mean)):
    x,y = GetCoord(xedges, yedges, valid_mean[i], predict_mean[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_true==0], label='y_true=0', alpha=0.5, bins=50)
plt.hist(y_score[y_true!=0], label = 'y_true!=0', alpha=0.5, bins=50)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()

subplot(235)
(counts, xedges, yedges, Image) = plt.hist2d(y_score,predict_mean, norm=LogNorm(), bins=20)
plt.colorbar()


from sklearn.metrics import roc_curve, auc
y_true = (test_mean>0.0000001)*1
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

session.commit("Nadaraya-Watson. Report 1.")