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

#%%px
param1 = 13
df_ts_rolling_sum = pd.rolling_sum(df_time_series, window=param1,axis=1)[range(param1,105)]

# <codecell>

#%%px
ws = 13#window_size
fh = 13#forecast horizont
param2 = 105-param1

def N_M_Transformation(time_serie, ws, fh):
    x_cols = ['x'+str(i) for i in range(1,ws+1)]#columns names
    time_serie_table = pd.DataFrame(columns=x_cols+['y'])
    time_serie_4predict = pd.DataFrame(columns=x_cols)
    #Data for train and test
    for row_num in range(0, param2-fh-ws):
        time_serie_table.loc[row_num] = list(time_serie.icol(range(row_num+1, row_num+ws+1)).values[0])\
        +list(time_serie.icol([row_num+ws+fh]).values[0])#y variable 
    #Data for prediction
    for row_num in range(param2-fh-ws,param2-ws):
        time_serie_4predict.loc[row_num-(param2-fh-ws)] = list(time_serie.icol(range(row_num+1, row_num+ws+1)).values[0]) 
        #print row_num

    return time_serie_table, time_serie_4predict

def N_M_Transformation_Bolean(time_serie, ws, fh):
    x_cols = ['x'+str(i) for i in range(1,ws+1)]#columns names
    time_serie_table = pd.DataFrame(columns=x_cols+['y'])
    time_serie_4predict = pd.DataFrame(columns=x_cols)
    #Data for train and test
    for row_num in range(0, param2-fh-ws):
        time_serie_table.loc[row_num] = list(time_serie.icol(range(row_num+1, row_num+ws+1)).values[0])\
        +list((time_serie.icol([row_num+ws+fh]).values[0]>0)*1)#y variable 
    #Data for prediction
    for row_num in range(param2-fh-ws,param2-ws):
        time_serie_4predict.loc[row_num-(param2-fh-ws)] = list(time_serie.icol(range(row_num+1, row_num+ws+1)).values[0]) 
        #print row_num

    return time_serie_table, time_serie_4predict

# <codecell>

#%%px
param3 = param2-fh-ws
print param3

# <codecell>

# %%px
# results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+range(0,param3))
# results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv')

# <codecell>

def ANN(rows_range):
    
    import neurolab as nl
    keys = [str(i) for i in range(1,param3+1)]
    results = pd.DataFrame(columns=["Index","Error_train","Error_valid", "Error_test"]+keys)

    param4 = fh+10
    f = nl.trans.TanSig()

    for row in rows_range:
        print row
        #Take a row and transfrom it
        ts_train = df_ts_rolling_sum.irow([row])
        #time_serie_table, time_serie_4predict = N_M_Transformation_Bolean(ts_train, ws, fh)
        time_serie_table, time_serie_4predict = N_M_Transformation(ts_train, ws, fh)
        max_value = ts_train.max(axis=1).values[0]
        #Transform the row's values to the [0,1] values
        #time_serie_table['y'] = max_value*time_serie_table['y'].values
        time_serie_table = time_serie_table/(1.0*max_value)
        time_serie_4predict = time_serie_4predict/(1.0*max_value)
        x_cols = ['x'+str(i) for i in range(1,ws+1)]
        #Get train data
        x_train = time_serie_table[x_cols].irow(range(0,param3-param4)).values
        y_train = time_serie_table['y'].irow(range(0,param3-param4)).values
        size = len(y_train)
        y_train = y_train.reshape(len(y_train),1)
        #Get validation data
        x_valid = time_serie_table[x_cols].irow(range(param3-param4,param3-fh)).values
        y_valid = time_serie_table['y'].irow(range(param3-param4,param3-fh)).values
        y_valid = y_valid.reshape(len(y_valid),1)
        #Get test data
        x_test = time_serie_table[x_cols].irow(range(param3-fh,param3)).values
        y_test = time_serie_table['y'].irow(range(param3-fh,param3)).values
        y_test = y_test.reshape(len(y_test),1)
        # Create network with 2 layers and random initialized
        init = []
        for i in range(0, x_train.shape[1]):
            init.append([0,1])
        min_error = 10
        for k in range(0,20):
            cur_net = nl.net.newff(init,[5,1],transf=[f, f])
            for l in cur_net.layers:
                l.initf = nl.init.init_rand(l, min=-0.5, max=0.5, init_prop='w')
            # new initialization
            cur_net.init()
            if k==0:
                net=cur_net
                error = 10

            # Train network
            cur_net.trainf = nl.train.train_bfgs
            cur_error = cur_net.train(x_train, y_train, epochs=50, show=0, goal=0.0001)

            out_train = cur_net.sim(x_train)
            out_valid = cur_net.sim(x_valid)

            tar_out_valid = np.concatenate((y_valid, out_valid), axis=1)
            tar_out_train = np.concatenate((y_train, out_train), axis=1)
            tar_out = np.concatenate((tar_out_train, tar_out_valid), axis=0)
            max_abs = np.abs(tar_out[:,0]-tar_out[:,1])
            maef = nl.error.MAE()
            saef = nl.error.SAE()
            msef = nl.error.MSE()
            #check_error = maef(tar_out_valid)
            #check_error = msef(tar_out)
            #check_error = saef(tar_out)
            #check_error = cur_error[-1]
            check_error = max_abs.max()

            print check_error
            if check_error<min_error:
                min_error = check_error
                net = cur_net
                error = cur_error


        # Simulate network
        print 'min_error', min_error
        out_train = net.sim(x_train)

        # Plot result
#         plt.subplot(1,1,1)
#         plt.plot(error)
#         plt.xlabel('Epoch number')
#         plt.ylabel('error (default SSE)')
#         plt.show()

        out_valid = net.sim(x_valid)
        out_test = net.sim(x_test)
#         plt.subplot(1,1,1)
#         plt.plot(np.concatenate((y_train,y_valid, y_test),axis=0))
#         plt.plot(np.concatenate((out_train,out_valid,out_test),axis=0))
#         plt.show()


        #Get results
        index = ts_train.index[0]
        error_train = maef(tar_out_train)
        error_valid = maef(tar_out_valid)
        tar_out_test = np.concatenate((y_test, out_test), axis=1)
        error_test = maef(tar_out_test)
        values = list(np.concatenate((out_train,out_valid,out_test)))
        values = np.reshape(values,(len(values),))
        data_dict = {"Index":[index],"Error_train":[error_train],"Error_valid":[error_valid], "Error_test":[error_test]}
        for i in range(1,param3+1):
            data_dict[str(i)] = [values[i-1]]
        new_row = pd.DataFrame(data=data_dict)
        results = results.append(new_row)
        
    #results.to_csv('/mnt/w76/notebook/datasets/mikhail/ann_res.csv',mode='a',header=False)
    return results

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('ann_res_50.csv')
results.columns

# <codecell>

results['nb_peaks'] = [data['nb_peaks'].ix[int(i)] for i in results['Index'].values]

# <codecell>

df_ts_rolling_sum.columns
#df_ts_rolling_sum = (df_ts_rolling_sum>0)*1

# <codecell>

val_cols = [str(i) for i in range(1,67)]  
non_nan_res = results[(pd.isnull(results).sum(axis=1)==0)*(results['Error_valid']<=1)*(results['Error_train']<=1)*\
                      (results['nb_peaks']>=0)]
#non_nan_res[val_cols] = (non_nan_res[val_cols].values>=0.95)*1
non_nan_res.shape

# <codecell>

max_values = df_ts_rolling_sum.max(axis=1)
df_ts_rolling_sum_std = df_ts_rolling_sum.copy()
for col in df_ts_rolling_sum.columns:
    df_ts_rolling_sum_std[col] = df_ts_rolling_sum[col]/max_values

# <codecell>

#print error hists
figure(figsize=(15, 5))
subplot(121)
plt.hist(non_nan_res['Error_test'].values, color='r', bins=20, label='test', alpha=1, histtype='step')
plt.hist(non_nan_res['Error_train'].values, color='b', bins=20, label='train', alpha=1, histtype='step')
plt.hist(non_nan_res['Error_valid'].values, color='g', bins=20, label='valid', alpha=1, histtype='step')
plt.title('Errors')
plt.legend(loc='best')
#plt.show()

#print predict value for the last point
subplot(122)
plt.hist(non_nan_res['66'].values, bins=10, label='last point')
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

# <codecell>

y_last=[]
for i in non_nan_res['Index']:
    i=int(i)
    cur_serie = df_ts_rolling_sum.xs(i).values
    y_last.append(cur_serie[104-fh]/(1.0*cur_serie.max()))
y_last = np.array(y_last)

# <codecell>

non_nan_res[y_last==0].shape

# <codecell>

figure(figsize=(15, 10))
#print predict value for the last point
subplot(2,2,1)
values = non_nan_res['66'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,2)
values = non_nan_res['Error_test'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_test')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,3)
values = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Relative valid error')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,4)
values = non_nan_res['Error_valid'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_valid')
plt.legend(loc='best')
#plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
y_score = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
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

plt.hist2d(non_nan_res['66'].values, y_last)
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, y_last)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, y_last, norm=LogNorm())
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, y_last, norm=LogNorm(), bins=50)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, y_last, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, non_nan_res['Error_train'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, non_nan_res['Error_valid'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, non_nan_res['Error_test'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['66'].values, non_nan_res['Error_train'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['Error_train'].values, non_nan_res['Error_train'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['Error_valid'].values, non_nan_res['Error_train'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['Error_test'].values, non_nan_res['Error_train'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(non_nan_res['Error_test'].values, non_nan_res['Error_valid'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

avg_value_predict_test = []
avg_value_true_test = []
avg_value_predict_valid = []
avg_value_true_valid = []
test_cols = [str(i) for i in range(53,66)]
valid_cols = [str(i) for i in range(43,53)]

for row in range(0,non_nan_res.shape[0]):
    avg_val_pred_test = non_nan_res[test_cols].irow([row]).mean(axis=1).values[0]
    avg_value_predict_test.append(avg_val_pred_test)
    avg_val_true_test = df_ts_rolling_sum_std[range(92,105)].irow([row]).mean(axis=1).values[0]
    avg_value_true_test.append(avg_val_true_test)
    
    avg_val_pred_valid = non_nan_res[valid_cols].irow([row]).mean(axis=1).values[0]
    avg_value_predict_valid.append(avg_val_pred_valid)
    avg_val_true_valid = df_ts_rolling_sum_std[range(82,92)].irow([row]).mean(axis=1).values[0]
    avg_value_true_valid.append(avg_val_true_valid)
    
avg_value_predict_test = np.array(avg_value_predict_test)
avg_value_true_test = np.array(avg_value_true_test)
avg_value_predict_valid = np.array(avg_value_predict_valid)
avg_value_true_valid = np.array(avg_value_true_valid)

# <codecell>

figure(figsize=(15, 10))

subplot(2,2,1)
values = avg_value_predict_test
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Predict values')
plt.legend(loc='best')

subplot(2,2,2)
values = avg_value_predict_valid - avg_value_true_valid
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Error valid')
plt.legend(loc='best')

subplot(2,2,3)
values = (avg_value_predict_valid - avg_value_true_valid)/(avg_value_predict_test+2.0)
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Relative valid error')
plt.legend(loc='best')

subplot(2,2,4)
values = avg_value_predict_valid - avg_value_true_valid
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Error_valid')
plt.legend(loc='best')

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true_avg = (avg_value_true_test>0)*1
#y_score_avg = 0.5*(avg_value_predict_test+2.0)
y_score_avg = 0.5*(avg_value_predict_valid - avg_value_true_valid)/(avg_value_predict_test+2.0)+0.5
fpr_avg, tpr_avg, _ = roc_curve(y_true_avg, y_score_avg, pos_label=None, sample_weight=None)
roc_auc_avg = auc(fpr_avg, tpr_avg)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr_avg, tpr_avg)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc_avg

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(avg_value_predict_test, avg_value_true_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(avg_value_predict_valid, avg_value_true_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(avg_value_predict_test, avg_value_true_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(avg_value_predict_test, avg_value_true_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

plt.hist2d(avg_value_predict_test, avg_value_true_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

(counts, xedges, yedges, Image) = plt.hist2d(non_nan_res['66'].values, y_last, norm=LogNorm(), bins=20)
plt.colorbar()
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
    x,y = GetCoord(xedges, yedges, non_nan_res['66'].values[i], y_last[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, non_nan_res['66'].values[i], y_last[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(y_score, y_last, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
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

y_last=[]
y_valid_last = []
for i in non_nan_res['Index']:
    i=int(i)
    cur_serie = df_ts_rolling_sum.xs(i).values
    y_last.append(cur_serie[104-fh]/(1.0*cur_serie.max()))
    y_valid_last.append(cur_serie[104-fh-13]/(1.0*cur_serie.max()))
y_last = np.array(y_last)
y_valid_last = np.array(y_valid_last)

# <codecell>

from matplotlib.colors import LogNorm

(counts, xedges, yedges, Image) = plt.hist2d(non_nan_res['66'].values, y_valid_last, norm=LogNorm(), bins=20)
plt.colorbar()
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
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, non_nan_res['66'].values[i], y_valid_last[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(y_score, y_last, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
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

from matplotlib.colors import LogNorm

(counts, xedges, yedges, Image) = plt.hist2d(avg_value_predict_test, avg_value_true_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from matplotlib.colors import LogNorm

(counts, xedges, yedges, Image) = plt.hist2d(avg_value_predict_valid, avg_value_true_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_predict_valid, y_valid_last[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_predict_valid[i], avg_value_true_valid[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

from matplotlib.colors import LogNorm

(counts, xedges, yedges, Image) = plt.hist2d(non_nan_res['66'].values, y_valid_last, norm=LogNorm(), bins=20)
plt.colorbar()
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
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, non_nan_res['66'].values[i], y_valid_last[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(y_score, y_last, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
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

plt.hist2d(y_score, avg_value_true_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_valid>0)*1
#y_score = non_nan_res['66'].values
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

plt.hist2d(y_score, avg_value_true_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_test>0)*1
#y_score = non_nan_res['66'].values
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

from matplotlib.colors import LogNorm

(counts, xedges, yedges, Image) = plt.hist2d(non_nan_res['53'].values, y_valid_last, norm=LogNorm(), bins=20)
plt.colorbar()
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
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, non_nan_res['53'].values[i], y_valid_last[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

(counts, xedges, yedges, Image) = plt.hist2d(y_score, y_last, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
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

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('Test distribution')

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('Test distribution')

subplot(232)
plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('Valid distribution')

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('Test distribution')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('Valid distribution')

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(233)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(y_valid_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(y_valid_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
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
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(y_valid_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0.1)*1
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
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(y_valid_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
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
plt.hist2d(avg_value_true_test, avg_value_predict_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(avg_value_true_valid, avg_value_predict_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_true_valid[i], avg_value_predict_valid[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[avg_value_true_test==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[avg_value_true_test!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(avg_value_true_test, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(avg_value_true_valid, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_test>0)*1
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

(counts, xedges, yedges, Image) = plt.hist2d(avg_value_predict_valid, avg_value_true_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_predict_valid[i], avg_value_true_valid[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

# <codecell>

plt.hist2d(y_score, avg_value_true_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_test>0)*1
#y_score = non_nan_res['66'].values
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

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(avg_value_true_test, avg_value_predict_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(avg_value_true_valid, avg_value_predict_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_true_valid[i], avg_value_predict_valid[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[avg_value_true_test==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[avg_value_true_test!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(avg_value_true_test, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(avg_value_true_valid, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_valid>0)*1
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
plt.hist2d(avg_value_true_test, avg_value_predict_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(avg_value_true_valid, avg_value_predict_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_true_valid[i], avg_value_predict_valid[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[avg_value_true_test==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[avg_value_true_test!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(avg_value_true_test, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(avg_value_true_valid, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_test>0)*1
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true_avg = (avg_value_true_test>0)*1
#y_score_avg = 0.5*(avg_value_predict_test+2.0)
y_score_avg = 0.5*(avg_value_predict_valid - avg_value_true_valid)/(avg_value_predict_test+2.0)+0.5
fpr_avg, tpr_avg, _ = roc_curve(y_true_avg, 1-y_score_avg, pos_label=None, sample_weight=None)
roc_auc_avg = auc(fpr_avg, tpr_avg)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr_avg, tpr_avg)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc_avg

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true_avg = (avg_value_true_test>0)*1
#y_score_avg = 0.5*(avg_value_predict_test+2.0)
y_score_avg = 0.5*(avg_value_predict_valid - avg_value_true_valid)/(avg_value_predict_test+2.0)+0.5
fpr_avg, tpr_avg, _ = roc_curve(y_true_avg, y_score_avg, pos_label=None, sample_weight=None)
roc_auc_avg = auc(fpr_avg, tpr_avg)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr_avg, tpr_avg)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc_avg

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('ann_res_50.csv')
results['nb_peaks'] = [data['nb_peaks'].ix[int(i)] for i in results['Index'].values]

val_cols = [str(i) for i in range(1,67)]  
non_nan_res = results[(pd.isnull(results).sum(axis=1)==0)*(results['Error_valid']<=0.5)*(results['Error_train']<=0.05)*\
                      (results['nb_peaks']>=0)]
non_nan_res.shape

# <codecell>

max_values = df_ts_rolling_sum.max(axis=1)
df_ts_rolling_sum_std = df_ts_rolling_sum.copy()
for col in df_ts_rolling_sum.columns:
    df_ts_rolling_sum_std[col] = df_ts_rolling_sum[col]/max_values

# <codecell>

val_cols = [str(i) for i in range(1,67)]
val_x = range(105-66,105)
cols = range(13,105)
a=0
b=60
N=b-a
figure(figsize=(15, 5*(N//3+1)))
for row in range(a,b):
    subplot(N//3+1,3,row)
    plt.plot(val_x,non_nan_res[val_cols].irow([row]).values[0], color='r', label='predict')
    index = int(non_nan_res.irow([row])['Index'].values)
    plt.plot(cols, df_ts_rolling_sum_std[cols].xs(index), color='b', label='real')
    plt.plot([param3+fh+ws,param3+fh+ws], [-1,1], color='black')
    plt.plot([param3+fh-10+ws,param3+fh-10+ws], [-1,1], color='black')
    plt.title('Index is '+str(index))
    plt.xlim(ws,105)
    plt.ylim(-1,1.1)
    plt.legend(loc='best')
    #plt.show()

# <codecell>

#print error hists
figure(figsize=(15, 5))
subplot(121)
plt.hist(non_nan_res['Error_test'].values, color='r', bins=20, label='test', alpha=1, histtype='step')
plt.hist(non_nan_res['Error_train'].values, color='b', bins=20, label='train', alpha=1, histtype='step')
plt.hist(non_nan_res['Error_valid'].values, color='g', bins=20, label='valid', alpha=1, histtype='step')
plt.title('Errors')
plt.legend(loc='best')
#plt.show()

#print predict value for the last point
subplot(122)
plt.hist(non_nan_res['66'].values, bins=10, label='last point')
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

# <codecell>

y_last=[]
for i in non_nan_res['Index']:
    i=int(i)
    cur_serie = df_ts_rolling_sum.xs(i).values
    y_last.append(cur_serie[104-fh]/(1.0*cur_serie.max()))
y_last = np.array(y_last)
non_nan_res[y_last==0].shape

# <codecell>

figure(figsize=(15, 10))
#print predict value for the last point
subplot(2,2,1)
values = non_nan_res['66'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,2)
values = non_nan_res['Error_test'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_test')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,3)
values = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Relative valid error')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,4)
values = non_nan_res['Error_valid'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_valid')
plt.legend(loc='best')
#plt.show()

# <codecell>

#print error hists
figure(figsize=(15, 5))
subplot(121)
plt.hist(non_nan_res['Error_test'].values, color='r', bins=20, label='test', alpha=1, histtype='step')
plt.hist(non_nan_res['Error_train'].values, color='b', bins=20, label='train', alpha=1, histtype='step')
plt.hist(non_nan_res['Error_valid'].values, color='g', bins=20, label='valid', alpha=1, histtype='step')
plt.title('Errors')
plt.legend(loc='best')
#plt.show()

#print predict value for the last point
subplot(122)
plt.hist(non_nan_res['66'].values, bins=10, label='last point')
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

# <codecell>

y_last=[]
for i in non_nan_res['Index']:
    i=int(i)
    cur_serie = df_ts_rolling_sum.xs(i).values
    y_last.append(cur_serie[104-fh]/(1.0*cur_serie.max()))
y_last = np.array(y_last)
non_nan_res[y_last==0].shape

# <codecell>

figure(figsize=(15, 10))
#print predict value for the last point
subplot(2,2,1)
values = non_nan_res['66'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,2)
values = non_nan_res['Error_test'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_test')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,3)
values = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Relative valid error')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,4)
values = non_nan_res['Error_valid'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_valid')
plt.legend(loc='best')
#plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
y_score = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
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
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(y_valid_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

y_last=[]
y_valid_last = []
for i in non_nan_res['Index']:
    i=int(i)
    cur_serie = df_ts_rolling_sum.xs(i).values
    y_last.append(cur_serie[104-fh]/(1.0*cur_serie.max()))
    y_valid_last.append(cur_serie[104-fh-13]/(1.0*cur_serie.max()))
y_last = np.array(y_last)
y_valid_last = np.array(y_valid_last)
non_nan_res[y_last==0].shape

# <codecell>

figure(figsize=(15, 10))
#print predict value for the last point
subplot(2,2,1)
values = non_nan_res['66'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Predict values')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,2)
values = non_nan_res['Error_test'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_test')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,3)
values = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Relative valid error')
plt.legend(loc='best')
#plt.show()

#print predict value for 66th week
subplot(2,2,4)
values = non_nan_res['Error_valid'].values
plt.hist(values[y_last==0], bins=10, label='y_last=0', alpha=0.5)
plt.hist(values[y_last!=0], bins=10, label='y_last!=0', alpha=0.5)
plt.title('Error_valid')
plt.legend(loc='best')
#plt.show()

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
#y_score = non_nan_res['66'].values
y_score = non_nan_res['Error_valid'].values/(non_nan_res['66'].values+2.0)
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
plt.hist2d(y_last, non_nan_res['66'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(y_valid_last, non_nan_res['53'].values, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, y_valid_last[i], non_nan_res['53'].values[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[y_last==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[y_last!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(y_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(y_valid_last, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (y_last>0)*1
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
roc_auc = auc(fpr, tpr)

subplot(2,3,6)
plt.plot(fpr, tpr, label='ROC auc = '+str(roc_auc))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# <codecell>

avg_value_predict_test = []
avg_value_true_test = []
avg_value_predict_valid = []
avg_value_true_valid = []
test_cols = [str(i) for i in range(53,66)]
valid_cols = [str(i) for i in range(43,53)]

for row in range(0,non_nan_res.shape[0]):
    avg_val_pred_test = non_nan_res[test_cols].irow([row]).mean(axis=1).values[0]
    avg_value_predict_test.append(avg_val_pred_test)
    avg_val_true_test = df_ts_rolling_sum_std[range(92,105)].irow([row]).mean(axis=1).values[0]
    avg_value_true_test.append(avg_val_true_test)
    
    avg_val_pred_valid = non_nan_res[valid_cols].irow([row]).mean(axis=1).values[0]
    avg_value_predict_valid.append(avg_val_pred_valid)
    avg_val_true_valid = df_ts_rolling_sum_std[range(82,92)].irow([row]).mean(axis=1).values[0]
    avg_value_true_valid.append(avg_val_true_valid)
    
avg_value_predict_test = np.array(avg_value_predict_test)
avg_value_true_test = np.array(avg_value_true_test)
avg_value_predict_valid = np.array(avg_value_predict_valid)
avg_value_true_valid = np.array(avg_value_true_valid)

# <codecell>

figure(figsize=(15, 10))

subplot(2,2,1)
values = avg_value_predict_test
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Predict values')
plt.legend(loc='best')

subplot(2,2,2)
values = avg_value_predict_valid - avg_value_true_valid
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Error valid')
plt.legend(loc='best')

subplot(2,2,3)
values = (avg_value_predict_valid - avg_value_true_valid)/(avg_value_predict_test+2.0)
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Relative valid error')
plt.legend(loc='best')

subplot(2,2,4)
values = avg_value_predict_valid - avg_value_true_valid
plt.hist(values[avg_value_true_test==0], bins=20, label='avg_value_true=0', alpha=0.5)
plt.hist(values[avg_value_true_test!=0], bins=20, label='avg_value_true!=0', alpha=0.5)
plt.title('Error_valid')
plt.legend(loc='best')

# <codecell>

from sklearn.metrics import roc_curve, auc

y_true_avg = (avg_value_true_test>0)*1
#y_score_avg = 0.5*(avg_value_predict_test+2.0)
y_score_avg = 0.5*(avg_value_predict_valid - avg_value_true_valid)/(avg_value_predict_test+2.0)+0.5
fpr_avg, tpr_avg, _ = roc_curve(y_true_avg, y_score_avg, pos_label=None, sample_weight=None)
roc_auc_avg = auc(fpr_avg, tpr_avg)

figure(figsize=(15, 5))
subplot(1,2,1)
plt.plot(fpr_avg, tpr_avg)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print 'ROC AUC is ', roc_auc_avg

# <codecell>

figure(figsize=(20, 10))

subplot(231)
plt.hist2d(avg_value_true_test, avg_value_predict_test, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in test')
plt.ylabel('Predicted value of the last point in test')
plt.title('LogNormed histogram for test')

subplot(232)
(counts, xedges, yedges, Image) = plt.hist2d(avg_value_true_valid, avg_value_predict_valid, norm=LogNorm(), bins=20)
plt.colorbar()
plt.xlabel('Value of the last point in valid')
plt.ylabel('Predicted value of the last point in valid')
plt.title('LogNormed histogram for valid')

counts_std = counts/counts.max()
y_score = []
for i in range(0, len(y_last)):
    x,y = GetCoord(xedges, yedges, avg_value_true_valid[i], avg_value_predict_valid[i])
    y_score.append(1-counts_std[x,y])
y_score = np.array(y_score)

subplot(2,3,3)
plt.hist(y_score[avg_value_true_test==0], label='y_true=0', alpha=0.5)
plt.hist(y_score[avg_value_true_test!=0], label = 'y_true!=0', alpha=0.5)
plt.legend(loc='best')
plt.title("y_score distribution")

subplot(234)
plt.hist2d(avg_value_true_test, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in test')
plt.ylabel('y_score')
plt.title('LogNormed histogram for test')
plt.colorbar()

subplot(235)
plt.hist2d(avg_value_true_valid, y_score, norm=LogNorm(), bins=20)
plt.xlabel('Value of the last point in valid')
plt.ylabel('y_score')
plt.title('LogNormed histogram for valid')
plt.colorbar()

from sklearn.metrics import roc_curve, auc

y_true = (avg_value_true_test>0)*1
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

session.commit("ANN of Neurolab. Report 2. y_score added.")

# <codecell>

session.commit("ANN of Neurolab. Report 2. y_score added.")

# <codecell>

import ipykee
#ipykee.create_project(project_name="D._UsageForecast", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="D._UsageForecast")

# <codecell>

session.commit("ANN of Neurolab. Report 2. y_score added.")