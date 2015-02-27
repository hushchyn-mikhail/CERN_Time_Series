# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data_path = 'popularity-910days.csv'
data = pd.read_csv(data_path)

data.irow(range(0,2))

# <codecell>

from DataPopularity import DataPopularityEstimator

estimator = DataPopularityEstimator(data=data, nb_of_weeks=130)
%time estimator.train()
estimator.roc_curve()

# <codecell>

cd DataPopularity/Packages/kernel_regression-master/

# <codecell>

!pip install .

# <codecell>

cd ../

# <codecell>

cd ../

# <codecell>

!pip install .

# <codecell>

from DataPopularity import DataPopularityEstimator

estimator = DataPopularityEstimator(data=data, nb_of_weeks=130)
%time estimator.train()
estimator.roc_curve()

# <codecell>

popularity = estimator.get_popularity()
popularity.irow(range(0,5))

# <codecell>

cut = estimator.popularity_cut_fpr(fpr_value=0.05)
cut

# <codecell>

from DataPopularity.DataAccessPredictor import DataAccessPredictor

predictor = DataAccessPredictor(data=data, nb_of_weeks=130)
%time prediction = predictor.predict(zero_one_scale=False)

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk']
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    set_disk_tape_marker(total_report, pop_cut)
    set_nb_replicas(total_report, q=[0.5, 0.75, 0.95])
    set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
#total_size = (data['DiskSize'].values).sum()
total_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

x = data.sort(columns='Name')['Nb_Replicas'].values
y = total_report['NbReplicas'].values
plt.scatter(x,y)
plt.show()

# <codecell>

def loss_function2(total_report, v_disk=1, v_tape=100):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    inten = total_report['Intensity']
    
    t_disk = (inten/nb_replicas*lfn_size/v_disk*marker).sum()
    t_tape = (inten/nb_replicas*lfn_size/v_tape*(1-marker)).sum()
    total_time = t_disk + t_tape
    
    disk_size = (nb_replicas*lfn_size*marker).sum()
    return total_time, disk_size

def get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=100):
    cuts = np.array(arange(0, 1, 0.01))
    total_time = []
    disk_size = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        time, size = loss_function2(report, v_disk, v_tape)
        total_time.append(time)
        disk_size.append(size)
    total_time = np.array(total_time)
    disk_size = np.array(disk_size)

    return cuts, total_time, disk_size

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=100, v_tape=1)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.title('Total time')
plt.xlabel('Popularity cut')
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.title('Disk Size')
plt.xlabel('Popularity cut')
plt.subplot(223)
a = 0.5
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())
plt.title('Diff')
plt.xlabel('Popularity cut')

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=[0.7, 0.8, 0.9])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
#total_size = (data['DiskSize'].values).sum()
total_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=[0.7, 0.8, 0.9])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=[0.9, 0.95, 0.97])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=[0.1, 0.2, 0.3])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    set_disk_tape_marker(total_report, pop_cut)
    set_nb_replicas(total_report, q=[0.5, 0.75, 0.95])
    set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
#total_size = (data['DiskSize'].values).sum()
total_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=[0.1, 0.2, 0.3])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

x = data.sort(columns='Name')['Nb_Replicas'].values
y = total_report['NbReplicas'].values
plt.scatter(x,y)
plt.show()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.1, 0.2, 0.3])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.1, 0.2, 0.5])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.1, 0.2, 0.1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.1, 0.1, 0.1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas']

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.5, 0.7, 0.8])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=[0.5, 0.75, 0.95])
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
#total_size = (data['DiskSize'].values).sum()
total_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.5, 0.7, 0.8])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.1, 0.7, 0.8])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
#total_size = (data['DiskSize'].values).sum()
total_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.1, 0.7, 0.8])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.7, 0.8, 0.9])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.9, 0.95, 0.99])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
#     for i in q:
#         total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
#total_size = (data['DiskSize'].values).sum()
total_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.9, 0.95, 0.99])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.9, 0.95, 0.99])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['DiskSize'].values\
+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
#     for i in q:
#         total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[0.9, 0.95, 0.99])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report['NbReplicas'].sum()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.5, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.9, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.1, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.01, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.001, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.01, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=10000, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100000, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100000, c_tape=1, c_miss=700000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
#     for i in q:
#         total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.01, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['DiskSize'].values\
+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
#     for i in q:
#         total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.01, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data_path = 'popularity-910days.csv'
data = pd.read_csv(data_path)
data = data[data['Storage']=='Disk']

data.irow(range(0,2))

# <codecell>

cd ../

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data_path = 'popularity-910days.csv'
data = pd.read_csv(data_path)
data = data[data['Storage']=='Disk']

data.irow(range(0,2))

# <codecell>

from DataPopularity import DataPopularityEstimator

estimator = DataPopularityEstimator(data=data, nb_of_weeks=130)
%time estimator.train()
estimator.roc_curve()

# <codecell>

popularity = estimator.get_popularity()
popularity.irow(range(0,5))

# <codecell>

cut = estimator.popularity_cut_fpr(fpr_value=0.05)
cut

# <codecell>

from DataPopularity.DataAccessPredictor import DataAccessPredictor

predictor = DataAccessPredictor(data=data, nb_of_weeks=130)
%time prediction = predictor.predict(zero_one_scale=False)

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['DiskSize'].values\
+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

pop = estimator.get_popularity()['Popularity'].values
label = estimator.train_report['Label'].values

plt.hist(pop[label==1], color='b', alpha=0.5, label='1')
plt.hist(pop[label==0], color='r', alpha=0.5, label='0')
plt.hist(pop[pop>=cut], color='g', alpha=1, label='fpr<=fpr_value', histtype='step', linewidth=3)
plt.legend(loc='best')
plt.title('Popularity hists')
plt.show()

# <codecell>

data.shape

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
#     for i in q:
#         total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.2, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.6, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=1000, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=10000, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=10000, c_tape=1, c_miss=100000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.14, q=[1, 1, 1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=10000, c_tape=1, c_miss=100000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.14, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.14, q=[0.8, 0.9, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.14, q=[0.9, 0.95, 0.98])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.5, q=[0.9, 0.95, 0.98])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.4, q=[0.9, 0.95, 0.98])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['LFNSize']#data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=10000, c_tape=1, c_miss=100000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.4, q=[0.9, 0.95, 0.98])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.4, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['LFNSize']#data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

total_report

# <codecell>

data.sort(columns='Name')

# <codecell>

data.sort(columns='Name')['LFNSize']

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['LFNSize'].values#data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

data.sort(columns='Name')['LFNSize']
total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=10000, c_tape=1, c_miss=100000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.4, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

plt.hist(prediction)
plt.show()

# <codecell>

plt.hist(prediction['Access'].values)
plt.show()

# <codecell>

plt.hist(np.log(prediction['Access'].values)
plt.show()

# <codecell>

plt.hist(np.log(prediction['Access'].values))
plt.show()

# <codecell>

plt.hist(np.log(prediction['Access'].values+1))
plt.show()

# <codecell>

plt.hist(np.log(prediction['Access'].values+1), bins=20)
plt.show()

# <codecell>

plt.hist(np.log(prediction['Access'].values+1), bins=40)
plt.show()

# <codecell>

p = prediction['Access'].values
p[p<1].shape

# <codecell>

p = prediction['Access'].values
p[p>0].shape

# <codecell>

p = prediction['Access'].values
p[p>0].shape

# <codecell>

p = prediction['Access'].values
p[p>0].shape
p.shape

# <codecell>

p = prediction['Access'].values
print p.shape
p[p>0].shape

# <codecell>

p = prediction['Access'].values
print p.shape
p[p>=0].shape

# <codecell>

p = prediction['Access'].values
print p.shape
p[p==0].shape

# <codecell>

p = prediction['Access'].values
print p.shape
p[p==0].shape
p[p==0]

# <codecell>

p = prediction['Access'].values
print p.shape
p[p==0].shape

# <codecell>

p = prediction['Access'].values
print p.shape

plt.hist(np.log(p[p==0]+1), bins=40)
plt.show()
p[p==0].shape

# <codecell>

p = prediction['Access'].values
print p.shape

plt.hist(np.log(p[p>0]+1), bins=40)
plt.show()
p[p==0].shape

# <codecell>

p = prediction['Access'].values
print p.shape

plt.hist(np.log(p[p>1]+1), bins=40)
plt.show()
p[p>0].shape

# <codecell>

p = prediction['Access'].values
print p.shape

plt.hist(np.log(p[p>0.01]+1), bins=40)
plt.show()
p[p>0].shape

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][marker==1]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[1])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

p = prediction['Access'].values
print p.shape

plt.hist(p[p>0.01], bins=40)
plt.show()
p[p>0].shape

# <codecell>

p = prediction['Access'].values
print p.shape

plt.hist(np.log(p[p>0.01]+1), bins=40)
plt.show()
p[p>0].shape

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][(marker==1)*(inten>0.01)]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][(marker==1)*(inten>0.01)]
    total_report['NbReplicas'] = 1
#     for i in q:
#         total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=70000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.7, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.65, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.59, q=[0.5, 0.75, 0.95])
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = popularity.sort(columns='Name')
total_report['Intensity'] = prediction.sort(columns='Name')['Access']
total_report['Label'] = estimator.train_report.sort(columns='Name')['Label']
total_report['LFNSize'] = data.sort(columns='Name')['LFNSize'].values#data.sort(columns='Name')['DiskSize'].values#\
#+data.sort(columns='Name')['TapeSize'].values#data.sort(columns='Name')['LFNSize']
#total_report

# <codecell>

def set_disk_tape_marker(total_report, pop_cut=0.5):
    pop = total_report['Popularity'].values
    total_report['OnDisk'] = (pop<=pop_cut)*1
    return total_report

def set_nb_replicas(total_report, q=[0.5, 0.75, 0.95]):
    marker = total_report['OnDisk'].values
    inten = total_report['Intensity']
    inten_cut = total_report['Intensity'][(marker==1)*(inten>0.01)]
    total_report['NbReplicas'] = 1
    for i in q:
        total_report['NbReplicas'] = total_report['NbReplicas'].values + (marker==1)*(inten.values>inten_cut.quantile(i))*1
    return total_report

def set_missing(total_report):
    marker = total_report['OnDisk']
    label = total_report['Label']
    total_report['Missing'] = (label==0)*(marker==0)*1
    return total_report

def report_upgrade(total_report, pop_cut=0.5, q=[0.5, 0.75, 0.95]):
    total_report = set_disk_tape_marker(total_report, pop_cut)
    total_report = set_nb_replicas(total_report, q=q)
    total_report = set_missing(total_report)
    return total_report

# <codecell>

def loss_function(total_report, c_disk=100, c_tape=1, c_miss=10000):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    
    nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
    nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
    nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
    
    loss = (nc_disk + nc_tape + nc_miss)
    return loss

def get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000):
    cuts = np.array(arange(0, 1, 0.01))
    loss_curve = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        loss = loss_function(report, c_disk, c_tape, c_miss)
        loss_curve.append(loss)
    loss_curve = np.array(loss_curve)
    min_loss = loss_curve.min()
    min_cut = cuts[loss_curve==min_loss]
    return cuts, loss_curve, min_cut, min_loss

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.5, 0.75, 0.95], c_disk=100, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.25, 0.5, 0.75], c_disk=100, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.25, 0.5, 0.75], c_disk=200, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=[0.25, 0.5, 0.75], c_disk=100, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

q = [0.25, 0.5, 0.75]
cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=q, c_disk=100, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.59, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

x = data.sort(columns='Name')['Nb_Replicas'].values
y = total_report['NbReplicas'].values
plt.scatter(x,y)
plt.show()

# <codecell>

x = data.sort(columns='Name')['Nb_Replicas'].values
y = total_report['NbReplicas'].values
plt.scatter(x,y, alpha=0.1)
plt.show()

# <codecell>

def loss_function2(total_report, v_disk=1, v_tape=100):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    inten = total_report['Intensity']
    
    t_disk = (inten/nb_replicas*lfn_size/v_disk*marker).sum()
    t_tape = (inten/nb_replicas*lfn_size/v_tape*(1-marker)).sum()
    total_time = t_disk + t_tape
    
    disk_size = (nb_replicas*lfn_size*marker).sum()
    return total_time, disk_size

def get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=100):
    cuts = np.array(arange(0, 1, 0.01))
    total_time = []
    disk_size = []
    for i in cuts:
        report = report_upgrade(total_report=total_report, pop_cut=i,  q=q)
        time, size = loss_function2(report, v_disk, v_tape)
        total_time.append(time)
        disk_size.append(size)
    total_time = np.array(total_time)
    disk_size = np.array(disk_size)

    return cuts, total_time, disk_size

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=q, v_disk=100, v_tape=1)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.title('Total time')
plt.xlabel('Popularity cut')
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.title('Disk Size')
plt.xlabel('Popularity cut')
plt.subplot(223)
a = 0.5
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())
plt.title('Diff')
plt.xlabel('Popularity cut')

# <codecell>

data['Storage']

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.title('Total time')
plt.xlabel('Popularity cut')
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.title('Disk Size')
plt.xlabel('Popularity cut')
plt.subplot(223)
a = 0.8
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())
plt.title('Diff')
plt.xlabel('Popularity cut')

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.title('Total time')
plt.xlabel('Popularity cut')
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.title('Disk Size')
plt.xlabel('Popularity cut')
plt.subplot(223)
a = 0.3
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())
plt.title('Diff')
plt.xlabel('Popularity cut')

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.title('Total time')
plt.xlabel('Popularity cut')
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.title('Disk Size')
plt.xlabel('Popularity cut')
plt.subplot(223)
a = 0.4
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())
plt.title('Diff')
plt.xlabel('Popularity cut')

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.title('Total time')
plt.xlabel('Popularity cut')
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.title('Disk Size')
plt.xlabel('Popularity cut')
plt.subplot(223)
a = 0.6
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())
plt.title('Diff')
plt.xlabel('Popularity cut')

# <codecell>

q = [0.25, 0.5, 0.75]
cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=q, c_disk=120, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

q = [0.25, 0.5, 0.75]
cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=q, c_disk=150, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=0.34, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
total_size - used_size

# <codecell>

q = [0.25, 0.5, 0.75]
cuts, loss, min_cut, min_loss = get_loss_curve(total_report, q=q, c_disk=100, c_tape=1, c_miss=10000)

# <codecell>

print 'Min point is ', (min_cut[0], min_loss)
plt.plot(cuts, np.log(loss))
plt.xlabel('Popularity cuts')
plt.ylabel('log(Loss)')
plt.title('Loss function1')
plt.show()

# <codecell>

#total_size = (data['LFNSize'].values*data['Nb_Replicas'].values).sum()
total_size = (data['DiskSize'].values+data['TapeSize'].values).sum()
total_size_d = (data['DiskSize'].values).sum()
print total_size
print total_size_d

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
total_size - used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=1, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
total_size - used_size

# <codecell>

total_report = report_upgrade(total_report, pop_cut=min_cut, q=q)
used_size = (total_report['LFNSize'].values*total_report['NbReplicas'].values*total_report['OnDisk'].values).sum()
total_size - used_size

# <codecell>

a = data['Nb Replicas']

# <codecell>

data.columns
a = data['Nb Replicas']

# <codecell>

data.columns
#a = data['Nb Replicas']

# <codecell>

a = data['Nb_Replicas']

# <codecell>

a = data['Nb_Replicas']
a

# <codecell>

a = data['Nb_Replicas'].values
a

# <codecell>

a = data['Nb_Replicas'].values
plt.hisct(a)
plt.show

# <codecell>

a = data['Nb_Replicas'].values
plt.hisct(a)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a, bins=7)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a, bins=8)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a, bins=20)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a, bins=20, normed=true)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a, bins=20, normed=True)
plt.show()

# <codecell>

a = data['Nb_Replicas'].values
plt.hist(a, bins=10, normed=True)
plt.show()

# <codecell>

import ipykee
#ipykee.create_project(project_name="E._LossFunctionOptimization", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="E._LossFunctionOptimization")

