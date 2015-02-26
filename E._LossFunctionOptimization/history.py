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

def loss_function2():
    pass

# <codecell>

import ipykee
#ipykee.create_project(project_name="E._LossFunctionOptimization", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="E._LossFunctionOptimization")

# <codecell>

def loss_function2(total_report, v_disk=1, v_tape=100):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    inten = total_report['Intensity']
    
    t_disk = (inten/nb_replical*lfn_size/v_disk*marker).sum()
    t_tape = (inten/nb_replical*lfn_size/v_tape*(1-marker)).sum()
    tota_time = t_disk + t_tape
    
    disk_size = (nb_replical*lfn_size*marker).sum()
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

    return total_time, disk_size

# <codecell>

def loss_function2(total_report, v_disk=1, v_tape=100):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    inten = total_report['Intensity']
    
    t_disk = (inten/nb_replical*lfn_size/v_disk*marker).sum()
    t_tape = (inten/nb_replical*lfn_size/v_tape*(1-marker)).sum()
    tota_time = t_disk + t_tape
    
    disk_size = (nb_replical*lfn_size*marker).sum()
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

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=100)

# <codecell>

def loss_function2(total_report, v_disk=1, v_tape=100):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    inten = total_report['Intensity']
    
    t_disk = (inten/nb_replicas*lfn_size/v_disk*marker).sum()
    t_tape = (inten/nb_replicas*lfn_size/v_tape*(1-marker)).sum()
    tota_time = t_disk + t_tape
    
    disk_size = (nb_replical*lfn_size*marker).sum()
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

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=100)

# <codecell>

def loss_function2(total_report, v_disk=1, v_tape=100):
    lfn_size = total_report['LFNSize'].values
    nb_replicas = total_report['NbReplicas']
    marker = total_report['OnDisk'].values
    miss = total_report['Missing'].values
    inten = total_report['Intensity']
    
    t_disk = (inten/nb_replicas*lfn_size/v_disk*marker).sum()
    t_tape = (inten/nb_replicas*lfn_size/v_tape*(1-marker)).sum()
    tota_time = t_disk + t_tape
    
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

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=100)

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

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=100)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=0.1, v_tape=100)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=0.001, v_tape=100)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=10000)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=1)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=10)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=1, v_tape=2)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

cuts, total_time, disk_size = get_loss_curve2(total_report, q=[0.5, 0.75, 0.95], v_disk=100, v_tape=1)

# <codecell>

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(cuts, total_time)
plt.subplot(122)
plt.plot(cuts, disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time - disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time - 10*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time - 100*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time - 1000*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time - 10000*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 10*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 100*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 1000*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 200*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 110*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 120*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 130*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 140*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 150*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 160*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 190*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 200*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 300*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time + 500*disk_size)

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
plt.plot(cuts, total_time/total_time.max() + disk_size/disk_size.max())

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
a = 0.1
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
a = 0.5
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
a = 0.7
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
a = 0.6
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())

# <codecell>

plt.figure(figsize=(15,10))
plt.subplot(221)
plt.plot(cuts, total_time)
plt.subplot(222)
plt.plot(cuts, disk_size)
plt.subplot(223)
a = 0.5
plt.plot(cuts, (1-a)*total_time/total_time.max() + a*disk_size/disk_size.max())

# <codecell>

import ipykee
#ipykee.create_project(project_name="E._LossFunctionOptimization", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="E._LossFunctionOptimization")

# <codecell>

session.commit("Loss function 1 added.")

