# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

res = pd.read_csv('Battle results.csv')
res

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
%matplotlib inline

# <codecell>

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data / 255.0, mnist.target)

# <codecell>

res = pd.read_csv('Battle results.csv')
res

# <codecell>

res = pd.read_csv('Battle results.csv')
res

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

sig_data = pd.read_csv('howto/toy_datasets/toy_datasets/toyMC_sig_mass.csv', sep='\t')
bck_data = pd.read_csv('howto/toy_datasets/toy_datasets/toyMC_bck_mass.csv', sep='\t')

labels = np.array([1] * len(sig_data) + [0] * len(bck_data))
data = pd.concat([sig_data, bck_data])

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

sig_data = pd.read_csv('howto/toy_datasets/toyMC_sig_mass.csv', sep='\t')
bck_data = pd.read_csv('howto/toy_datasets/toyMC_bck_mass.csv', sep='\t')

labels = np.array([1] * len(sig_data) + [0] * len(bck_data))
data = pd.concat([sig_data, bck_data])

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.5)

# <codecell>

X_train

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

sig_data = pd.read_csv('howto/toy_datasets/toyMC_sig_mass.csv', sep='\t')
bck_data = pd.read_csv('howto/toy_datasets/toyMC_bck_mass.csv', sep='\t')

labels = np.array([1] * len(sig_data) + [0] * len(bck_data))
data = pd.concat([sig_data, bck_data])
variables = ["FlightDistance", "FlightDistanceError", "IP", "VertexChi2", "pt", "p0_pt", "p1_pt", "p2_pt", 'LifeTime','dira']

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(data[variables], labels, train_size=0.5)

# <codecell>

X_train

# <codecell>

from sklearn.preprocessing import StandardScaler

a = StandardScaler()

# <codecell>

from sklearn.preprocessing import StandardScaler

a = StandardScaler().fit_transform(X_train.values)
a

# <codecell>

from sklearn.preprocessing import MinMaxScaler

a = MinMaxScaler().fit_transform(X_train.values)
a

# <codecell>

from sklearn.preprocessing import MinMaxScaler

a = MinMaxScaler().fit_transform(data.values)
data[variables] = a
data

# <codecell>

from sklearn.preprocessing import MinMaxScaler, Imputer

a = MinMaxScaler().fit_transform(Imputer().fit_transform(data.values))
data[variables] = a
data

# <codecell>

Imputer().fit_transform(data.values)

# <codecell>

from sklearn.preprocessing import MinMaxScaler, Imputer

b = Imputer().fit_transform(data.values)
a = MinMaxScaler().fit_transform(b)
data[variables] = a
data

# <codecell>

a.shape

# <codecell>

data.shape

# <codecell>

a.shape

# <codecell>

data[variables]

# <codecell>

from sklearn.preprocessing import MinMaxScaler, Imputer

b = Imputer().fit_transform(data.values)
a = MinMaxScaler().fit_transform(b)
data[variables][variables] = a
data

# <codecell>

from sklearn.preprocessing import MinMaxScaler, Imputer

b = Imputer().fit_transform(data[variables].values)
a = MinMaxScaler().fit_transform(b)
data[variables] = a
data

# <codecell>

from sklearn.preprocessing import MinMaxScaler, Imputer

b = Imputer().fit_transform(data[variables].values)
a = MinMaxScaler().fit_transform(b)
data[variables] = a
data[variables]

# <codecell>

from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score

sig_data = pd.read_csv('howto/toy_datasets/toyMC_sig_mass.csv', sep='\t')
bck_data = pd.read_csv('howto/toy_datasets/toyMC_bck_mass.csv', sep='\t')

labels = np.array([1] * len(sig_data) + [0] * len(bck_data))
data = pd.concat([sig_data, bck_data])
variables = ["FlightDistance", "FlightDistanceError", "IP", "VertexChi2", "pt", "p0_pt", "p1_pt", "p2_pt", 'LifeTime','dira']

from sklearn.preprocessing import MinMaxScaler, Imputer

data1 = Imputer().fit_transform(data[variables].values)
data2 = MinMaxScaler().fit_transform(data1)
data[variables] = data2

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(data[variables], labels, train_size=0.5)

# <codecell>

X_train

# <codecell>

y_train

# <codecell>

#!pip install nolearn

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

!pip install nolearn

# <codecell>

#!pip install nolearn

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 1],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 2],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

y_train

# <codecell>

y_train.min()

# <codecell>

X_train

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(data[variables].values, labels, train_size=0.5)

# <codecell>

X_train

# <codecell>

#!pip install nolearn

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 1],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 2],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

y_pred = clf.predict(X_test)
print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

import neurolab as nl
f2 = nl.trans.SoftMax()
f = nl.trans.LogSig()
init = []
for i in range(0, X_train.shape[1]):
    init.append([0,1])
net = nl.net.newff(init,[X_train.shape[1], 300, 2], [f, f, f])
for l in net.layers:
    #l.initf = nl.init.init_rand(l, min=0, max=0.05, init_prop='w')
    #l.initf = nl.init.midpoint(l)
    l.initf = nl.init.init_zeros(l)
    net.init()   
net.trainf = nl.train.train_rprop

# <codecell>

from sklearn.preprocessing import OneHotEncoder

y = y_train.reshape((len(y_train),1))
label_train = np.array(OneHotEncoder(n_values=2).fit_transform(y).todense())

# <codecell>

%time net.train(X_train, label_train, epochs=10, show=1)

# <codecell>

predict_labels = net.sim(X_test)

# <codecell>

predict_labels

# <codecell>

y_pred = []
for l in predict_labels:
    y_pred.append(list(l).index(l.max()))
y_pred = np.array(y_pred)
y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

#!python setup.py install

# <codecell>

X_train.shape

# <codecell>

X_train.shape

# <codecell>

X_train.shape

# <codecell>

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer

net = buildNetwork(784, 300, 2, bias=True, outclass=SoftmaxLayer)

# <codecell>

!python setup.py install

# <codecell>

ls

# <codecell>

cd pybrain-master/

# <codecell>

!python setup.py install

# <codecell>

#!python setup.py install

# <codecell>

X_train.shape

# <codecell>

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer

net = buildNetwork(784, 300, 2, bias=True, outclass=SoftmaxLayer)

# <codecell>

from pybrain.datasets import SupervisedDataSet
from sklearn.preprocessing import OneHotEncoder

y = y_train.reshape((len(y_train),1))
label = np.array(OneHotEncoder(n_values=2).fit_transform(y).todense())

ds = SupervisedDataSet(X_train.shape[1], 2)
for i in range(0, len(y_train)):
    ds.addSample(tuple(list(X_train[i,:])), tuple(list(label[i])))

# <codecell>

%%time
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0)

for i in range(10):
    error = trainer.train()
    print "Epoch: %d, Error: %7.4f" % (i, error) 

# <codecell>

ds

# <codecell>

label

# <codecell>

%%time
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0)

for i in range(10):
    error = trainer.train()
    print "Epoch: %d, Error: %7.4f" % (i, error) 

# <codecell>

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer

net = buildNetwork(X_train.shape[1], 300, 2, bias=True, outclass=SoftmaxLayer)

# <codecell>

from pybrain.datasets import SupervisedDataSet
from sklearn.preprocessing import OneHotEncoder

y = y_train.reshape((len(y_train),1))
label = np.array(OneHotEncoder(n_values=2).fit_transform(y).todense())

ds = SupervisedDataSet(X_train.shape[1], 2)
for i in range(0, len(y_train)):
    ds.addSample(tuple(list(X_train[i,:])), tuple(list(label[i])))

# <codecell>

%%time
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0)

for i in range(10):
    error = trainer.train()
    print "Epoch: %d, Error: %7.4f" % (i, error) 

# <codecell>

y_pred = []
for i in X_test:
    pred = net.activate(i)
    val = list(pred).index(pred.max())
    y_pred.append(val)

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

!pip install theanets
#!pip install skdata

# <codecell>

import matplotlib.pyplot as plt
import theanets

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(784, 300, 2))

# <codecell>

import matplotlib.pyplot as plt
import theanets

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(X_train.shape[1], 300, 2))

# <codecell>

train = (X_train.astype(np.float32), y_train.astype(np.uint8))
test = (X_test.astype(np.float32), y_test.astype(np.uint8))

# <codecell>

%%time
n = 0
for train, valid in exp.itertrain(train, optimize='nag', learning_rate=0.001, momentum=0.9):
    print('training loss:', train['loss'])
    print('most recent validation loss:', valid['loss'])
    n = n+1
    if n==10:
        break

# <codecell>

predict_labels = exp.network.predict(X_test.astype(np.float32))

# <codecell>

y_pred = []
for l in predict_labels:
    y_pred.append(list(l).index(l.max()))
y_pred = np.array(y_pred)
y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier(), features=variables)
# Training classifier
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier())
# Training classifier
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

prob = sk.predict_proba(test_data)
print prob
y_pred = prob[:,1]

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier())
# Training classifier

# <codecell>

%%time
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier(n_estimators=10))
# Training classifier

# <codecell>

%%time
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

prob = sk.predict_proba(X_test)
print prob
y_pred = prob[:,1]

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

prob = sk.predict(X_test)
print prob

# <codecell>

y_pred = sk.predict(X_test)
print y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

res = pd.read_csv('Battle results.csv')
res

# <codecell>

ls

# <codecell>

cd ../

# <codecell>

res = pd.read_csv('Battle results.csv')
res

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier(n_estimators=100))
# Training classifier

# <codecell>

%%time
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

y_pred = sk.predict(X_test)
print y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier(n_estimators=500))
# Training classifier

# <codecell>

%%time
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

y_pred = sk.predict(X_test)
print y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier(n_estimators=10))
# Training classifier

# <codecell>

%%time
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

y_pred = sk.predict(X_test)
print y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 2],
    learn_rates=0.1,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

y_pred = clf.predict(X_test)
print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 2],
    learn_rates=0.01,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

y_pred = clf.predict(X_test)
print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 2],
    learn_rates=0.1,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=0,
    )

# <codecell>

%time clf.fit(X_train, y_train)

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

y_pred = clf.predict(X_test)
print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

%%time
n = 0
for train, valid in exp.itertrain(train, optimize='nag', learning_rate=0.1, momentum=0.9):
    print('training loss:', train['loss'])
    print('most recent validation loss:', valid['loss'])
    n = n+1
    if n==10:
        break

# <codecell>

#!pip install theanets
#!pip install skdata

# <codecell>

import matplotlib.pyplot as plt
import theanets

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(X_train.shape[1], 300, 2))

# <codecell>

train = (X_train.astype(np.float32), y_train.astype(np.uint8))
test = (X_test.astype(np.float32), y_test.astype(np.uint8))

# <codecell>

%%time
n = 0
for train, valid in exp.itertrain(train, optimize='nag', learning_rate=0.1, momentum=0.9):
    print('training loss:', train['loss'])
    print('most recent validation loss:', valid['loss'])
    n = n+1
    if n==10:
        break

# <codecell>

predict_labels = exp.network.predict(X_test.astype(np.float32))

# <codecell>

y_pred = []
for l in predict_labels:
    y_pred.append(list(l).index(l.max()))
y_pred = np.array(y_pred)
y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

from rep.classifiers import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Using gradient boosting with default settings
sk = SklearnClassifier(GradientBoostingClassifier(n_estimators=100))
# Training classifier

# <codecell>

%%time
sk.fit(X_train, y_train)
print('training complete')

# <codecell>

y_pred = sk.predict(X_test)
print y_pred

# <codecell>

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss

print "Accuracy:", zero_one_loss(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)

# <codecell>

res = pd.read_csv('Battle results.csv')
res

# <codecell>

import ipykee
ipykee.create_project(project_name="Battle_of_ANNs", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="Battle_of_ANNs")

# <codecell>

session.commit("Toy Dataset.")

# <codecell>

import ipykee
#ipykee.create_project(project_name="Battle_of_ANNs", repository="git@github.com:hushchyn-mikhail/CERN_Time_Series.git")
session = ipykee.Session(project_name="Battle_of_ANNs")

