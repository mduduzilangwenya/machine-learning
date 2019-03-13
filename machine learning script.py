# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:54:34 2018

@author: mdudu
"""
# Import lots of tools
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# load default data set
credit = pd.read_csv("defaultBal.csv")
# set up data for scikit-learn
# note:  You don't really have to use X,y
# this is our target
y = credit.default
# These are predictors (skip id)
Xall= credit.values[:,1:24]
# copy of all data to predictor
X = Xall.copy()
# Set up as just real values, use this for last part of problem
# Restricts predictor to only real values (not discrete)
# X = Xall[:,12:23].copy()
print(X.shape)
print(np.mean(y))
yGuess = np.mean(y)

# display dataframe in nice table just to see
credit.head()

# Note:Use test_size=0.25 throughout
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

##  Naive Bayes example

# Try a naive Bayes classifier
gnb = GaussianNB()
trainFit = gnb.fit(X_train,y_train)
print(trainFit.score(X_train,y_train))
print(trainFit.score(X_test, y_test))

# start monte-carlo for GaussianNB()
nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
gnb = GaussianNB()
for i in range(nmc):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    trainFit = gnb.fit(X_train,y_train)
    trainScore[i] = trainFit.score(X_train,y_train)
    testScore[i] =  trainFit.score(X_test,y_test)
    
print(np.mean(trainScore))
print(np.std(trainScore))
print(np.mean(testScore))
print(np.std(testScore))
print(np.mean(testScore>yGuess))

## Repeat the monte-carlo with a LinearDiscriminant classifier.
nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
lda = LinearDiscriminantAnalysis()
for i in range(nmc):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    trainFit = lda.fit(X_train,y_train)
    trainScore[i] = trainFit.score(X_train,y_train)
    testScore[i] =  trainFit.score(X_test,y_test)
print(np.mean(trainScore))
print(np.std(trainScore))
print(np.mean(testScore))
print(np.std(testScore))
print(np.mean(testScore>yGuess))

#Repeat this with Logistic regression for C=100, C=1, C=0.01. 
nmc = 50
C = [100, 1, 0.01]
# empty dictionaries
log_train_means = {}
log_train_std = {}

log_test_means = {}
log_test_std = {}

## Repeat the monte-carlo with a Logistic Regression
for i in C:
    logistictrain = []
    logistictest  = []
    logisticMod = LogisticRegression(C= i)
    for j in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        trainFit = logisticMod.fit(X_train,y_train)
        logistictrain.append(trainFit.score(X_train,y_train))
        logistictest.append(trainFit.score(X_test,y_test))
    # train
    log_train_means[i] = round(np.mean(logistictrain), 4)  
    log_train_std[i] = round(np.std(logistictrain), 4)  
    # test 
    log_test_means[i] = round(np.mean(logistictest), 4)
    log_test_std[i] = round(np.std(logistictest), 4)

print("\n Training Means")
for i in log_train_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Training Standard Deviation")
for i in log_train_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))

print("\n Testing Means")
for i in log_test_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Testing Standard Deviation")
for i in log_test_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))


#Repeat this with Linear SVC for C=100, C=1, C=0.01. 
nmc = 50
C = [100, 1, 0.01]
# empty dictionaries
lin_train_means = {}
lin_train_std = {}

lin_test_means = {}
lin_test_std = {}

## Repeat the monte-carlo with a Linear SVC
for i in C:
    LinearSVCtrain = []
    LinearSVCtest  = []
    LinearSVCMod = LinearSVC(C= i)
    for j in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        trainFit = LinearSVCMod.fit(X_train,y_train)
        LinearSVCtrain.append(trainFit.score(X_train,y_train))
        LinearSVCtest.append(trainFit.score(X_test,y_test))
    # train
    lin_train_means[i] = round(np.mean(LinearSVCtrain), 4)  
    lin_train_std[i] = round(np.std(LinearSVCtrain), 4)  
    # test 
    lin_test_means[i] = round(np.mean(LinearSVCtest), 4)
    lin_test_std[i] = round(np.std(LinearSVCtest), 4)

print("\n Training Means")
for i in lin_train_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Training Standard Deviation")
for i in lin_train_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))

print("\n Testing Means")
for i in lin_test_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Testing Standard Deviation")
for i in lin_test_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))

#Repeat this with KNeighborClassifier for n_neighbors = 3, 11, 25. 
nmc = 50
n_neighbors = [3, 11, 25]
# empty dictionaries
knn_train_means = {}
knn_train_std = {}

knn_test_means = {}
knn_test_std = {}

## Repeat the monte-carlo with a KNeighborClassifier 
for i in n_neighbors:
    knntrain = []
    knntest  = []
    clf = KNeighborsClassifier(n_neighbors = i)
    for j in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        trainFit = clf.fit(X_train,y_train)
        knntrain.append(trainFit.score(X_train,y_train))
        knntest.append(trainFit.score(X_test,y_test))
    # train
    knn_train_means[i] = round(np.mean(knntrain), 4)  
    knn_train_std[i] = round(np.std(knntrain), 4)  
    # test 
    knn_test_means[i] = round(np.mean(knntest), 4)
    knn_test_std[i] = round(np.std(knntest), 4)

print("\n Training Means")
for i in knn_train_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Training Standard Deviation")
for i in knn_train_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))

print("\n Testing Means")
for i in knn_test_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Testing Standard Deviation")
for i in knn_test_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    

#Repeat this with Decision Tree for max_depth = 5, 10, 25. 
nmc = 50
max_depth = [5, 10, 25]
# empty dictionaries
tree_train_means = {}
tree_train_std = {}

tree_test_means = {}
tree_test_std = {}

## Repeat the monte-carlo with a Decision Tree 
for i in max_depth:
    treetrain = []
    treetest  = []
    tree = DecisionTreeClassifier(max_depth = i, random_state=0)
    for j in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        trainFit = tree.fit(X_train,y_train)
        treetrain.append(trainFit.score(X_train,y_train))
        treetest.append(trainFit.score(X_test,y_test))
    # train
    tree_train_means[i] = round(np.mean(treetrain), 4)  
    tree_train_std[i] = round(np.std(treetrain), 4)  
    # test 
    tree_test_means[i] = round(np.mean(treetest), 4)
    tree_test_std[i] = round(np.std(treetest), 4)

print("\n Training Means")
for i in tree_train_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Training Standard Deviation")
for i in tree_train_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))

print("\n Testing Means")
for i in tree_test_means.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))
    
print("\n Testing Standard Deviation")
for i in tree_test_std.items():
    print("C = " + str(i[0]) + " : " + str(i[1]))

#Repeat the linear LinearDiscriminant, but use just the real valued data. See example file for line to move just these fields into X.
# Set up as just real values, use this for last part of problem
# Restricts predictor to only real values (not discrete)
X = Xall[:,12:23].copy()
print(X.shape)

## Repeat the monte-carlo with a LinearDiscriminant classifier.
nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
lda = LinearDiscriminantAnalysis()
for i in range(nmc):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    trainFit = lda.fit(X_train,y_train)
    trainScore[i] = trainFit.score(X_train,y_train)
    testScore[i] =  trainFit.score(X_test,y_test)
print(np.mean(trainScore))
print(np.std(trainScore))
print(np.mean(testScore))
print(np.std(testScore))
print(np.mean(testScore>yGuess))





    
    
    














