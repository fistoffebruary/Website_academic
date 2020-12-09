#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Enter your name(s) here
# Ryan Bailey
# Eleanor Young


# # Assignment 3 : SVMs, Neural Nets, Ensembles
# 
# In this assignment you'll implement SVMs, Neural Nets, and Ensembling methods to classify patients as either having or not having diabetic retinopathy. For this task we'll be using the same Diabetic Retinopathy data set which was used in the previous assignments. You can find additional details about the dataset [here](http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set). You'll explore how to train SVMs, NNs, and Ensembles using the `scikit-learn` library. The scikit-learn documentation can be found [here](http://scikit-learn.org/stable/documentation.html).

# In[2]:


#You may add additional imports
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import cross_val_score


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Read the data from csv file
col_names = []
for i in range(20):
    if i == 0:
        col_names.append('quality')
    if i == 1:
        col_names.append('prescreen')
    if i >= 2 and i <= 7:
        col_names.append('ma' + str(i))
    if i >= 8 and i <= 15:
        col_names.append('exudate' + str(i))
    if i == 16:
        col_names.append('euDist')
    if i == 17:
        col_names.append('diameter')
    if i == 18:
        col_names.append('amfm_class')
    if i == 19:
        col_names.append('label')

data = pd.read_csv("messidor_features.txt", names = col_names)
print(data.shape)
data.head(10)


# ### 1. Data prep

# Q1. Separate the feature columns from the class label column. You should end up with two separate data frames - one that contains all of the feature values and one that contains the class labels. Print the shape of the features DataFrame, the shape of the labels DataFrame, and the head of the features DataFrame.

# In[5]:


# your code goes here
datay = data['label']
datax = data.drop('label',axis =1 )
print(datay.shape)
print(datax.shape)
datax.head()


# ### 2. Support Vector Machines (SVM) and Pipelines

# Q2. For some classification algorithms, like KNN, SVMs, and Neural Nets, scaling of the data is critical for the algorithm to operate correctly. For other classification algorithms, like Naive Bayes, and Decision Trees, data scaling is not necessary (take a minute to think about why that is the case). 
# 
# We discussed in class how the data scaling should happen on the _training set only_, which means that it should happen _inside_ of the cross validation loop. In other words, in each fold of the cross validation, the data will be separated in to training and test sets. The scaling (calculating mean and std, for instance) should happen based on the values in the _traning set only_. Then the test set can be scaled using the values found on the training set. (Refer to the concept of [data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/).)
# 
# In order to do this with scikit-learn, you must create what's called a `Pipeline` and pass that in to the cross validation. This is a very important concept for Data Mining and Machine Learning, so let's practice it here.
# 
# Do the following:
# * Create a `sklearn.preprocessing.StandardScaler` object to standardize the dataset’s features (mean = 0 and variance = 1). Do not call `fit` on it yet. Just create the `StandardScaler` object.
# * Create a sklearn.svm.SVC classifier (do not set any arguments - use the defaults). Do not call fit on it yet. Just create the SVC object.
# * Create a `sklearn.pipeline.Pipeline` and set the `steps` to the scaler and the SVC objects that you just created. 
# * Pass the `pipeline` in to a `cross_val_score` as the estimator, along with the features and the labels, and use a 5-fold-CV. 
# 
# In each fold of the cross validation, the training phase will use _only_ the training data for scaling and training the model. Then the testing phase will scale the test data into the scaled space (found on the training data) and run the test data through the trained classifier, to return an accuracy measurement for each fold. Print the average accuracy across all 5 folds. 

# In[6]:


# your code goes here
from sklearn.preprocessing import StandardScaler as SS
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
SS = SS()
clf = SVC()
pipe = Pipeline(steps=[('scaler', SS), ('SVC', clf)])

nested_score = cross_val_score(pipe, datax, datay, cv=5)
print('Mean Nested Score:',nested_score.mean())


# Q3. The `svm.SVC` defaults to using an rbf (radial basis function) kernel. This kernel may or may not be the best choice for our dataset. We can use nested cross validation to find the best kernel for this dataset.
# 
# Set up the inner CV loop:
# * Starter code is provided to create the "parameter grid" to search. You will need to change this code! Where I have "svm__kernel", this indicates that I want to tune the "kernel" parameter in the "svm" part of the pipeline. When you created your pipeline above, you named the SVM part of the pipeline with a string. You should replace "svm" in the param_grid below with whatever you named your SVM part of the pipeline: **<replace_this>__kernel.** 
# * Create a `sklearn.model_selection.GridSearchCV` that takes in the pipeline you created above (as the estimator), the parameter grid, and uses a 5-fold-CV. Call `fit` on the `GridSearchCV` to find the best kernel. 
# * Print out the best kernel (`best_params_`) for this dataset. 

# In[7]:


# for the 'svm' part of the pipeline, tune the 'kernel' hyperparameter
param_grid = {'SVC__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
grid_search = GSCV(pipe, param_grid, cv=5, scoring='accuracy')
grid_search.fit(datax,datay)
best_kernel = grid_search.best_params_.get('SVC__kernel')
print(grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)
# your code goes here


# Q4. Now put what you did in Q3 in to an outer CV loop to evaluate the accuracy of using that best-found kernel on unseen test data. 
# * Pass the `GridSearchCV` in to a `cross_val_score` with 5-fold-CV. Print out the accuracy.
# 
# Note that the accuracy increases from Q2 because of a better choice of kernel function.

# In[8]:


# your code goes here


nested_score = cross_val_score(grid_search, datax, datay, cv=5)
print('Mean Nested Score: ',nested_score.mean())


# Q5. Let's see if we can get the accuracy even higher by tuning additional hyperparameters. SVMs have a parameter called 'C' that is the cost for a misclassification. (More info [here](https://medium.com/@pushkarmandot/what-is-the-significance-of-c-value-in-support-vector-machine-28224e852c5a)).
# * Create a parameter grid that includes the kernel (as you have above) and the C value as well. Try values of C from 50 to 100 by increments of 10. (You can use the range function to help you with this.)
# * Create a `GridSearchCV` with the pipeline from above, this new parameter grid, and a 5-fold-CV.
# * Pass the `GridSearchCV` into a `cross_val_score` with a 5-fold-CV and print out the accuracy.
# 
# Be patient as this can take some time to run. Note that the accurcay has increased even further because the best value of C was found and used on the test data.
# 
# Now we're actually starting to get closer to some decent accuracies on this dataset!

# In[9]:


# your code goes here
param_grid = {'SVC__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],'SVC__C':list(range(50,110,10))}
grid_search = GSCV(pipe, param_grid, cv=5, scoring='accuracy')
grid_search.fit(datax,datay)
best_kernel = grid_search.best_params_.get('SVC__kernel')
print(grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)

nested_score = cross_val_score(grid_search, datax, datay, cv=5)
print('Mean Nested Score:',nested_score.mean())


# ### 3. Neural Networks (NN)

# Q6. Train a multi-layer perceptron with a single hidden layer using `sklearn.neural_network.MLPClassifier`. 
# * Scaling is critical to neural networks. Create a pipeline that includes scaling and an `MLPClassifier`.
# * The `hidden_layer_sizes` parameter works as follows: if you want 3 hidden layers with 10 nodes in each hidden layer, you'd set `hidden_layer_sizes = (10, 10, 10)`. If you want one hidden layer with 100 nodes in that layer, you'd set `hidden_layer_sizes = (100,)`.
# * Let's use one hidden layer and find the best number of nodes for it. 
# * Use `GridSearchCV` with 5 folds to find the best hidden layer size and the best activation function. 
# * Try values of `hidden_layer_sizes` ranging from `(10,)` to `(60,)` with gaps of 10.
# * Try activation functions `logistic`, `tanh`, `relu`.
# 
# Wrap your `GridSearchCV` in a 5-fold `cross_val_score` and report the accuracy of your neural net.
# 
# Be patient, as this can take a few minutes to run. 

# In[10]:


# your code goes here
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neural_network import MLPClassifier as MLP
SS = SS()
clf = MLP()
#print(clf.get_params().keys())
pipe = Pipeline(steps=[('scaler', SS), ('MLP', clf)])
params = {'MLP__hidden_layer_sizes':list(range(10,70,10)),'MLP__activation':['logistic', 'tanh', 'relu']}


grid_search = GSCV(pipe, params, cv=5, scoring='accuracy')
grid_search.fit(datax,datay)
best_act = grid_search.best_params_.get('MLP__activation')
best_hl = grid_search.best_params_.get('MLP__hidden_layer_size')
print('Best Parameters:',grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)

nested_score = cross_val_score(grid_search, datax, datay, cv=5)
print('Mean Nested Score:',nested_score.mean())



# ### 4. Ensemble Classifiers
# 
# Ensemble classifiers combine the predictions of multiple base estimators to improve the accuracy of the predictions. One of the key assumptions that ensemble classifiers make is that the base estimators are built independently (so they are diverse).

# **A. Random Forests**
# 
# Q7. Use `sklearn.ensemble.RandomForestClassifier` to classify the data. Use a `GridSearchCV` to tune the hyperparameters to get the best results. 
# * Try `max_depth` ranging from 35-55 (you can use the range function to help you with this)
# * Try `min_samples_leaf` of 8, 10, 12
# * Try `max_features` of `"sqrt"` and `"log2"`
# 
# Wrap your GridSearchCV in a cross_val_score with 5-fold CV to report the accuracy of the model.
# 
# Be patient, this can take a few minutes to run.

# In[11]:


# your code goes here
from sklearn.ensemble import RandomForestClassifier as RFC
clf = RFC()
params = {'max_depth':list(range(35,56)), 'min_samples_leaf':[8,10,12], 'max_features':['sqrt','log2']}
grid_search = GSCV(clf, params, cv=5, scoring='accuracy')
grid_search.fit(datax,datay)

best_depth = grid_search.best_params_.get('max_depth')
best_msl = grid_search.best_params_.get('min_samples_leaf')
best_features = grid_search.best_params_.get('max_features')

print('Best Parameters:',grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)


# **B. AdaBoost**
# 
# Random Forests are a kind of ensemble classifier where many estimators are built independently in parallel. In contrast, there is another method of creating an ensemble classifier called *boosting*. Here the classifiers are trained one-by-one in sequence and each time the sampling of the training set depends on the performance of previously generated models.
# 
# Q8. Evaluate a `sklearn.ensemble.AdaBoostClassifier` classifier on the data. By default, `AdaBoostClassifier` uses decision trees as the base classifiers (but this can be changed). 
# * Use a GridSearchCV to find the best number of trees in the ensemble (`n_estimators`). Try values from 50-250 with increments of 25. (you can use the range function to help you with this.)
# * Wrap your GridSearchCV in a cross_val_score with 5-fold CV to report the accuracy of the model.
# 
# Be patient, this can take a few minutes to run.

# In[12]:


# your code goes here
from sklearn.ensemble import AdaBoostClassifier as ABC
clf = ABC()
params = {'n_estimators':list(range(50,275,25))}
grid_search = GSCV(clf, params, cv=5, scoring='accuracy')
grid_search.fit(datax,datay)
best_estim = grid_search.best_params_.get('n_estimators')
print('Best Parameters:',grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)


nested_score = cross_val_score(grid_search, datax, datay, cv=5)
print('Mean Nested Score:',nested_score.mean())


# ### 5. Deploying a final model

# Over the course of three programming assignments, you have tested all kinds of classifiers on this data. Some have performed better than others. 
# 
# We could continue trying to improve the accuracy of our models by tweaking their parameters more and/or we could do some feature engineering on our dataset.

# Q9. Let's say we got to the point where we had a model with high accuracy and we want to deploy that model and use it for real-world predictions.
# 
# * Let's say we're going to deploy our neural net classifier.
# * We need to make one final version of this model, where we use ALL of our available data for training (we do not hold out a test set this time, so no outer cross-validation loop). 
# * We need to tune the parameters of the model on the FULL dataset, so copy the code you entered for Q6, but remove the outer cross validation loop (remove `cross_val_score`). Just run the `GridSearchCV` by calling `fit` on it and passing in the full dataset. This results in the final trained model with the best parameters for the full dataset. You can print out `best_params_` to see what they are.
# * The accuracy of this model is what you assessed and reported in Q6.
# 
# 
# * Use the `pickle` package to save your model. We have provided the lines of code for you, just make sure your final model gets passed in to `pickle.dump()`. This will save your model to a file called finalized_model.sav in your current working directory. 

# In[13]:


import pickle
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.preprocessing import StandardScaler as SS

SS = SS()
clf = MLP()
#print(clf.get_params().keys())
pipe = Pipeline(steps=[('scaler', SS), ('MLP', clf)])
params = {'MLP__hidden_layer_sizes':list(range(10,70,10)),'MLP__activation':['logistic', 'tanh', 'relu']}


grid_search = GSCV(pipe, params, cv=5, scoring='accuracy')
grid_search.fit(datax,datay)
best_act = grid_search.best_params_.get('MLP__activation')
best_hl = grid_search.best_params_.get('MLP__hidden_layer_size')
print('Best Parameters:',grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)


# your code goes here

#set this final_model to your final model
final_model = grid_search

filename = 'finalized_model.sav'
pickle.dump(final_model, open(filename, 'wb'))


# Q10. Now if someone wants to use your trained, saved classifier to classify a new record, they can load the saved model and just call predict on it. 
# * Given this new record, classify it with your saved model and print out either "Negative for disease" or "Positive for disease."

# In[14]:


# some time later...

# use this as the new record to classify
record = [ 0.05905386, 0.2982129, 0.68613149, 0.75078865, 0.87119216, 0.88615694,
  0.93600623, 0.98369184, -0.47426472, -0.57642756, -0.53115361, -0.42789774,
 -0.21907738, -0.20090532, -0.21496782, -0.2080998, 0.06692373, -2.81681183,
 -0.7117194 ]

 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict([record])
if pred[0] == 0:
    print('Negative for disease')
elif pred[0] == 1:
    print('Positive for disease')
# your code goes here


# In[ ]:




