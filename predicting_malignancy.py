#!/usr/bin/env python
# coding: utf-8

# In[38]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
mpl.rcParams['figure.dpi'] = 120  # make plots bigger by default


# In[2]:


# This section will read the data into a table with labeled columns for each feature.
data = pd.read_csv('wdbc.data',names=["ID", "type", "mean radius", "radius se", "radius worst value", "mean texture", "texture se", "texture worst value", "mean perimeter", "perimeter se", "perimeter worst value", "mean area", "area se", "area worst value", "mean smoothness", "smoothness se", "smoothness worst value", "mean compactness", "compactness se", "compactness worst value", "mean concavity", "concavity se", "concavity worst value", "mean concave points", "concave points se", "concave points worst value", "mean symmetry", "symmetry se", "symmetry worst value", "mean fractal dimension", "fractal dimension se", "fractal dimension worst value"])
data.head() 


# In[3]:


# This will create a dataframe for only the mean values of the 10 features. It does so by dropping the SE and worst value columns. 
data_mean = data.drop(columns=["radius se","texture se","perimeter se","area se","smoothness se","compactness se","concavity se","concave points se","symmetry se","fractal dimension se","radius worst value","texture worst value","perimeter worst value","area worst value","smoothness worst value","compactness worst value","concavity worst value","concave points worst value","symmetry worst value","fractal dimension worst value"])
data_mean.head()


# In[4]:


# This shows how balanced the dataset is in terms of y values. This dataset is considered to be balanced.  
y = data_mean.type
mal_freq = 0
ben_freq = 0
for w in y:
    if w == 'M':
        mal_freq += 1
    if w == 'B':
        ben_freq += 1
print("Malignant: " + str(mal_freq))
print("Benign: " + str(ben_freq))


# In[5]:


# This plots the mean radii of the tumors and color codes if they are benign or malignant. 
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean radius'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean radius'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Radii") # title
plt.xlabel("mean radius") # labels
plt.ylabel("frequency")


# In[6]:


# This plots the mean texture of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean texture'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean texture'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Texture") # title
plt.xlabel("mean texture") # labels
plt.ylabel("frequency")


# In[7]:


# This plots the mean perimeter of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean perimeter'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean perimeter'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Perimeter") # title
plt.xlabel("mean perimeter") # labels
plt.ylabel("frequency")


# In[8]:


# This plots the mean area of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean area'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean area'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Area") # title
plt.xlabel("mean area") # labels
plt.ylabel("frequency")


# In[9]:


# This plots the mean smoothness of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean smoothness'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean smoothness'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Smoothness") # title
plt.xlabel("mean smoothness") # labels
plt.ylabel("frequency")


# In[10]:


# This plots the mean compactness of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean compactness'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean compactness'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Compactness") # title
plt.xlabel("mean compactness") # labels
plt.ylabel("frequency")


# In[11]:


# This plots the mean concavity of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean concavity'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean concavity'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Concavity") # title
plt.xlabel("mean concavity") # labels
plt.ylabel("frequency")


# In[12]:


# This plots the mean concave points of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean concave points'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean concave points'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Concave Points") # title
plt.xlabel("mean concave points") # labels
plt.ylabel("frequency")


# In[13]:


# This plots the mean symmetry of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean symmetry'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean symmetry'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Symmetry") # title
plt.xlabel("mean symmetry") # labels
plt.ylabel("frequency")


# In[14]:


# This plots the mean fractal dimension of the tumors and color codes if they are benign or malignant.
datas = data[data.type == 'M'] # this takes the ones that are type M
sns.distplot(datas['mean fractal dimension'],  kde=False, label='Malignant') # this plots the mean radii of the malignant ones
datas = data[data.type == 'B'] # this takes the ones that are type B
sns.distplot(datas['mean fractal dimension'],  kde=False, label='Benign') # this plots the mean radii of the benign ones
plt.legend() # this adds a legend
plt.title("Mean Fractal Dimension") # title
plt.xlabel("mean fractal dimension") # labels
plt.ylabel("frequency")


# In[15]:


data_mean.corr()
# The most correlated features are texture:radius and perimeter:fractal dimension. It is not good to have correlated features because they can be redundant and slow down the program. They also can increase bias. Based on this knowledge, several features which are highly correlated may be removed. These features could include .


# In[16]:


# This makes a new datafram which removes the column that tells whether it is malignant or benign. 
y = data.type
data_p=data.drop(columns="type") #create a new data array
data_p.head()


# In[17]:


# First Machine Learning Model
# This uses the linear model method with logistic regression to perform machine learning. This model is fitted to the mean features and tumor types. 
from sklearn import linear_model
from sklearn import model_selection
xreg = data_mean[['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension']] # this makes an x variable with all the mean features 
yreg = data.type # this makes the y variable the type of tumor column
clf = linear_model.LogisticRegression() # this sets up logistic regression
clf.fit(xreg, yreg) # this fits it to the new x variable and y variable
print(clf.coef_,clf.intercept_) # prints the coefficients and the intercept


# In[18]:


clf.score(xreg,yreg) # this returns a score of how well the model is working


# In[19]:


# Perform a cross validation and find the mean and standard deviation of this machine learning method. 
a = model_selection.cross_val_score(clf,xreg,yreg,cv=20) # this performs crossvalidation
print(a.mean()) # mean
print(a.std()) # standard deviation


# In[20]:


# Second Machine Learning Model
# This method uses the random forest classifier to perform machine learning. 


# In[21]:


clf=RandomForestClassifier()
parameters = {
    'n_estimators': [6, 8, 10, 12],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 5, 10, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1,5,10]
}


# In[22]:


# This fits the data to the parameters. 
x = data_p 
cv = GridSearchCV(clf, parameters, cv=3)
cv.fit(x, y) 


# In[36]:


# This prints the best parameters and the best score. 
print(cv.best_params_)
print(cv.best_score_) 
clf.fit(x,y)


# In[28]:


# This graphically represents the relative importance of the features. 
x1=data_p.columns
y1=clf.feature_importances_
sns.barplot(x1,y1)
plt.title("Feature Importances") #labeling graph
plt.xlabel("Columns")
plt.xticks(rotation='vertical')
plt.ylabel("Feature Importance")


# In[29]:


# Create the test train splits by doing odd and even training and testing variables. 
x_train=x[::2] #pulling out even entries
x_test=x[1::2] #pulling out odd entries
y_train=y[::2] #pulling out even entries
y_test=y[1::2] #pulling out odd entries


# In[30]:


# This will fit and train the model on the training data. 
cv = GridSearchCV(clf, parameters, cv=3)
cv.fit(x_train, y_train)


# In[31]:


# This will predict y values (whether it is benign or malignant) from x test values. 
predicted_y=cv.predict(x_test)
malp_freq = 0
benp_freq = 0
for w in predicted_y:
    if w == 'M':
        malp_freq += 1
    if w == 'B':
        benp_freq += 1
print("Predicted Malignant: " + str(malp_freq))
print("Predicted Benign: " + str(benp_freq))


# In[32]:


# This prints the predicted tumor types. 
print(predicted_y)


# In[33]:


# Print the accuracy score of your predicted y values
accuracy_score(y_test,predicted_y)


# In[34]:


# Here is a cross validation for the second method of machine learning. 
b = model_selection.cross_val_score(clf,x_test,predicted_y,cv=20) # this performs crossvalidation
print(b.mean()) # mean
print(b.std()) # standard deviation


# In[37]:


# Cross validations were performed for both machine learning methods. The first method had a mean of 92.8% with a standard deviation of 5.38. The second method had a mean of 95.5% and a standard deviation of 4.54. The second method has a higher mean and lower standard deviation which means it has a lower generalization error. It can be applied to out-of-sample data successfully. 
# These methods are fairly successful at learning the training data and applying it to unknown data. 


# In[ ]:




