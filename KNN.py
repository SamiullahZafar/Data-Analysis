#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[24]:


data = pd.read_csv("Mydata.csv")
data.head()


# In[25]:


data.isnull().sum()


# In[26]:


data.head()


# In[27]:


data.tail()


# In[28]:


data.info()


# In[29]:


data.describe()


# In[30]:


corr_matrix = data.corr()
print(corr_matrix)


# In[31]:


top_corr_features = corr_matrix.index


# In[32]:


plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[33]:


data.hist(figsize=(20,20))


# In[36]:


sns.set_style('whitegrid')
sns.countplot(x='age',data=data,palette='RdBu_r')


# In[56]:


feature_cols = ['id','age','heart_disease']
X = data[feature_cols]
y = data['smoking_status']


# In[57]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,5):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
plt.plot([k for k in range(1,5)], knn_scores, color = 'red')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[58]:


knn = KNeighborsClassifier(n_neighbors = 5)
score=cross_val_score(knn_classifier,X,y,cv=10)
print(score.mean())


# In[59]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)


# In[60]:


print(metrics.accuracy_score(y_test,y_pred))


# In[63]:


test1 = (3,53,1)
test1 = np.asarray(test1)
test1 = test1.reshape(1,-1)
test1_predict = knn.predict(test1)
if test1_predict==1:
    print("Person has high chance of Smoker")
else:
    print("Person has low chance of not Smoker")

