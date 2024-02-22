#!/usr/bin/env python
# coding: utf-8

# In[267]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[268]:


dataset = pd.read_csv('Mydata.csv')
X = dataset.iloc[:, [3,4]].values
Y = dataset.iloc[:, 11].values


# In[269]:


dataset.head()


# In[270]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[271]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)


# In[272]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)


# In[273]:


Y_Pred = classifier.predict(X_Test)


# In[274]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)


# In[275]:


from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine (Training set)')
plt.xlabel('Age')
plt.ylabel('Smoking Status')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




