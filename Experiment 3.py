#!/usr/bin/env python
# coding: utf-8

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[26]:


irisData = load_iris()


# In[2]:


X = irisData.data
y = irisData.target
print(X)
print(y)


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size = 0.2, random_state=42)


# In[39]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[40]:


knn.fit(X_train, y_train)


# In[41]:


knn.predict(X_test)
knn.score(X_test, y_test)


# In[ ]:




