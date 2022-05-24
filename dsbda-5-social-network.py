#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\aarti\Downloads\Social_Network_Ads.csv')

x= dataset.iloc[:, :2].values
y = dataset.iloc[:, 4].values


# In[21]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
 x, y, test_size = 0.25, random_state = 0)


# In[22]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)
print (xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)


# In[23]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix:  \n",  cm)


# In[32]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))


# In[38]:


from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 0].min() - 1, 

                               stop = X_set[:, 0].max() + 1, step = 0.01))


# In[39]:


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 0],

                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




