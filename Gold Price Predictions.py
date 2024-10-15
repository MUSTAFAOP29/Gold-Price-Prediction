#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[2]:


gold_data=pd.read_csv('gld_price_data.csv')


# In[3]:


gold_data.head(5)


# In[4]:


gold_data.tail(5)


# In[5]:


gold_data.shape


# In[6]:


gold_data.info


# In[7]:


gold_data.isnull().sum()


# In[8]:


correlation=gold_data.corr()


# In[11]:


gold_data.describe()


# In[13]:


plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


# In[14]:


#corre;ation value


# In[15]:


print(correlation['GLD'])


# In[16]:


# checking distribution of gold price


# In[19]:


sns.distplot(gold_data['GLD'],color='black')


# In[20]:


# splittin features and taget


# In[21]:


X=gold_data.drop(['Date','GLD'],axis=1)
Y=gold_data['GLD']


# In[22]:


print(X)


# In[23]:


print(Y)


# In[24]:


#splitting into training and test data


# In[25]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=2)


# In[26]:


#Model Training 
#Using Random Forest Model


# In[28]:


regressor=RandomForestRegressor(n_estimators=100)


# In[29]:


regressor.fit(X_train,Y_train)


# In[30]:


test_data_prediction=regressor.predict(X_test)


# In[31]:


print(test_data_prediction)


# In[32]:


# R square error


# In[35]:


error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R Squared Error: ",error_score)


# In[36]:


#compare actual and predicted values in graph


# In[37]:


Y_test=list(Y_test)


# In[41]:


plt.plot(Y_test,color='Blue',label='Actual Value')
#plt.plot(test_data_prediction,color='Green',label='Predicted Value')
plt.title("Actual Price VS Predicted Price")
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[40]:


plt.plot(Y_test,color='Blue',label='Actual Value')
plt.plot(test_data_prediction,color='Green',label='Predicted Value')
plt.title("Actual Price VS Predicted Price")
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:




