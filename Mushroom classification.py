#!/usr/bin/env python
# coding: utf-8

# # Mushroom classification using ML by KODI VENU

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
data=pd.read_csv('mushrooms.csv')


# Top 5 rows

# In[4]:


data.head()


# Last 5 rows

# In[5]:


data.tail()


# Dataset Shape

# In[6]:


data.shape


# In[7]:


print('Number of rows', data.shape[0])
print('Number of columns', data.shape[1])


# Dataset Information

# In[8]:


data.info()


# Check null values in the dataset

# In[9]:


data.isnull().sum()


# Dataset Statistics

# In[10]:


data.describe()


# Data Manipulation

# In[11]:


data.head()
data.info()


# In[12]:


data=data.astype('category')


# In[13]:


data.dtypes


# In[14]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for column in data.columns:
    data[column]=le.fit_transform(data[column])


# In[15]:


data.head()


# store feature matrix in X & response (target) in vector y

# In[16]:


X=data.drop('class',axis=1)
y=data['class']


# Applying PCA

# In[17]:


from sklearn.decomposition import PCA
pca1=PCA(n_components=7)
pca_fit=pca1.fit_transform(X)


# In[18]:


pca1.explained_variance_ratio_
sum(pca1.explained_variance_ratio_)


# Train/Test split

# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(pca_fit,y,test_size=0.20,random_state=42)


# In[20]:


y_train


# Import models

# In[21]:


data.head()


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Model Training

# In[26]:


lr=LinearRegression()
lr.fit(X_train,y_train)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

svc=SVC()
svc.fit(X_train,y_train)

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

gr=GradientBoostingClassifier()
gr.fit(X_train,y_train)


# Prediction on Test Data

# In[58]:


y_pred1=lr.predict(X_test)


# In[59]:


y_pred1


# In[60]:


y_pred2=knn.predict(X_test)


# In[61]:


y_pred2


# In[62]:


y_pred3=svc.predict(X_test)


# In[63]:


y_pred3


# In[64]:


y_pred4=dt.predict(X_test)


# In[65]:


y_pred4


# In[66]:


y_pred5=rf.predict(X_test)


# In[67]:


y_pred5


# In[68]:


y_pred6=gr.predict(X_test)


# In[69]:


y_pred6


# Evaluating the algorithm

# In[76]:


from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[78]:


accuracy_score(y_test,y_pred1)


# In[44]:


final_data=pd.DataFrame({'Models':['LR','KNN','SVC','DT','RF','GR'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100,accuracy_score(y_test,y_pred4)*100,accuracy_score(y_test,y_pred5)*100,accuracy_score(y_test,y_pred6)*100]


# In[45]:


final_data


# In[46]:


import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC'])


# Save the model

# In[47]:


rf_model=RandomForestClassifier()
rf_model.fit(pca_fit,y)


# In[48]:


import joblib
joblib.dump(rf_model,"Mushroom_prediction")
model=joblib.load('Mushroom_prediction')
P=model.predict(pca1.transform([[5,2,4,1,6,1,0,1,4,0,3,2,2,7,7,0,2,1,4,2,3,5]]))


# In[49]:


P


# In[50]:


if P[0]==1:
    print('Poissonous')
else:
    print('Edible')


# In[ ]:




