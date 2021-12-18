#!/usr/bin/env python
# coding: utf-8

# FAKE NEWS DETECTION

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#Data flair - make necessary imports
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


get_ipython().system('pip install pandas')


# In[4]:


#read the data
df=pd.read_csv('news.csv')

#get shape and head
df.shape
df.head()


# In[16]:


#Dataflair - get the labels
labels=df.label
labels.head()


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

 


# In[19]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[23]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[24]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[30]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:





# In[ ]:




