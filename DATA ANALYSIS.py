#!/usr/bin/env python
# coding: utf-8

# In[8]:


### import the datasets
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


boston_real_estate_data = load_boston


# In[14]:


print(DataFrame)


# In[15]:


DataFrame.head()


# In[16]:


## we are creating a histogram for the data
#### plotting a graph for each column

DataFrame.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


# In[18]:


### Heat maps

"""
ADVANTAGES
1. Draws attention to risk-prone areas
2. Uses the entire dataset to draw meaningful insights
3. Is used for cluster analysis and can deal with large datasets
"""

# a demo of heatmaps

import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


flight_data = sns.load_dataset('flights')


# In[21]:


flight_data.head()


# In[24]:


flight_data = flight_data.pivot('month','year','passengers')


# In[27]:


print(flight_data)


# In[28]:


sns.heatmap(flight_data)


# In[29]:


### pie charts

#### the dataset
job_data = ['40','20','17','8','5','10']

#### defining the labels of the departments

labels = 'Alex','Emily','Husband','Wife','Jomo','Aoko'

### size to explode


explode = (0.05,0,0,0,0,0)

### draw the pie chart and setting the parameters
plt.pie(job_data,labels=labels,explode=explode)
plt.show()


# In[30]:


### METHODS OF MULTIPLICATION
numbers = [3, 9, 11, 15]
squared_numbers = []
for num in numbers:
    squared_numbers.append(num ** 2)
print(squared_numbers)


# In[32]:


### METHOD 2 OF MULTIPLICATION
numbers = [20, 25, 30, 35, 40]
squared_numbers = [num ** 2 for num in numbers] #### this is a shorter code for squaring numbers

print(squared_numbers)


# In[ ]:




