#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing requried packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[2]:


###### loading dataset from remote url ####

url = 'http://bit.ly/w-data'
Data = pd.read_csv(url)
print('Data loaded successfully')


# In[3]:


Data.head(10)


# In[4]:


Data.tail()


# In[5]:


###### Find the shape of data ####
Data.shape


# In[6]:


##### Find the information of the data ####
Data.info()


# In[7]:


#### find the statistical properties of the dataset
Data.describe()


# In[8]:


#### Find the unique values of the data
Data.nunique()


# In[10]:


##### Data visualization ####

Data.plot(x = "Hours", y = "Scores", style = '*', color = "red")
plt.title('Hours vs. Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.show()


# In[11]:


###### Plotting correlation between feature and target
sns.heatmap(Data.corr(), annot = True, linecolor = 'black')


# In[13]:


###### Plotting regression plot to confirm the above relationship between feature and target
sns.regplot(x = Data['Hours'], y = Data['Scores'], data = Data, color = 'red')
plt.title('study hours vs Percentage scores')
plt.xlabel('study hours')
plt.ylabel('percentage')
plt.show()


# In[14]:


##### DATA PREPARATION ######
### DIVIDING DATA ####

x = Data.iloc[:, :-1].values  ### This is an attribute #####
y = Data.iloc[:, 1].values  #### Lavels######
print("Hours studied = ", x[0:5])
print("scores obtained = ", y[0:5])


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.24, random_state = 42)


# In[21]:


##### Train-Test Split #####
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("Training complete.")


# In[23]:


##### finding regression coefficient and intercept points
print ("Coefficient -", regressor.coef_)
print ("Intercept - ", regressor.intercept_)


# In[25]:


##### plot the regression line ####

line = regressor.coef_*x + regressor.intercept_

#### plot for the test data ####
plt.scatter(x, y)
plt.plot(x, line, color = 'green', label = 'Line of regression')
plt.legend()
plt.show()


# In[26]:


##### PREDICTION #####
print(x_test) ### This is for the data test ####
y_pred = regressor.predict(x_test) ### This is to predict the data


# In[27]:


######## Compare original data and the predicted values##########
df = pd.DataFrame({'Original':y_test, 'Predicted':y_pred})
df


# In[29]:


########## Training and Testing scores ##########
print("Training Score:", regressor.score(x_train, y_train))
print("Test Score:", regressor.score(x_test, y_test))


# In[30]:


#### FROM THIS, ONE CAN CLEARLY SEE THE ACCURACY

##### PLOT THE ORIGINAL AND PREDICTED VALUES
df.plot(kind = 'bar', figsize = (8,6))
plt.show()


# In[31]:


####### TESTING ######
##### give a value of study hours to know the score

### TESTING WITH NEW DATA #####

hours = float(input("Enter the study hours : "))
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[33]:


##### EVALUATING THE MODEL
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', metrics.r2_score(y_test, y_pred))


# In[ ]:




