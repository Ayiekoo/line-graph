#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


from matplotlib import pyplot as plt


# In[8]:


x = [3, 6, 9]
y = [2, 4, 6]
z = [4, 7, 10]
plt.plot(x, y)
plt.plot(x, z)
plt.title("test plot")
plt.xlabel("time")
plt.ylabel("volume and pressure")
plt.legend(["this is volume", "this is pressure"])
plt.show()


# In[ ]:




