#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3


# In[3]:


conn = sqlite3.connect('nehasri.db')


# In[4]:


cur = conn.cursor()


# In[5]:


cur.execute("create table std(id int, name char)")


# In[6]:


cur.execute("insert into std values(1,'neha'),(2,'varsha'),(3,'sri')")


# In[7]:


for row in cur.execute('select * from std'):
    print(row)


# In[ ]:




