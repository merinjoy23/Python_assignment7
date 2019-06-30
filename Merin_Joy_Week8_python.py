#!/usr/bin/env python
# coding: utf-8

# ### 1. Using sliding window, create data that includes lab values and selected vitals for predicting mortality within 24 hours. Apply to patients 65+ yo

# In[1]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


conn = sqlite3.connect('/Users/Merin/Desktop/HAP880/HAP_880_Project/mimic.db/mimic.db')


# In[4]:


# all admissions for patients older than 65
admissions = pd.read_sql('select *, (julianday(date(admittime))-julianday(date(dob)))/365.25 as age from admissions, patients where admissions.subject_id = patients.subject_id and (julianday(date(admittime))-julianday(date(dob)))/365.25 >= 65',conn)


# In[5]:


import time


# In[6]:


# calculate los in hours
admissions['los_hrs']=admissions.apply(lambda r: (time.mktime(time.strptime(r['DISCHTIME'],'%Y-%m-%d %H:%M:%S')) - 
                 time.mktime(time.strptime(r['ADMITTIME'],'%Y-%m-%d %H:%M:%S')))/3600.0, axis=1)


# In[7]:


admissions['los_hrs'][:20]


# In[8]:


# define loof forward and look backward windows
window_back = 24
window_forward = 24
shift = 1


# In[9]:


window_size = window_back + window_forward


# In[10]:


pts = admissions.iloc[:,1].unique()


# In[11]:


pts


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


#split admissions by training and testing 
trp, tsp = train_test_split(pts, train_size=0.8)
tr = admissions[admissions.iloc[:,1].isin(trp)]
ts = admissions[admissions.iloc[:,1].isin(tsp)]


# In[14]:


# select training data with more than minimul window size
tr_sel = tr[tr['los_hrs']>=window_size]


# In[15]:


# look at the first ad mission
pt = tr_sel[tr_sel['EXPIRE_FLAG']=='1'].iloc[0,:]


# In[16]:


pt


# In[17]:


dts = []
for i in admissions.index[:100]:
  pt = admissions.iloc[i]  
  print('Patient:', pt.iloc[1])
  for t in range(window_back, int(pt['los_hrs']), shift):
    #print('Current time: ', t)
    labs = pd.read_sql("select * from labevents where hadm_id = \"" + pt[2] + 
                       "\" and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 >= " + str( t - window_back) +
                       " and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 < " + str( t ), conn)
    #print('   found labs: ',len(labs.index))
    if len(labs.index) > 0:
        # output
        flag = 0
        if ((pt['HOSPITAL_EXPIRE_FLAG'] == '1') 
            # check if died on day of discharge
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_year == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_year)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mon == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mon)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mday == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mday)
            # check if within the window
            and (time.mktime(time.strptime(pt['ADMITTIME'],'%Y-%m-%d %H:%M:%S')) + (t + window_forward) * 3600 >
                 time.mktime(time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')) )
           ):
            flag = 1
            #print('yes')
               
        labs=labs.replace("",np.nan)
        labs['vn'] = labs.apply(lambda r: float(r['VALUENUM']),axis=1)
        gr = labs.groupby('ITEMID')
        #print(pd.DataFrame(gr['vn'].mean()).T)
        d = pd.DataFrame(gr['vn'].mean()).T
        d['class'] = flag
        dts.append(d)
print('concatenating data')
dt = pd.concat(dts, ignore_index=True) 
del dts


# In[33]:


dt['class'].mean()
# very imbalanced data


# In[34]:


# apply attribute selection to the data - mainly filters out a lot of non-predictive attributes
from sklearn.feature_selection import chi2
dt2=pd.DataFrame(dt1)


# In[35]:


cls = list(dt.columns)
cls.remove('class')
# frop columns with all missing values
dt=dt.dropna(axis=1, how='all')


# In[36]:


from sklearn.preprocessing import Imputer
imp = Imputer()
imp.fit(dt)
dt1=pd.DataFrame(imp.transform(dt),columns=list(dt.columns))


# In[37]:


cls = list(dt.columns)
cls.remove('class')


# In[38]:


dt2[dt2<0] = 0


# In[39]:


dt2=dt2.dropna(axis=1, how='all')


# In[40]:


chi2(dt2[cls],dt2['class'])


# ### 2. Use three different supervised learning methods to build models to predict mortality within 24 hours.

# ### Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(dt2[cls],dt2['class'])

probs = lr.predict_proba(dt2[cls])

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(dt2['class'], probs[:,1])

auc(fpr,tpr)

from matplotlib import pyplot as plt

plt.plot(fpr,tpr)


# ### Random Forest

# In[42]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(dt2[cls],dt2['class'])

probs = rf.predict_proba(dt2[cls])

fpr, tpr, thresholds = roc_curve(dt2['class'], probs[:,1])
auc(fpr,tpr)
plt.plot(fpr,tpr)


# ### Naive Bayes

# In[43]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(dt2[cls],dt2['class'])

probs = nb.predict_proba(dt2[cls])

fpr, tpr, thresholds = roc_curve(dt2['class'], probs[:,1])
auc(fpr,tpr)
plt.plot(fpr,tpr)


# ### 3. Test if changing look backward window size between 6, 12, 24 hours  makes difference in model accracy.

# ### Window_back = 6

# In[46]:


# define loof forward and look backward windows
window_back = 6
window_forward = 24
shift = 1
window_size = window_back + window_forward
pts = admissions.iloc[:,1].unique()

from sklearn.model_selection import train_test_split

#split admissions by training and testing 
trp, tsp = train_test_split(pts, train_size=0.8)
tr = admissions[admissions.iloc[:,1].isin(trp)]
ts = admissions[admissions.iloc[:,1].isin(tsp)]
# select training data with more than minimul window size
tr_sel = tr[tr['los_hrs']>=window_size]
# look at the first ad mission
pt = tr_sel[tr_sel['EXPIRE_FLAG']=='1'].iloc[0,:]

dts = []
for i in admissions.index[:100]:
  pt = admissions.iloc[i]  
  print('Patient:', pt.iloc[1])
  for t in range(window_back, int(pt['los_hrs']), shift):
    #print('Current time: ', t)
    labs = pd.read_sql("select * from labevents where hadm_id = \"" + pt[2] + 
                       "\" and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 >= " + str( t - window_back) +
                       " and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 < " + str( t ), conn)
    #print('   found labs: ',len(labs.index))
    if len(labs.index) > 0:
        # output
        flag = 0
        if ((pt['HOSPITAL_EXPIRE_FLAG'] == '1') 
            # check if died on day of discharge
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_year == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_year)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mon == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mon)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mday == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mday)
            # check if within the window
            and (time.mktime(time.strptime(pt['ADMITTIME'],'%Y-%m-%d %H:%M:%S')) + (t + window_forward) * 3600 >
                 time.mktime(time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')) )
           ):
            flag = 1
            #print('yes')
               
        labs=labs.replace("",np.nan)
        labs['vn'] = labs.apply(lambda r: float(r['VALUENUM']),axis=1)
        gr = labs.groupby('ITEMID')
        #print(pd.DataFrame(gr['vn'].mean()).T)
        d = pd.DataFrame(gr['vn'].mean()).T
        d['class'] = flag
        dts.append(d)
print('concatenating data')
dt = pd.concat(dts, ignore_index=True) 
del dts


# In[48]:


from sklearn.feature_selection import chi2
dt2=pd.DataFrame(dt1)

cls = list(dt.columns)
cls.remove('class')
# frop columns with all missing values
dt=dt.dropna(axis=1, how='all')

from sklearn.preprocessing import Imputer
imp = Imputer()
imp.fit(dt)
dt1=pd.DataFrame(imp.transform(dt),columns=list(dt.columns))

cls = list(dt.columns)
cls.remove('class')

dt2[dt2<0] = 0

dt2=dt2.dropna(axis=1, how='all')

chi2(dt2[cls],dt2['class'])

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(dt1[cls],dt1['class'])

probs = lr.predict_proba(dt1[cls])

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(dt1['class'], probs[:,1])

auc(fpr,tpr)

from matplotlib import pyplot as plt

plt.plot(fpr,tpr)


# ### Window size = 12

# In[49]:


# define loof forward and look backward windows
window_back = 12
window_forward = 24
shift = 1
window_size = window_back + window_forward
pts = admissions.iloc[:,1].unique()

from sklearn.model_selection import train_test_split

#split admissions by training and testing 
trp, tsp = train_test_split(pts, train_size=0.8)
tr = admissions[admissions.iloc[:,1].isin(trp)]
ts = admissions[admissions.iloc[:,1].isin(tsp)]
# select training data with more than minimul window size
tr_sel = tr[tr['los_hrs']>=window_size]
# look at the first ad mission
pt = tr_sel[tr_sel['EXPIRE_FLAG']=='1'].iloc[0,:]

dts = []
for i in admissions.index[:100]:
  pt = admissions.iloc[i]  
  print('Patient:', pt.iloc[1])
  for t in range(window_back, int(pt['los_hrs']), shift):
    #print('Current time: ', t)
    labs = pd.read_sql("select * from labevents where hadm_id = \"" + pt[2] + 
                       "\" and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 >= " + str( t - window_back) +
                       " and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 < " + str( t ), conn)
    #print('   found labs: ',len(labs.index))
    if len(labs.index) > 0:
        # output
        flag = 0
        if ((pt['HOSPITAL_EXPIRE_FLAG'] == '1') 
            # check if died on day of discharge
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_year == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_year)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mon == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mon)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mday == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mday)
            # check if within the window
            and (time.mktime(time.strptime(pt['ADMITTIME'],'%Y-%m-%d %H:%M:%S')) + (t + window_forward) * 3600 >
                 time.mktime(time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')) )
           ):
            flag = 1
            #print('yes')
               
        labs=labs.replace("",np.nan)
        labs['vn'] = labs.apply(lambda r: float(r['VALUENUM']),axis=1)
        gr = labs.groupby('ITEMID')
        #print(pd.DataFrame(gr['vn'].mean()).T)
        d = pd.DataFrame(gr['vn'].mean()).T
        d['class'] = flag
        dts.append(d)
print('concatenating data')
dt = pd.concat(dts, ignore_index=True) 
del dts


# In[50]:


from sklearn.feature_selection import chi2
dt2=pd.DataFrame(dt1)

cls = list(dt.columns)
cls.remove('class')
# frop columns with all missing values
dt=dt.dropna(axis=1, how='all')

from sklearn.preprocessing import Imputer
imp = Imputer()
imp.fit(dt)
dt1=pd.DataFrame(imp.transform(dt),columns=list(dt.columns))

cls = list(dt.columns)
cls.remove('class')

dt2[dt2<0] = 0

dt2=dt2.dropna(axis=1, how='all')

chi2(dt2[cls],dt2['class'])

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(dt1[cls],dt1['class'])

probs = lr.predict_proba(dt1[cls])

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(dt1['class'], probs[:,1])

auc(fpr,tpr)

from matplotlib import pyplot as plt

plt.plot(fpr,tpr)


# ### Window size = 24

# In[51]:


# define loof forward and look backward windows
window_back = 24
window_forward = 24
shift = 1
window_size = window_back + window_forward
pts = admissions.iloc[:,1].unique()

from sklearn.model_selection import train_test_split

#split admissions by training and testing 
trp, tsp = train_test_split(pts, train_size=0.8)
tr = admissions[admissions.iloc[:,1].isin(trp)]
ts = admissions[admissions.iloc[:,1].isin(tsp)]
# select training data with more than minimul window size
tr_sel = tr[tr['los_hrs']>=window_size]
# look at the first ad mission
pt = tr_sel[tr_sel['EXPIRE_FLAG']=='1'].iloc[0,:]

dts = []
for i in admissions.index[:100]:
  pt = admissions.iloc[i]  
  print('Patient:', pt.iloc[1])
  for t in range(window_back, int(pt['los_hrs']), shift):
    #print('Current time: ', t)
    labs = pd.read_sql("select * from labevents where hadm_id = \"" + pt[2] + 
                       "\" and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 >= " + str( t - window_back) +
                       " and (julianday(CHARTTIME) - julianday(\"" + pt[3] +
                       "\"))*24.0 < " + str( t ), conn)
    #print('   found labs: ',len(labs.index))
    if len(labs.index) > 0:
        # output
        flag = 0
        if ((pt['HOSPITAL_EXPIRE_FLAG'] == '1') 
            # check if died on day of discharge
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_year == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_year)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mon == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mon)
            and ((time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')).tm_mday == (time.strptime(pt['DISCHTIME'],'%Y-%m-%d %H:%M:%S')).tm_mday)
            # check if within the window
            and (time.mktime(time.strptime(pt['ADMITTIME'],'%Y-%m-%d %H:%M:%S')) + (t + window_forward) * 3600 >
                 time.mktime(time.strptime(pt['DOD_HOSP'],'%Y-%m-%d %H:%M:%S')) )
           ):
            flag = 1
            #print('yes')
               
        labs=labs.replace("",np.nan)
        labs['vn'] = labs.apply(lambda r: float(r['VALUENUM']),axis=1)
        gr = labs.groupby('ITEMID')
        #print(pd.DataFrame(gr['vn'].mean()).T)
        d = pd.DataFrame(gr['vn'].mean()).T
        d['class'] = flag
        dts.append(d)
print('concatenating data')
dt = pd.concat(dts, ignore_index=True) 
del dts


# In[52]:


from sklearn.feature_selection import chi2
dt2=pd.DataFrame(dt1)

cls = list(dt.columns)
cls.remove('class')
# frop columns with all missing values
dt=dt.dropna(axis=1, how='all')

from sklearn.preprocessing import Imputer
imp = Imputer()
imp.fit(dt)
dt1=pd.DataFrame(imp.transform(dt),columns=list(dt.columns))

cls = list(dt.columns)
cls.remove('class')

dt2[dt2<0] = 0

dt2=dt2.dropna(axis=1, how='all')

chi2(dt2[cls],dt2['class'])

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(dt1[cls],dt1['class'])

probs = lr.predict_proba(dt1[cls])

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(dt1['class'], probs[:,1])

auc(fpr,tpr)

from matplotlib import pyplot as plt

plt.plot(fpr,tpr)


# In[ ]:




