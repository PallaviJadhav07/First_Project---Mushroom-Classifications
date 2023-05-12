#!/usr/bin/env python
# coding: utf-8
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/shitake-tasty-mushroom-isolated-white-background_188078-18051.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

import pickle
#libraries
import pandas as pd # data processing
import numpy as np # linear algebra

#ploting libraries
import seaborn as sns
import matplotlib.pyplot as plt 

#feature engineering
from sklearn import preprocessing

# data transformation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')


from sklearn.svm import SVC

import pickle as pickle5
import streamlit as st
st.title("Mushroom Type Prediction")
#st.title("Mushroom Classification")
st.subheader("Are your Mushrooms Edible or Poisonous..? Lets find out...")
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://unsplash.com/photos/o9JapjMqjfc.jpg")
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html = True
    )
add_bg_from_url()
        

#PallaviJadhav
# In[70]:


# Loading data
data=pd.read_csv('mushrooms.csv')


# In[71]:


# Droppping insignificant columns
data=data.drop(['gill-attachment','ring-number','stalk-surface-below-ring','stalk-color-below-ring','veil-type','veil-color'],axis=1)


# In[72]:



# Renaming Column names
data.rename(columns = {'cap-shape':'cap_shape','cap-surface':'cap_surface',
                       'cap-color':'cap_color','gill-spacing':'gill_spacing',
                       'gill-size':'gill_size','gill-color':'gill_color',
                       'stalk-shape':'stalk_shape','stalk-root':'stalk_root',
                       'stalk-surface-above-ring':'stalk_surface_above_ring',
                       'stalk-color-above-ring':'stalk_color_above_ring',
                       'ring-type':'ring_type','spore-print-color':'spore_print_color'}, inplace = True)


# In[73]:



data['cap_shape']=np.where(data['cap_shape'].isin(['x','f','k']),
                           data['cap_shape'].str.title(),
                           'Other_shape')


# in cap-surface contribution of 'g=grooves'is negligible,so we can directly drop it.
data=data[data['cap_surface'] != 'g']


# In[38]:


data['cap_color']=np.where(data['cap_color'].isin(['n','y','w','g','e']),
                           data['cap_color'].str.title(),
                           'Other_color')


# In[39]:


data['odor']=np.where(data['odor'].isin(['n','f','y','s','a','l']),
                           data['odor'].str.title(),
                           'Other')


# In[40]:


data['gill_color']=np.where(data['gill_color'].isin(['k','n','g','p','w','h','u','b']),
                           data['gill_color'].str.title(),
                           'Other_color')


# In[41]:


data['stalk_root']=np.where(data['stalk_root'].isin(['e','b']),
                           data['stalk_root'].str.title(),
                           'Other')


# In[42]:


data['stalk_surface_above_ring']=np.where(data['stalk_surface_above_ring'].isin(['s','k']),
                           data['stalk_surface_above_ring'].str.title(),
                           'Other')


# In[43]:


data['stalk_color_above_ring']=np.where(data['stalk_color_above_ring'].isin(['w','g','p','n','b']),
                           data['stalk_color_above_ring'].str.title(),
                           'Other')


# In[44]:


data['ring_type']=np.where(data['ring_type'].isin(['p','e','l']),
                           data['ring_type'].str.title(),
                           'Other')


# In[45]:


data['spore_print_color']=np.where(data['spore_print_color'].isin(['k','n','h','w']),
                           data['spore_print_color'].str.title(),
                           'Other')


# In[46]:


data['population']=np.where(data['population'].isin(['s','v','y']),
                           data['population'].str.title(),
                           'Other')


# In[47]:


data['habitat']=np.where(data['habitat'].isin(['g','d','p','l']),
                           data['habitat'].str.title(),
                           'Other')



# In[74]:


data['cap_shape']=data['cap_shape'].replace({'X':'convex','F':'flat','K':'knobbed'})
data['cap_surface']=data['cap_surface'].replace({'s':'smooth','y':'scaly','f':'fibrous'})
data['cap_color']=data['cap_color'].replace({'N':'brown','Y':'yellow','W':'white','G':'grey','E':'red'})
data['bruises']=data['bruises'].replace({'t':'bruises','f':'no_bruises'})
data['odor']=data['odor'].replace({'N':'none','F':'foul','Y':'fishy','S':'spicy','A':'almond','L':'anise'})
data['gill_spacing']=data['gill_spacing'].replace({'c':'close','w':'crowded'})


data['gill_size']=data['gill_size'].replace({'n':'narrow','b':'broad'})

data['gill_color']=data['gill_color'].replace({'K':'black','N':'brown','G':'grey','P':'pink','W':'white','H':'chocolate','U':'purple','B':'buff'})

data['stalk_shape']=data['stalk_shape'].replace({'e':'enlarging','t':'tapering'})

data['stalk_root']=data['stalk_root'].replace({'E':'equal','B':'bulbous'})

data['stalk_surface_above_ring']=data['stalk_surface_above_ring'].replace({'S':'smoth','K':'silky'})

data['stalk_color_above_ring']=data['stalk_color_above_ring'].replace({'W':'white','G':'grey','P':'pink','N':'brown','B':'buff'})

data['ring_type']=data['ring_type'].replace({'P': 'pendant','E':'evanescent','L':'large'})

data['spore_print_color']=data['spore_print_color'].replace({'K':'black','N':'brown','H':'chocolate','W':'white'})

data['population']=data['population'].replace({'S':'scattered','V':'several','Y':'solitary'})

data['habitat']=data['habitat'].replace({'G':'grasses','D':'woods','P':'paths','L':'leaves'})
data['class']=data['class'].replace({'p':'Poisonous','e':'Edible'})
data.head(10)


# In[75]:


# In[48]:
data['class'].unique()




# In[76]:


m_cap_shape =st.selectbox('cap_shape', data['cap_shape'].unique())

if m_cap_shape=='convex':
    m_cap_shape=0
elif m_cap_shape=='flat':
    m_cap_shape=1
elif m_cap_shape=='knobbed':
    m_cap_shape=2
else:
    m_cap_shape=3

m_cap_surface = st.selectbox('mushroom_surface', data['cap_surface'].unique())
if m_cap_surface=='smooth':
    m_cap_surface=0
elif m_cap_surface=='scaly':
    m_cap_surface=1
else:
    m_cap_surface=2
    
m_cap_color = st.selectbox('cap_color', data['cap_color'].unique())
if m_cap_color=='brown':
    m_cap_color=0
elif m_cap_color=='yellow':
    m_cap_color=1
elif m_cap_color=='white':
    m_cap_color=2 
elif m_cap_color=='grey':
    m_cap_color=3    
elif m_cap_color=='red':
    m_cap_color=4
else:
    m_cap_color=5
    
m_bruises = st.selectbox('bruises', data['bruises'].unique())
if m_bruises=='bruises':
    m_bruises=0
else:
    m_bruises=1
    
m_odor = st.selectbox('odor', data['odor'].unique())
if m_odor=='almond':
    m_odor=0
elif m_odor=='anise':
    m_odor=1
elif m_odor=='none':
    m_odor=2
elif m_odor=='foul':
    m_odor=3
elif m_odor=='fishy':
    m_odor=4
elif m_odor=='spicy':
    m_odor=5
else:
    m_odor=6
    
m_gill_spacing = st.selectbox('gill_spacing', data['gill_spacing'].unique())
if m_gill_spacing=='close':
    m_gill_spacing=0
else:
    m_gill_spacing=1
    
m_gill_size = st.selectbox('gill_size', data['gill_size'].unique())
if m_gill_size=='narrow':
    m_gill_size=0
else:
    m_gill_size=1
    
m_gill_color = st.selectbox('gill_color', data['gill_color'].unique())
if m_gill_color=='black':
    m_gill_color=0
elif m_gill_color=='brown':
    m_gill_color=1
elif m_gill_color=='grey':
    m_gill_color=2
elif m_gill_color=='pink':
    m_gill_color=3
elif m_gill_color=='white':
    m_gill_color=4
elif m_gill_color=='chocolate':
    m_gill_color=5
elif m_gill_color=='purple':
    m_gill_color=6
else:
    m_gill_color=7
    
m_stalk_shape  = st.selectbox('stalk_shape', data['stalk_shape'].unique())
if m_stalk_shape=='enlarging':
    m_stalk_shape=0
else:
    m_stalk_shape=1
    
m_stalk_root = st.selectbox('stalk_root', data['stalk_root'].unique())
if m_stalk_root=='equal':
    m_stalk_root=0
elif m_stalk_root=='bulbous':
    m_stalk_root=1
else:
    m_stalk_root=2
    
m_stalk_surface_above_ring = st.selectbox('stalk_surface_above_ring', data['stalk_surface_above_ring'].unique())
if m_stalk_surface_above_ring=='smooth':
    m_stalk_surface_above_ring=0
elif m_stalk_surface_above_ring=='silky':
    m_stalk_surface_above_ring=1
else:
    m_stalk_surface_above_ring=2
    
m_stalk_color_above_ring = st.selectbox('stalk_color_above_ring', data['stalk_color_above_ring'].unique())
if m_stalk_color_above_ring=='white':
    m_stalk_color_above_ring=0
elif m_stalk_color_above_ring=='grey':
    m_stalk_color_above_ring=1
elif m_stalk_color_above_ring=='pink':
    m_stalk_color_above_ring=2
elif m_stalk_color_above_ring=='brown':
    m_stalk_color_above_ring=3
elif m_stalk_color_above_ring=='buff':
    m_stalk_color_above_ring=4
else:
    m_stalk_color_above_ring=5
    
m_ring_type  = st.selectbox('ring_type', data['ring_type'].unique())
if m_ring_type=='pendant':
    m_ring_type=0
elif m_ring_type=='evanescent':
    m_ring_type=1
elif m_ring_type=='large':
    m_ring_type=2
else:
    m_ring_type=3
    
m_spore_print_color = st.selectbox('spore_print_color', data['spore_print_color'].unique())
if m_spore_print_color=='black':
    m_spore_print_color=0
elif m_spore_print_color=='brown':
    m_spore_print_color=1
elif m_spore_print_color=='chocolate':
    m_spore_print_color=2
elif m_spore_print_color=='white':
    m_spore_print_color=3
else:
    m_spore_print_color=4
    
m_population = st.selectbox('population', data['population'].unique())
if m_population=='scattered':
    m_population=0
elif m_population=='several':
    m_population=1
elif m_population=='solitary':
    m_population=2
else:
    m_population=3
    
m_habitat = st.selectbox('habitat', data['habitat'].unique())
if m_habitat=='grasses':
    m_habitat=0
elif m_habitat=='woods':
    m_habitat=1
elif m_habitat=='paths':
    m_habitat=2
elif m_habitat=='leaves':
    m_habitat=3
else:
    m_habitat=4


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
data['class']= lr.fit_transform(data['class'])
data['cap_shape']= lr.fit_transform(data['class'])
data['cap_surface']= lr.fit_transform(data['class'])
data['cap_color']= lr.fit_transform(data['class'])
data['bruises']= lr.fit_transform(data['class'])
data['odor']= lr.fit_transform(data['class'])
data['gill_spacing']= lr.fit_transform(data['class'])
data['gill_size']= lr.fit_transform(data['class'])
data['gill_color']= lr.fit_transform(data['class'])
data['stalk_shape']= lr.fit_transform(data['class'])
data['stalk_root']= lr.fit_transform(data['class'])
data['stalk_surface_above_ring']= lr.fit_transform(data['stalk_surface_above_ring'])
data['stalk_color_above_ring']= lr.fit_transform(data['stalk_color_above_ring'])
data['ring_type']= lr.fit_transform(data['ring_type'])
data['spore_print_color']= lr.fit_transform(data['spore_print_color'])
data['population']= lr.fit_transform(data['population'])
data['habitat']= lr.fit_transform(data['habitat'])


# In[78]:


# Dividing data into Features(X) & Target(y)
X = data.iloc[:,1:]


y=data['class']

# Train-Test Split 
#Train test split will be a 70:30 ratio respectively.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[79]:


#SVM Clasification
svm = SVC(C=1, kernel='linear')         #bydefault kernel=rbf      C=to control soft margin
svm1=svm.fit(X_train,y_train)
result_svm = svm1.score(X_test,y_test)


# In[81]:


filename = 'final_svm_model.pkl'
pickle.dump(svm, open(filename,'wb'))
pickled_model=pickle.load(open('final_svm_model.pkl','rb'))
pickled_model.fit(X_train,y_train)
pk=pickled_model.predict(X_test)


# In[82]:


if st.button('Mushroom type'):
    df={'cap_shape':m_cap_shape,'cap_surface':m_cap_surface,' cap_color':m_cap_color,' bruises':m_bruises,' odor':m_odor,'gill_spacing':m_gill_spacing,'gill_size':m_gill_size,' gill_color':m_gill_color,' stalk_shape':m_stalk_shape,' stalk_root':m_stalk_root,'stalk_surface_above_ring':m_stalk_surface_above_ring,'stalk_color_above_ring':m_stalk_color_above_ring,' ring_type':m_ring_type,' spore_print_color':m_spore_print_color,' population':m_population,'habitat':m_habitat}
  

   
    df1=pd.DataFrame(df,index=[1])
    df1==pd.get_dummies(df1)
    predictions=pickled_model.predict(df1)
    
    if predictions.any()==1:
        prediction_value = 'Poisonous'
    else:
        prediction_value = 'Edible'
    
    st.title("Mushroom type is " + str(prediction_value))
        




# 
