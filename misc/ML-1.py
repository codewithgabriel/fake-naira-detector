# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:40:18 2025

@author: HP
"""
#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%

# data visualization 
dataset = pd.read_csv("Salary_Data.csv")
dataset.describe()

#%%

# data preprocessing 
# removing null/nan values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan)
dataset = imputer.fit_transform(dataset)


#%%
X = dataset[: , 0]
y = dataset[: , 1]
#%%
# data scaling 
from sklearn.preprocessing import StandardScaler
sc_y = StandardScaler() 
dataset = sc_y.fit_transform(dataset)


#%%
# splitting the dataset 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y, test_size=0.2, random_state=0)

#%%


# training the model 
# test the model


