# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:47:38 2021

@author: 91876
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import ols
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

machine= pd.read_csv("C:/Users/91876/Desktop/Kaggle Individual/6_Machine predictive maintainence/predictive_maintenance.csv")
df= pd.DataFrame(machine)
df
df.isna().sum()
df.describe()
df.info()
df.columns
df.shape



''' Target variable '''
''' Target '''
df.Target.isna().sum()
df.Target.value_counts().sort_index()
df.Target.describe()
df.Target.unique()
''' Categorical '''
# No null values
''' 0 - No failure '''
''' 1 - Failure '''

#Countplot
sns.countplot(x ='Target', data = df)



df.info()
'''############   1 - UDI   ###########'''
df.UDI.isna().sum()
df.UDI.value_counts().sort_index()
df.UDI.describe()
df.UDI.unique()
''' Continuous '''

#Histogram
sns.distplot(df.UDI, color = 'red')
plt.xlabel('UDI')
plt.title('Histogram of UDI')

#Boxplot
plt.boxplot(df['UDI'],1,'rs',1)
plt.xlabel('UDI')
plt.ylabel('counts')
plt.title('Boxplot of UDI')

# there are no outliers

''' However we will drop this column as it is a unique id
    and it has no effect on the model '''




'''###########   2 - Product ID   ############'''
df= df.rename(columns= {'Product ID':'Product_ID'})
df.info()
df.Product_ID.isna().sum()
df.Product_ID.value_counts().sort_index()
df.Product_ID.describe()
df.Product_ID.unique()
''' Continuous '''

#Histogram
sns.distplot(df.Product_ID, color = 'red')
plt.xlabel('Product_ID')
plt.title('Histogram of Product_ID')

#Boxplot
plt.boxplot(df['Product_ID'],1,'rs',1)
plt.xlabel('Product_ID')
plt.ylabel('counts')
plt.title('Boxplot of Product_ID')

# strings can not be plotted
''' We will drop this column as it is a product id and every 
   product has a unique one, so no use '''


df = df.drop(['UDI','Product_ID'], axis = 1)
df.info()





'''###########   3 - Type   ###############'''
df.Type.isna().sum()
df.Type.value_counts().sort_index()
df.Type.describe()
df.Type.unique()
''' Categorical '''

#Countplot
sns.countplot(x ='Type', data = df)

#Individual boxplot
sns.boxplot(x='Type', y='Target', data = df)
tukey3= pairwise_tukeyhsd(df.Target, df.Type, alpha=0.05)
print(tukey3)

#Merging
df.Type.value_counts().sort_index()
df['Type'].replace('M', 'H', inplace = True)
df.Type.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='Type', data = df)

#Applying chi square coz categorical vs categorical
ct3 = pd.crosstab(df['Type'], df.Target)
ct3
chi2_contingency(ct3, correction = False)
'''P value is 0.00036487049777598574'''
'''Good predictor'''





'''##########   4 - Air temperature [K]   #############'''
df= df.rename(columns={'Air temperature [K]':'Air_temp'})
df.info()
df.Air_temp.isna().sum()
df.Air_temp.value_counts().sort_index()
df.Air_temp.describe()
df.Air_temp.unique()
''' Continuous '''

#Histogram
sns.distplot(df.Air_temp, color = 'red')
plt.xlabel('Air_temp')
plt.title('Histogram of Air_temp')

#Boxplot
plt.boxplot(df['Air_temp'],1,'rs',1)
plt.xlabel('Air_temp')
plt.ylabel('counts')
plt.title('Boxplot of Air_temp')

# there are no outliers

''' Continuous vs Categorical go for Ind t test '''

''' Independent T Test '''

df.Target.value_counts()

zero = df[df.Target == 0]
one = df[df.Target == 1]

scipy.stats.ttest_ind(zero.Air_temp, one.Air_temp)
''' Pvalue is 1.3548001481732193e-16 which is less than 0.05 '''
''' Good predictor '''





'''###########  5 - Process temperature [K]  #############'''
df= df.rename(columns={'Process temperature [K]':'Process_temp'})
df.info()
df.Process_temp.isna().sum()
df.Process_temp.value_counts().sort_index()
df.Process_temp.describe()
df.Process_temp.unique()
''' Continuous '''

#Histogram
sns.distplot(df.Process_temp, color = 'red')
plt.xlabel('Process_temp')
plt.title('Histogram of Process_temp')

#Boxplot
plt.boxplot(df['Process_temp'],1,'rs',1)
plt.xlabel('Process_temp')
plt.ylabel('counts')
plt.title('Boxplot of Process_temp')

# there are no outliers

''' Continuous vs Categorical go for Ind t test '''

''' Independent T Test '''

df.Target.value_counts()

zero = df[df.Target == 0]
one = df[df.Target == 1]

scipy.stats.ttest_ind(zero.Process_temp, one.Process_temp)
''' Pvalue is 0.00032400575504099217 which is less than 0.05 '''
''' Good predictor '''





'''############  6 - Rotational speed [rpm]   #############'''
df= df.rename(columns={'Rotational speed [rpm]':'Rotational_speed'})
df.info()
df.Rotational_speed.isna().sum()
df.Rotational_speed.value_counts().sort_index()
df.Rotational_speed.describe()
df.Rotational_speed.unique()
''' Continuous '''

#Histogram
sns.distplot(df.Rotational_speed, color = 'red')
plt.xlabel('Rotational_speed')
plt.title('Histogram of Rotational_speed')

#Boxplot
plt.boxplot(df['Rotational_speed'],1,'rs',1)
plt.xlabel('Rotational_speed')
plt.ylabel('counts')
plt.title('Boxplot of Rotational_speed')

# there are outliers

# Outliers Count
IQR6 = df['Rotational_speed'].quantile(0.75) - df['Rotational_speed'].quantile(0.25)
IQR6

UL6 = df['Rotational_speed'].quantile(0.75) + (1.5*IQR6)
UL6

df.Rotational_speed[(df.Rotational_speed > UL6)].value_counts().sum()
# 418

df.Rotational_speed = np.where(df.Rotational_speed > UL6, UL6, df.Rotational_speed)

df.Rotational_speed[(df.Rotational_speed > UL6)].value_counts().sum()
# 0

''' Continuous vs Categorical go for Ind t test '''

''' Independent T Test '''

df.Target.value_counts()

zero = df[df.Target == 0]
one = df[df.Target == 1]

scipy.stats.ttest_ind(zero.Rotational_speed, one.Rotational_speed)
''' Pvalue is 3.245575595115617e-39 which is less than 0.05 '''
''' Good predictor '''




'''############  7 - Torque [Nm]   ################'''
df= df.rename(columns={'Torque [Nm]':'Torque'})
df.info()
df.Torque.isna().sum()
df.Torque.value_counts().sort_index()
df.Torque.describe()
df.Torque.unique()
''' Continuous '''

#Histogram
sns.distplot(df.Torque, color = 'red')
plt.xlabel('Torque')
plt.title('Histogram of Torque')

#Boxplot
plt.boxplot(df['Torque'],1,'rs',1)
plt.xlabel('Torque')
plt.ylabel('counts')
plt.title('Boxplot of Torque')

# there are outliers

# Outliers Count
IQR7 = df['Torque'].quantile(0.75) - df['Torque'].quantile(0.25)
IQR7

UL7 = df['Torque'].quantile(0.75) + (1.5*IQR7)
UL7

LL7 = df['Torque'].quantile(0.25) - (1.5*IQR7)
LL7

df.Torque[(df.Torque > UL7)].value_counts().sum()
# 41
df.Torque[(df.Torque < LL7)].value_counts().sum()
# 28

df.Torque = np.where(df.Torque > UL7, UL7, df.Torque)

df.Torque = np.where(df.Torque < LL7, LL7, df.Torque)

df.Torque[(df.Torque > UL7)].value_counts().sum()
# 0
df.Torque[(df.Torque < LL7)].value_counts().sum()
# 0

''' Continuous vs Categorical go for Ind t test '''

''' Independent T Test '''

df.Target.value_counts()

zero = df[df.Target == 0]
one = df[df.Target == 1]

scipy.stats.ttest_ind(zero.Torque, one.Torque)
''' Pvalue is 1.2727414346609406e-82 which is less than 0.05 '''
''' Good predictor '''





'''###########  8 - Tool wear [min]   #############'''
df= df.rename(columns={'Tool wear [min]':'Tool_wear'})
df.info()
df.Tool_wear.isna().sum()
df.Tool_wear.value_counts().sort_index()
df.Tool_wear.describe()
df.Tool_wear.unique()
''' Continuous '''

#Histogram
sns.distplot(df.Tool_wear, color = 'red')
plt.xlabel('Tool_wear')
plt.title('Histogram of Tool_wear')

#Boxplot
plt.boxplot(df['Tool_wear'],1,'rs',1)
plt.xlabel('Tool_wear')
plt.ylabel('counts')
plt.title('Boxplot of Tool_wear')

# there are no outliers

''' Continuous vs Categorical go for Ind t test '''

''' Independent T Test '''

df.Target.value_counts()

zero = df[df.Target == 0]
one = df[df.Target == 1]

scipy.stats.ttest_ind(zero.Tool_wear, one.Tool_wear)
''' Pvalue is 3.9760759628693964e-26 which is less than 0.05 '''
''' Good predictor '''





'''############  9 - Failure Type   ################'''
df= df.rename(columns={'Failure Type':'Failure_Type'})
df.info()
df.Failure_Type.isna().sum()
df.Failure_Type.value_counts().sort_index()
df.Failure_Type.describe()
df.Failure_Type.unique()
''' Categorical '''

#Countplot
sns.countplot(x ='Failure_Type', data = df)

#Individual boxplot
sns.boxplot(x='Failure_Type', y='Target', data = df)
tukey9= pairwise_tukeyhsd(df.Target, df.Failure_Type, alpha=0.05)
print(tukey9)

#Merging
df.Failure_Type.value_counts().sort_index()
df['Failure_Type'].replace('Overstrain Failure', 'Heat Dissipation Failure', inplace = True)
df['Failure_Type'].replace('Power Failure', 'Heat Dissipation Failure', inplace = True)
df['Failure_Type'].replace('Tool Wear Failure', 'Heat Dissipation Failure', inplace = True)
df['Failure_Type'].replace('No Failure', 'Random Failures', inplace = True)
df.Failure_Type.value_counts().sort_index()

#Countplot after merging
sns.countplot(x ='Failure_Type', data = df)

#Applying chi square coz categorical vs categorical
ct9 = pd.crosstab(df['Failure_Type'], df.Target)
ct9
chi2_contingency(ct9, correction = False)
'''P value is 0.0'''
'''Bad predictor'''

''' EDA  is done '''

df.to_csv('C:/Users/91876/Desktop/Kaggle Individual/2_Machine predictive maintainence/Exported files/Machine_EDA.csv')




