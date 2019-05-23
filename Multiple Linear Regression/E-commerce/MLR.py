# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:45:13 2019

@author: arush
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks",color_codes=True)
import scipy.stats as stats

df=pd.read_csv("Ecommerce Customers.csv")

c=df.corr()
print(c)

x=df.iloc[:,[5,6,7]]

y=df['Time on App']

y=pd.DataFrame(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#linearity
sns.pairplot(pd.DataFrame(x),kind='reg')

from sklearn.linear_model import LinearRegression
x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)

x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)

regressor=LinearRegression()
regressor.fit(x_train,y_train)



sns.set(color_codes=True)

#kfold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor, X=x_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

y_pred=regressor.predict(x_test)
#endogeneity
error_residual = pd.DataFrame(y_test-y_pred)
error_residual.reset_index(inplace = True)
linearity_test_df = pd.DataFrame(x_test)
linearity_test_df['Residual'] = error_residual['Time on App']
sns.pairplot(linearity_test_df.iloc[:, 1:], kind="reg")
#homoscedacity
residual_test = np.column_stack([y_test,y_pred])
residual_test = pd.DataFrame(residual_test)
residual_test.columns='Y_test predictions'.split()
sns.jointplot(x='Y_test', y='predictions', data=residual_test, kind='reg')
stats.levene(residual_test['Y_test'], residual_test['predictions'])

#normality
stats.shapiro(error_residual['Time on App'])

from sklearn.metrics import mean_squared_error, r2_score

# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
print("Variance score: {}".format(r2_score(y_test, y_pred)))



