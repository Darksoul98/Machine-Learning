# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:40:50 2019

@author: arush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import scipy.stats as stats
import statsmodels.api as sm

df=pd.read_csv("forestfires.csv")
x=df['DC']
y=df['DMC']

x=x.fillna(x.mean())
y=y.fillna(y.mean())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)

x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
import seaborn as sns
sns.set(color_codes=True)

#training
df_training=pd.DataFrame()
df_training['DC']=x_train['DC']
df_training['DMC']=y_train
ax=sns.regplot(x="DC",y="DMC",data=df_training)

#making predictions
y_pred=regressor.predict(x_test)

#testing
df_test=pd.DataFrame()
df_test['DC']=x_test['DC']
df_test['DMC']=y_test
ax=sns.regplot(x="DC",y="DMC",data=df_test)

print('Coefficients: \n', regressor.coef_)
from sklearn.metrics import mean_squared_error, r2_score

print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

print("Variance score: {}".format(r2_score(y_test, y_pred)))

import statsmodels.api as sm

x = sm.add_constant(x)
results = sm.OLS(endog = y, exog=x).fit()
results.summary()

