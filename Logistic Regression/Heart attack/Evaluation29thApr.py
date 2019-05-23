# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:37:40 2019

@author: bhuvnesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv("dataLR.csv")
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

x=df['target'].value_counts()

sns.countplot(x=df['sex'],data=df)

x=df.groupby('target').median()

sns.countplot(x=df['age'],data=df)

x=df[df['target']==1]
y=x.groupby('cp').count()

y=df.iloc[:,[13]]
x=df.drop('target',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

tn=confusion_matrix[0][0]
tp=confusion_matrix[1][1]
fp=confusion_matrix[0][1]
fn=confusion_matrix[1][0]

accuracy=(tp+tn)/(tp+tn+fp+fn)
error=(fp+fn)/(tp+tn+fp+fn)
sensitivity=tp/(tp+fn)
specificity=tn/fp+tn
recall=tp/(tp+fn)
precision=tp/(fp+tp)
fscore=(2*precision*recall)/(precision+recall)
print("Accuracy=",accuracy)
print("Error=",error)
print("Sensitivity=",sensitivity)
print("Specificity=",specificity)
print("Recall=",recall)
print("Precision=",precision)
print("Fscore=",fscore)



df_s = df['slope']

df_onehot = pd.get_dummies(df_s, drop_first=False)

df.drop('slope', axis=1, inplace=True)

df1 = pd.concat([df, df_onehot], axis=1)


