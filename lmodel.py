#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:30:04 2024

@author: bishwaneupane
 prdiction model for Ecommerce Customers
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

####reading or the data 

dataset=pd.read_csv('Ecommerce Customers.txt')

dataset.info()

features=dataset.columns[3:-1]
target=dataset.columns[-1]


X=dataset[features]
y=dataset[target]

X=np.array(X)
y=np.array(y)


#split the data into train test data set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=101,test_size=0.2)

#scale the train data 
from sklearn.preprocessing import StandardScaler

scalar=StandardScaler()

#X_train_scale=scalar.fit_transform(X_train)

#X_test_scale=scalar.transform(X_test)

#selection of model

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


lr=LinearRegression()

steps=[('scaler',scalar),('model',lr)]

pipe=Pipeline(steps)

pipe.fit(X_train,y_train)
pipe['model'].coef_
pipe['model'].intercept_

y_predict=pipe.predict(X_test)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_predict, ax=ax,s=500)
ax.set_xlabel('Actual',fontsize=20)
ax.set_ylabel('Predict',fontsize=20)
ax.tick_params(axis='x', labelsize=20)  # Adjust fontsize of xticks as needed
ax.tick_params(axis='y', labelsize=20)  # Adjust fontsize of yticks as needed

# Adding the regression line
sns.regplot(x=y_test, y=y_predict, ax=ax,line_kws={"linewidth":5,'color':'red'})

plt.show()



import pickle
pickle.dump(pipe, open('lr_model.pkl','wb'))

model = pickle.load(open('lr_model.pkl','rb'))
print(model.predict([[32.2,14.8,38.3,1.52]]))


