# -*- coding: utf-8 -*-
"""
author: Anushka
"""

#Importing Dataset
import pandas as pd
dataset = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print(dataset)

#Plotting Scatterplot to visualise the relation between variables
import matplotlib.pyplot as plt
dataset.plot(x='Hours',y='Scores',style='o')
plt.title('Scatter Plot of Variables')
plt.xlabel('Hours Studies')
plt.ylabel('Marks Obtained')

print('Coefficient of correlation:')
dataset.corr()

'''
Performing Linear Regression
'''

#Creating arrays from table
import numpy as np
Hours = dataset.Hours
x=np.array(Hours).reshape(-1,1)  #to convert into 2d array
print('x=',x)
Scores = dataset.Scores
y=np.array(Scores)
print('y=',y)

from sklearn.linear_model import LinearRegression
#creating linear regression model
model=LinearRegression()
model.fit(x,y)

#Finding Coefficient of Determination (R^2)
r_sq = model.score(x, y)
print('Coefficient of determination:', r_sq)

#Finding b0 and b1:
print('intercept (b0) :', model.intercept_)
print('slope (b1) :', model.coef_)

'''
Predicting Response
#the aim is to predict the score if a student studies for 9.25 hrs per day
'''

predicted_y = model.predict(x)
print('Predicted Values:')
print(predicted_y)

#Creating a table for better understanding:
import pandas as pd
new_dataset = pd.DataFrame({'Hours':dataset.Hours,'Scores':y,'Predicted scores':predicted_y})
print(new_dataset)

#Predicting score for 9.25 hours of study
x_new=np.array(9.25).reshape(-1,1)
print('Hours:',x_new)
y_new = model.predict(x_new)
print('Predicted Score:',y_new)
