# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values. 

## Program:
```Python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HASNA MUBARAK AZEEM
RegisterNumber: 212223240052
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Testing Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print("MSE =",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE =",mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
```

## Output:
HEAD:

![image](https://github.com/user-attachments/assets/e2c14fcb-2a48-4afc-80dc-e0be0ada95a1)

TAIL:

![image](https://github.com/user-attachments/assets/735a0216-9fae-4bf7-af8a-efee4d702bb8)

Array Value of X:

![image](https://github.com/user-attachments/assets/ec713756-9555-4cae-a9ac-acc829ad6543)

Array Value of Y:

![image](https://github.com/user-attachments/assets/68c03a60-8b6b-4900-9d8b-4c2b046c0c90)

Y Prediction:

![image](https://github.com/user-attachments/assets/a664c65c-0ce9-4517-9628-74eb94ee0eb0)

Array Value of Y Test:

![image](https://github.com/user-attachments/assets/22b23da7-5a7b-43f7-bbd3-a8de85bb1bf5)

Graph of Training Set:

![image](https://github.com/user-attachments/assets/5801173e-f309-46a0-9b4e-2f671340ba26)

Graph of Testing Set:

![image](https://github.com/user-attachments/assets/b06d93d4-c5ab-4ac7-b2d2-dc21418c5e69)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
