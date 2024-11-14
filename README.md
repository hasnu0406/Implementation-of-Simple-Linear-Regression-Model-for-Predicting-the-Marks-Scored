# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, numpy, matplotlib, and scikit-learn.
2. Load the dataset student_scores.csv into a DataFrame and print it to verify contents.
3. Display the first and last few rows of the DataFrame to inspect the data structure.
4. Extract the independent variable (x) and dependent variable (y) as arrays from the DataFrame.
5. Split the data into training and testing sets, with one-third used for testing and a fixed random_state for reproducibility.
6. Create and train a linear regression model using the training data.
7. Make predictions on the test data and print both the predicted and actual values for comparison.
8. Plot the training data as a scatter plot and overlay the fitted regression line to visualize the model's fit.
9. Plot the test data as a scatter plot with the regression line to show model performance on unseen data.
10. Calculate and print error metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for evaluating model accuracy.
11. Display the plots to visually assess the regression results.

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

![image](https://github.com/user-attachments/assets/5e567898-263f-4f67-bcb2-8b7e37888b41)

Graph of Testing Set:

![image](https://github.com/user-attachments/assets/b06d93d4-c5ab-4ac7-b2d2-dc21418c5e69)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
