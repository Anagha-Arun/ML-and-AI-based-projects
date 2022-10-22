#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#loading dataset
path=r"student_info.csv"
df=pd.read_csv(path)
#checking if data has been accurately loaded
print(df.head())
print(df.shape)
#exploring the data to gain insight
print(df.info())
print(df.describe())
#visualising the data to gain insight
plt.scatter(df.hours,df.marks)
plt.xlabel("No. of hours a student studies")
plt.ylabel("Marks obtained by student")
plt.title("Graphical representation of given data")
plt.show()
#preparing data that is to be provided to the ML model
print(df.isnull().sum())
df.mean()
#imputing the mean value to the null values present
df.mean()
df2=df.fillna(df.mean())
#checking if data has been cleaned
print(df2.isnull().sum())
#splitting the dataset into columns
X=df2.drop("marks",axis="columns")
y=df2.drop("hours",axis="columns")
#checking if dataset has been split
print("Shape of X=",X.shape)
print("Shape of  y=",y.shape)
#splitting data between training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=51)
#checking if data has been accurately split
print("Shape of X_train=",X_train)
print("Shape of X_test=",X_test)
print("Shape of y_train=",y_train)
print("Shape of y_test=",y_test)
#selecting a model and training it
#x and y are related as y=m*x+c
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#verifying if the linear regression model has been applied on data
print(lr.fit(X_train,y_train))
print(lr.coef_)
print(lr.intercept_)
#testing model
y_pred=lr.predict(X_test)
print(pd.DataFrame(np.c_[X_test,y_test,y_pred],columns=["study_hours","student_marks_original","student_marks_predicted"]))
#fine-tuning the model
print(lr.score(X_test,y_test))
plt.plot(X_train,lr.predict(X_train),color="r")
plt.show()
#saving the linear regression ML model
import joblib
joblib.dump(lr,"students_mark_predictor_model.pkl")
model=joblib.load("students_mark_predictor_model.pkl")
a=int(input("Enter the number of hours you have studied:"))
b=int(model.predict([[a]]))
c=str(b)
print("The marks you would get, having studied for the given number of hours is",b,"marks")

