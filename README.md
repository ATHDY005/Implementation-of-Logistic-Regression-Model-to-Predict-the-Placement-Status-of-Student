# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset

2.Check for null and duplicate values

3.Assign x and y values

4.Split the data into training and testing data

5.Import logistic regression and fit the training data

6.Predict y value

7.Calculate accuracy and confusion matrix
 

## Program:
```
/*
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.Harini Shamlin
RegisterNumber:212220040040

import pandas as pd

data=pd.read_csv('/content/Placement_Data.csv')

data.head()

data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis=1)

data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])

data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"])

data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"])

data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])

data1["status"]=le.fit_transform(data1["status"])

data1

x=data1.iloc[:,:-1]

x

y=data1["status"]

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver="liblinear")

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix

confusion=(y_test,y_pred)

confusion

from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred)

print(cr)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/ 
*/
```

## Output:
# PLACEMENT DATA:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/a3ea7f5f-2cfd-4c6c-8e41-216206c4b42d)
# SALARY DATA:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/121dca44-7582-4c68-8191-7c378a4ef00d)
# CHECKING NULL VALUE:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/ae416f8f-793e-404d-ab0c-fa7745b88f6a)
# DATA DUPLICATE:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/b2274951-9842-47d8-ad4a-8bd943cdea7f)
# PRINT DATA:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/6deab240-0176-48e6-8881-d67e65236209)
# Y_PREDICTED ARRAY:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/5526b1fe-dc10-4c06-a768-cb8405f4b047)
# CONFUSION ARRAY:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/97313a62-1455-45db-8bd3-998834140245)
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/403a375c-6abd-4157-a011-0dd263675405)
# CLASSIFICATION REPORT:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/7fe4450f-178a-4d49-ad86-c8010f5f72ac)
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/7c0f706e-cace-4b48-b78b-192f326f8826)
# PREDICTION OF LR:
![image](https://github.com/ATHDY005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/84709944/850378b6-d707-4df9-b0b6-dd13188657de)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
