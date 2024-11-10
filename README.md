# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Certainly! Here’s a step-by-step breakdown of the algorithm for the code provided:

### Algorithm for SMS Spam Detection

1. **Import necessary libraries**
   - Import `chardet` to detect file encoding.
   - Import `pandas` for data manipulation.
   - Import `train_test_split` from `sklearn.model_selection` for splitting the dataset.
   - Import `CountVectorizer` from `sklearn.feature_extraction.text` for text vectorization.
   - Import `SVC` from `sklearn.svm` for the Support Vector Machine (SVM) classifier.
   - Import `metrics` from `sklearn` to evaluate the model.

2. **Detect file encoding using `chardet`**
   - Open the file in binary mode and read a portion of the file to detect its encoding. Here, the code reads the first 100,000 bytes of the file.
   - Use the detected encoding to read the file with `pandas`.

3. **Load dataset**
   - Read the file into a DataFrame using the detected encoding (`Windows-1252`).
   - Display the first few rows to understand the data structure.
   - Check for null values and dataset information.

4. **Separate input and output features**
   - Assign the `v1` column (containing labels like "ham" and "spam") to `x`.
   - Assign the `v2` column (containing the actual messages) to `y`.

5. **Split the dataset into training and testing sets**
   - Use `train_test_split` to divide `x` and `y` into training and testing sets.
   - Set `test_size` to 0.2, meaning 20% of the data will be used for testing, while the rest is for training.

6. **Convert text data to numerical format**
   - Initialize `CountVectorizer` as `cv`.
   - Transform `x_train` into a matrix of token counts by fitting and transforming `x_train`.
   - Apply the same transformation to `x_test`.

7. **Train the SVM classifier**
   - Initialize an SVM classifier (`SVC`) and fit it on the training data (`x_train` and `y_train`).

8. **Predict and evaluate the model**
   - Predict the classes for `x_test` using the trained SVM model.
   - Calculate the accuracy of predictions compared to `y_test`.

This algorithm primarily involves loading data, preparing it for machine learning, training an SVM classifier, and evaluating the model's performance.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANJAI.R
RegisterNumber:  212223040180
*/
```

```py
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result output:

![Screenshot 2024-11-10 141800](https://github.com/user-attachments/assets/fae18f17-291a-40aa-9bdb-8a40b80e37bf)

data.head():
![Screenshot 2024-11-10 141805](https://github.com/user-attachments/assets/1998b001-4409-4292-b6e6-16b4973cb551)

data.info():
![Screenshot 2024-11-10 141812](https://github.com/user-attachments/assets/ad64b4a3-f10b-401b-8de7-84a6742456f1)

data.isnull().sum():
![Screenshot 2024-11-10 141816](https://github.com/user-attachments/assets/ee51e428-837a-49c6-93ae-25de46c53840)

Y_prediction value:
![Screenshot 2024-11-10 141819](https://github.com/user-attachments/assets/e3585319-ef0a-4f9e-9304-5581a47dde1d)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
