# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: sanjai R
RegisterNumber: 212223040180
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

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

from sklearn.metrics import confusion_matrix, classification_report
con = confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
### Head:

![Screenshot 2024-10-24 093945](https://github.com/user-attachments/assets/1450800c-5939-49d9-8c41-5d35170a194a)

### Info:

![Screenshot 2024-10-24 093952](https://github.com/user-attachments/assets/aca83c7e-fbea-4b69-906c-26c106359083)

### isnull:

![Screenshot 2024-10-24 093958](https://github.com/user-attachments/assets/e257672a-1107-4b26-aaa6-2308c2752b1a)

### Accuracy:

![Screenshot 2024-10-24 094003](https://github.com/user-attachments/assets/392de935-f6ec-4837-8cba-b4aed9cafbf2)

### Confusion matrix and Classification Report:

![Screenshot 2024-10-24 094010](https://github.com/user-attachments/assets/c5838393-ba9b-4571-bde2-57342a143b78)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
