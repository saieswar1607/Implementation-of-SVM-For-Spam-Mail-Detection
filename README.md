# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sai Eswar Kandukuri
RegisterNumber:  212221240020
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

<img width="479" alt="output1" src="https://user-images.githubusercontent.com/93427011/174759304-02ac6a7f-8825-4a04-b542-6ed9b718943b.png">
<img width="479" alt="output2" src="https://user-images.githubusercontent.com/93427011/174759318-8df12d8c-567b-4f5d-884b-8118c3bb61f7.png">
<img width="479" alt="output3" src="https://user-images.githubusercontent.com/93427011/174759330-77abe0b8-4a9d-4078-b228-f9526729c7a0.png">
<img width="621" alt="output4" src="https://user-images.githubusercontent.com/93427011/174759339-01aa4eab-38b6-4a35-89e5-6b2d2f120d58.png">
<img width="631" alt="output5" src="https://user-images.githubusercontent.com/93427011/174759354-2b1975c1-114f-4e46-8446-1273c9799abb.png">


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
