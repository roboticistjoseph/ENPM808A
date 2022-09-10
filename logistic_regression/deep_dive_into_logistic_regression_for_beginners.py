
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# File system manangement
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

"""## Read in the data"""

# List files available
print(os.listdir("./"))

# Training Data

train = pd.read_csv('train.csv')
print('Training data shape: ', train.shape)
train.head()

"""The training data has 891 observations and 12 features (variables) including the TARGET (the label we want to predict).In this case we want to predict whether a passenger on Titanic **survived** or not."""

# Testing data features
test = pd.read_csv('test.csv')
print('Testing data shape: ', test.shape)
test.head()

"""## Exploratory Data Analysis(EDA)

The data page on Kaggle describes the columns in detail. It’s always worth exploring this in detail to get a full understanding of the data.

### Examining the Distribution of the Target Column
"""

train['Survived'].value_counts()

sns.countplot(x = 'Survived',data = train)

"""Thus, around 549 people perished while 342 survived."""

sns.countplot(x = 'Survived',hue = 'Sex',data = train)

"""We can see that females survived in much higher proportions than males did. Now, Let’s see how many people survived divided by class."""

sns.countplot(x = 'Survived',hue = 'Pclass',data = train)

"""Distribution of survival rate class wise"""

sns.boxplot(x='Pclass',y='Age',data=train)

"""## Examining Missing Values

Next we can look at the number and percentage of missing values in each column.
"""

print("Null in Training set")
print("---------------------")
print(train.isnull().sum())
print("---------------------")
print("Null in Testing set")
print("---------------------")
print(test.isnull().sum())

"""The three columns i.e Age, cabin and Embarked have missing values which needs to be taken care of.

#### 1. Age Column

Let’s create a function to impute ages regarding the corresponding age average per class.
"""

def add_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(train[train["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(add_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(add_age,axis=1)

"""#### 2. Missing values in Cabin 

Since we have lots of null values for Cabin column, so it is better to remove it.
"""

train.drop("Cabin",inplace=True,axis=1)
test.drop("Cabin",inplace=True,axis=1)

"""#### 3. Missing values in Embarked column

Since there are just two missing values, we shall impute them with the mode of the Embarked column.
"""

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)

"""#### 3. Missing values in Frame column in Test Dataset

Since there is one missing value, we shall impute them with the mean of the Fare column.
"""

test['Fare'].fillna(test['Fare'].mean(),inplace=True)

"""## Creating new Features

* WE shall create a new column called **Family** by combining Parch and SibSp columns

"""

def combine(df,col1,col2):
    df["Family"] = df[col1]+df[col2]
    df.drop([col1,col2],inplace=True,axis=1)
    return df

train = combine(train,'SibSp','Parch')
test = combine(test,'SibSp','Parch')

"""Let’s take a look at the Age column"""

train['Age'].describe()

"""the Age column needs to be treated slightly differently, as this is a continuous numerical column.we can separate this continuous feature into a categorical feature by dividing it into ranges."""

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar()

"""## Encoding Categorical Variables

We can use the pandas.get_dummies() function Now, we shall have to encode Sex, Embarked, Pclass and Age_categories. Name and Ticket columns have a lot of categories, hence we shall delete them.
"""

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex","Age_categories",'Embarked']:
    train = create_dummies(train,column)
    test = create_dummies(test,column)

"""## Dropping Unnecessary columns"""

train.drop(['Name','Sex','Ticket','Pclass','Age_categories','Embarked'],inplace=True,axis=1)
test.drop(['Name','Sex','Ticket','Pclass','Age_categories','Embarked'],inplace=True,axis=1)

"""## Logistic Regression Implementation

We will use Logistic Regressionfrom [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model. The only change we will make from the default model settings is to lower the [regularization parameter](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), C, which controls the amount of overfitting (a lower value should decrease overfitting). This will get us slightly better results than the default Logistic Regression.

The .fit() method accepts two arguments: X and y. X must be a two dimensional array (like a dataframe) of the features that we wish to train our model on, and y must be a one-dimensional array (like a series) of our target, or the column we wish to predict.
"""

lr = LogisticRegression()
columns = ['PassengerId', 'Age', 'Fare','Family',
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

lr.fit(train[columns], train["Survived"])

"""### Evaluating Accuracy of our model

The evaluation criteria given on the Titanic Data page is accuracy, i.e how many correct predictions we have made out of the total predictions. We have created our model but how will we know how accurate it is? We do have a Test dataset but since it doesn't have the Target column, everytime we optimize our model, we will have to submit our predictions to public Leaderboard to assess it accuracy. 

#### Creating a Validation set

Another option would be to create a validation set from the training set. We will hold out a part of the training set during the start of the experiment and use it for evaluating our predictions. We shall use the scikit-learn library's `model_selection.train_test_split()` function that we can use to split our data
"""

X = train[columns]
y = train['Survived']

train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.20,random_state=0)

"""#### Making predictions and measuring accuracy"""

lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(val_X)
accuracy = accuracy_score(val_y, predictions)
print(accuracy)
from sklearn.metrics import classification_report
print(classification_report(val_y,predictions))

"""#### Using cross validation for more robust error measurement

Using a Validation dataset has a drawback. Firstly, it decreases the training data and secondly since it is tested against a small amount of data, it has high chances of overfitting. To overcome this, there is a technique called **[cross validation](https://scikit-learn.org/stable/modules/cross_validation.html)**. The most common form of cross validation, and the one we will be using, is called k-fold cross validation. ‘Fold’ refers to each different iteration that we train our model on, and ‘k’ just refers to the number of folds. In the diagram above, we have illustrated k-fold validation where k is 5.

![](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

[source](https://scikit-learn.org/stable/modules/cross_validation.html)

"""

lr = LogisticRegression()
scores = cross_val_score(lr, X, y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)

"""#### Making Predictions on Test data"""

lr = LogisticRegression()
lr.fit(X,y)
predictions_test = lr.predict(test[columns])

"""## Submission """

submission = pd.read_csv('gender_submission.csv')
submission_df = pd.DataFrame({'PassengerId' : test['PassengerId'],
                              'Survived':predictions_test})
submission_df.head()

submission_df.to_csv("submission.csv",index=False)


