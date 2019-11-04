import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pylab as pyl
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# 对title提取
import re

train = pd.read_csv("./homeworkMid/data/train.csv")
test = pd.read_csv("./homeworkMid/data/test.csv")


def get_title(name):
    title = re.search('([A-Za-z]+)\.', name)
    if title:
        return (title.group(1))
    return ('')


train['Title'] = train['Name'].apply(get_title)

# print(pd.crosstab(train['Title'],train['Sex']))
train['Title'] = train['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
    'Jonkheer', 'Dona'
], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
# print(train[['Title','Survived']].groupby(['Title'],as_index=False).mean())
test['Title'] = test['Name'].apply(get_title)
# print(pd.crosstab(test['Title'],test['Sex']))
test['Title'] = test['Title'].replace([
    'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
    'Jonkheer', 'Dona'
], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

train['familysize'] = train['Parch'] + train['SibSp'] + 1
train['alone'] = 0
train.loc[train['familysize'] == 1, 'alone'] = 1
# 用s对缺失处进行填充 fillna 填充函数
train['Embarked'] = train['Embarked'].fillna('S')
train['Cabin'] = train['Cabin'].fillna('no')
age_df = train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_true = age_df.loc[(age_df.Age.notnull())]
age_df_null = age_df.loc[(age_df.Age.isnull())]
X = age_df_true.values[:, 1:]
y = age_df_true.values[:, 0]
rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rfr.fit(X, y)
preAge = rfr.predict(age_df_null.values[:, 1:])
train.loc[train.Age.isnull(), 'Age'] = preAge
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train_set = train.drop(drop_elements, axis=1)

# 对测试样本进行数据处理
test['familysize'] = test['Parch'] + test['SibSp'] + 1
test['alone'] = 0
test.loc[test['familysize'] == 1, 'alone'] = 1
test['Embarked'] = test['Embarked'].fillna('S')
test['Cabin'] = test['Cabin'].fillna('no')
age_df = test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_null = age_df.loc[(age_df.Age.isnull())]
preAge = rfr.predict(age_df_null.values[:, 1:])
test.loc[test.Age.isnull(), 'Age'] = preAge
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
test_set = test.drop(drop_elements, axis=1)

dummies_Embarked = pd.get_dummies(train_set['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(train_set['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(train_set['Pclass'], prefix='Pclass')
dummies_Title = pd.get_dummies(train_set['Title'], prefix='Title')
df = pd.concat([train_set, dummies_Embarked, dummies_Sex, dummies_Title],
               axis=1)
df.drop(['Sex', 'Embarked', 'Title'], axis=1, inplace=True)
train_np = df.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
dummies_Embarked = pd.get_dummies(test_set['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(test_set['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(test_set['Pclass'], prefix='Pclass')
dummies_Title = pd.get_dummies(test_set['Title'], prefix='Title')
bf = pd.concat([test_set, dummies_Embarked, dummies_Sex, dummies_Title],
               axis=1)
bf.drop(['Sex', 'Embarked', 'Title'], axis=1, inplace=True)
bf = bf.fillna(50)
test_np = bf.as_matrix()

predictions = clf.predict(test_np)
result = pd.DataFrame({
    'PassengerId': test['PassengerId'].as_matrix(),
    'Survived': predictions.astype(np.int32)
})
result.to_csv('feature_predictions.csv', index=False)

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(probability=True),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    xgb.XGBClassifier()
]

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
x = train_np[:, 1:]
y = train_np[:, 0]
accuracy = np.zeros(len(classifiers))
for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_num = 0
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        clf.fit(x_train, y_train)
        accuracy[clf_num] += (y_test == clf.predict(x_test)).mean()
        clf_num += 1
accuracy = accuracy / 10
plt.bar(np.arange(len(classifiers)), accuracy, width=0.5, color='b')
plt.xlabel('Alog')
plt.ylabel('Accuracy')
plt.xticks(
    np.arange(len(classifiers)) + 0.25,
    ('KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB', 'LDA', 'QDA', 'LR',
     'xgb'))

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
x = train_np[:, 1:]
y = train_np[:, 0]
x1_test = np.zeros((test.shape[0], len(classifiers)))
accuracy = np.zeros(len(classifiers))
for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_num = 0
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        clf.fit(x_train, y_train)
        x1_test[:, clf_num] += clf.predict(test_np)
        accuracy[clf_num] += (y_test == clf.predict(x_test)).mean()
        clf_num += 1
accuracy = accuracy / 10
x1_test = x1_test / 10
plt.bar(np.arange(len(classifiers)), accuracy, width=0.5, color='b')
plt.xlabel('Alog')
plt.ylabel('Accuracy')
plt.xticks(
    np.arange(len(classifiers)) + 0.25,
    ('KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB', 'LDA', 'QDA', 'LR',
     'xgb'))

pyl.pcolor(np.corrcoef(x1_test.T), cmap='Blues')
pyl.colorbar()
pyl.xticks(np.arange(0.5, 11.5), [
    'KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB', 'LDA', 'QDA', 'LR', 'xgb'
])

pyl.yticks(np.arange(0.5, 11.5), [
    'KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB', 'LDA', 'QDA', 'LR', 'xgb'
])

pyl.show

index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
linear_prediction = x1_test[:, index].mean(axis=1)
linear_prediction[linear_prediction >= 0.5] = 1
linear_prediction[linear_prediction < 0.5] = 0
mixRe = pd.DataFrame({
    'PassengerId': test['PassengerId'].as_matrix(),
    'Survived': linear_prediction.astype(np.int32)
})

mixRe.to_csv('mix2.csv', index=False)
