import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
print(X_train[:10])
print(y_train[:10])
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test.reshape(-1, 1))
regressor = SGDRegressor(loss='squared_loss', penalty="l1")
scores = cross_val_score(regressor, X_train, y_train.reshape(-1, 1), cv=5)
print('cv R', scores)
print('mean of cv R', np.mean(scores))
regressor.fit(X_train, y_train)
print('Test set R', regressor.score(X_test, y_test))
