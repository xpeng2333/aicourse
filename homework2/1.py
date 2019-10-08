from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    print(model.intercept_, model.coef_)
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)


# np.random.seed(20)
x = np.random.rand(200, 1)
y = 5+1.2*x-3.4*x**2+5.6*x**3 + np.random.normal(scale=0.1, size=(200, 1))

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=10)


poly_features = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly_features.fit_transform(X_train)

sgd_reg = SGDRegressor(eta0=0.0001, learning_rate='constant', max_iter=50000, tol=-
                       np.infty, penalty='none', random_state=10, fit_intercept=True)
plot_learning_curves(sgd_reg, x_poly, y_train.ravel())
#print(sgd_reg.intercept_, sgd_reg.coef_)
plt.axis([0, 80, 0, 2])
plt.show()


sgd_reg_lin = SGDRegressor(tol=-
                           np.infty, penalty='none', random_state=10, fit_intercept=True)
plot_learning_curves(sgd_reg_lin, X_train, y_train.ravel())
plt.axis([0, 80, 0, 2])
plt.show()


poly_features10 = PolynomialFeatures(degree=10, include_bias=False)
x_poly10 = poly_features10.fit_transform(X_train)
sgd_reg10 = SGDRegressor(tol=-
                         np.infty, penalty='none', random_state=10, fit_intercept=True)
plot_learning_curves(sgd_reg10, x_poly10, y_train.ravel())
plt.axis([0, 80, 0, 2])
plt.show()

sgd_reg10_l2 = SGDRegressor(eta0=0.01, learning_rate='constant', max_iter=500, tol=-
                            np.infty, penalty='l2', alpha=0.00001, random_state=10, fit_intercept=True)
plot_learning_curves(sgd_reg10_l2, x_poly10, y_train.ravel())
plt.axis([0, 80, 0, 2])
plt.show()
