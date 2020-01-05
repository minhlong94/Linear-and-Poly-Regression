import csv
import matplotlib.pyplot as plt
import numpy as np
import sympy as s
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def bubbleSort(arrX, arrY):
    n = len(arrX)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arrX[j] > arrX[j + 1]:
                arrX[j], arrX[j + 1] = arrX[j + 1], arrX[j]
                arrY[j], arrY[j + 1] = arrY[j + 1], arrY[j]


def linearReg(x, y):
    n = len(x)
    totalXY = x[0] + y[0]
    totalX = x[0]
    totalY = y[0]
    totalXsqrt = x[0] ** 2
    for i in range(1, n):
        totalXY += x[i] * y[i]
        totalX += x[i]
        totalY += y[i]
        totalXsqrt += x[i] ** 2
    a1 = (n * totalXY - totalX * totalY) / (n * totalXsqrt - totalX ** 2)
    a0 = totalY / n - a1 * totalX / n
    e = 0
    Sr = 0
    St = 0
    yMean = totalY / n
    for i in range(n):
        e += abs((y[i] - a0 - a1 * x[i]))
        Sr += (y[i] - a0 - a1 * x[i]) ** 2
        St += (y[i] - yMean) ** 2
    rSqrt = (St - Sr) / St
    plt.plot(x, y, 'o')
    result = []
    for i in range(n):
        result.append(a0 + a1 * x[i])
    for i in range(n):
        plt.plot(x, result, color='red')
    print("a1 = {}, a0 = {}, rSquare = {}".format(a1, a0, rSqrt))
    plt.show()


def polyReg(xData, yData, order):
    n = len(xData)
    matrixX = []
    matrixY = []
    for j in range(n):
        row = []
        for i in range(order):
            row.append(xData[j] ** i)
        matrixY.append([yData[j]])
        matrixX.append(row)
    matrixX = np.array(matrixX)
    matrixY = np.array(matrixY)
    XtX = np.dot(matrixX.transpose(), matrixX)
    XtY = np.dot(matrixX.transpose(), matrixY)
    matrixA = np.dot(np.linalg.inv(XtX), XtY)

    x = s.symbols('x')
    func = ""
    for i in range(order):
        func += "+ {}*x**{}".format(matrixA[i, 0], i)
    f = s.sympify(func)
    print("{}-th order function is: {}".format(order - 1, f))
    result = []
    e = 0
    Sr = 0
    St = 0
    totalY = 0
    for i in range(len(yData)):
        totalY += yData[i]
    yMean = totalY / len(yData)
    for i in range(len(xData)):
        result.append(f.subs(x, xData[i]))
        e += abs(yData[i] - f.subs(x, xData[i]))
        Sr += (yData[i] - f.subs(x, xData[i])) ** 2
        St += (yData[i] - yMean) ** 2
    rSquare = (St - Sr) / St
    print("r-Square of {}-th poly order reg: {}".format(order - 1, rSquare))
    plt.plot(xData, yData, 'o')
    plt.plot(xData, result, color='red')
    plt.show()
    if rSquare < 0.8:
        polyReg(xData, yData, order + 1)


# file = open('polydata.csv')
# file = open('one_variable_data.csv')
file = open('testPoly.csv')
csv_f = csv.reader(file)
xData = []
yData = []
for row in csv_f:
    xData.append(row[0])
    yData.append(row[1])
xData = [float(c) for c in xData]
yData = [float(c) for c in yData]
bubbleSort(xData, yData)

linearReg(xData, yData)
polyReg(xData, yData, 1)

# np.random.seed(0)
# x = np.random.normal()
# y = -1.524 + 3.4519*x - 1.6818*x**2 - 1.7581*(x ** 3) + 2.3*(x ** 4) + np.random.normal()
# x = x[:, np.newaxis]
# y = y[:, np.newaxis]
# polyFeature = PolynomialFeatures(degree=4)
# x_poly = polyFeature.fit_transform(x)
#
# model = LinearRegression()
# model.fit(x_poly, y)
# y_poly_pred = model.predict(x_poly)
#
# rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
# r2 = r2_score(y, y_poly_pred)
# print(rmse)
# print(r2)
#
# plt.scatter(x, y, s=10)
# sort_axis = operator.itemgetter(0)
# sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
# x, y_poly_pred = zip(*sorted_zip)
# plt.plot(x, y_poly_pred, color='m')
# plt.show()
