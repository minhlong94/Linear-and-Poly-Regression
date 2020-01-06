import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def func(x, y):
    return -1.66337046 + -1.65295871*x + 3.57249139*y


def bubbleSort(arrX, arrY):
    n = len(arrX[0])
    for i in range(n):
        for j in range(0, n - i - 1):
            if arrX[0][j] > arrX[0][j + 1]:
                for k in range(len(arrX)):
                    arrX[k][j], arrX[k][j + 1] = arrX[k][j + 1], arrX[k][j]
                arrY[j], arrY[j + 1] = arrY[j + 1], arrY[j]


def linearRegND(xData, yData):
    n = len(xData)
    matrixX = []
    matrixY = []
    totalY = 0
    for i in range(len(yData)):
        totalY += yData[i]
    yMean = totalY/len(yData)
    for i in range(n):
        row = [1]
        for j in range(len(xData[i])):
            row.append(xData[i][j])
        matrixY.append(yData[i])
        row = np.array(row)
        matrixX.append(row)
    matrixX = np.array(matrixX)
    matrixY = np.array(matrixY)
    XtX = np.dot(matrixX.transpose(), matrixX)
    XtY = np.dot(matrixX.transpose(), matrixY)
    matrixA = np.dot(np.linalg.inv(XtX), XtY)
    e, Sr, St = 0, 0, 0
    result = []
    for i in range(len(xData)):
        sum = 0
        for j in range(len(matrixA)):
            sum += matrixA[j]*matrixX[i][j]
        result.append(sum)
        e += abs(yData[i] - sum)
        Sr += (yData[i] - sum) ** 2
        St += (yData[i] - yMean)**2
    rSquare = (St - Sr)/St
    print(rSquare)
    print(matrixA)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1Data = []
    x2Data = []
    for i in range(len(xData)):
        x1Data.append(xData[i][0])
        x2Data.append(xData[i][1])
    ax.scatter(x1Data, x2Data, yData, label='Actual', color='red', marker='^')
    ax.scatter(x1Data, x2Data, result, label='Predicted', color='blue')
    # X, Y = np.meshgrid(x1Data, x2Data)
    # Z = func(X, Y)
    # Z = Z.reshape(X.shape)
    # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.legend()
    plt.show()
    print("Two - variable function is: f = {} + {}*x + {}*y".format(matrixA[0], matrixA[1], matrixA[2]))


file = open('multiLinear.csv')
csv_f = csv.reader(file)
xData, yData = [], []
for row in csv_f:
    eachRow = []
    for i in range(len(row)-1):
        eachRow.append(row[i])
    eachRow = [float(c) for c in eachRow]
    xData.append(eachRow)
    yData.append(row[len(row)-1])

yData = [float(c) for c in yData]
# print(xData)
# print(yData)
bubbleSort(xData, yData)
linearRegND(xData, yData)