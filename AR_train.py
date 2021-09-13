import statsmodels.api as sm
from patsy import dmatrices
import pandas
import numpy as np
import matplotlib.pyplot as plt
from dataAnalysis_func import *
from sklearn.linear_model import LinearRegression


# contrived dataset
data = np.array([4, 7, 2, 10, 21, 26, 11, 29, 83, 22, 92, 46, 119, 73, 97, 149,
                    148, 154, 155, 68, 143, 206, 224, 311, 308, 266, 325, 270, 392, 397, 501, 444, 343, 261, 415,
                    467, 578, 477, 478, 492, 392, 401, 456, 540, 455, 550, 502, 418, 366, 487, 507, 504, 515, 522,
                    416, 375, 402, 422, 483, 528, 433, 325, 260, 354, 476, 442, 432, 406, 259, 339, 321, 477,
                    429, 393, 468, 340, 328, 482, 589, 553, 550, 485, 463, 394, 525, 689, 683, 753, 648, 656,
                    666, 758, 829, 921, 841, 735, 681, 833, 940, 994, 1109, 948, 917, 646, 706, 664, 889, 876,
                    914, 823, 543, 564, 807, 810, 819, 800, 678, 612, 638, 836, 848, 809, 847, 731, 651, 673,
                    829, 856, 972, 1106, 920, 807, 919, 1022, 1197, 1090, 1172, 1112, 990, 1061, 1271, 1318,
                    1453, 1489, 1199, 1008, 1158, 1433, 1592, 1732, 1847, 1637, 1464, 1616, 1967, 2134, 2106,
                    2328, 1987, 1799, 1658, 1670, 1974, 2438, 2481, 2096, 2141, 2088, 2495, 2430, 2723, 2836,
                    2107, 2174, 2411, 2551, 2582, 3144, 3103, 2476, 2462, 2905], dtype=float)

# fit model
lags=7
predictions, predictionLevel, intercept, coefs = ArPrediction(data, lags)
print(f'prediction level is {predictionLevel}')
plt.plot(data, label='data')
plt.plot(predictions, label='predictions')
plt.title(f'prediction level is {round(predictionLevel, 2)}')
plt.legend()
plt.grid()
plt.show()
exit()
if False:

    from statsmodels.tsa.ar_model import AutoReg

    model = AutoReg(data, lags=lags, missing='drop')
    model_fit = model.fit()
    print(f'model params: {model_fit.params}')

    #indexes = np.random.randint(0, data.shape[0]-49, 3)
    #data[indexes]=np.nan

    '''
    # fit model
    model = AutoReg(data, lags=7, missing='drop')
    model_fit = model.fit()
    print(f'model params: {model_fit.params}')
    
    # fit model
    
    model = AutoReg(data[np.logical_not(np.isnan(data))], lags=7, missing='drop')
    model_fit = model.fit()
    print(f'model params: {model_fit.params}')
    '''
    # let's make prediction
    y = model_fit.predict(len(data), len(data) + 21)
    #print(y)
    myUnderstandingDiff = np.dot(np.flip(data[-7:]), model_fit.params[1:])+model_fit.params[0] - y[0]
    print(f'My understanding diff is {myUnderstandingDiff}')

    print(model_fit.summary())

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(y) + 1), y, 'b')

    plt.figure(figsize=(10,5))
    prognose = np.concatenate((data,y))
    plt.plot(range(1,len(prognose)+1), prognose, 'r')
    plt.plot(range(1,len(data)+1), data, 'b')


    y = model_fit.predict(lags, len(data)-1)

    myUnderstandingDiff = np.dot(np.flip(data[:lags]), model_fit.params[1:])+model_fit.params[0] - y[0]
    print(f'My understanding diff is {myUnderstandingDiff}')

    plt.figure(figsize=(10,5))
    plt.plot(range(lags,lags+len(y)), data[lags:], 'b')
    plt.plot(range(lags,lags+len(y)), y, 'r')
    # sklearn linear regression

    #indexes = np.random.randint(0, data.shape[0]-49, 3)
    #data[indexes] = np.nan

    X = np.zeros((len(data)-lags, lags))
    target = data[lags:]
    for r in range(lags):
        X[:, r] = data[r:r+X.shape[0]]

    reg = LinearRegression().fit(X, target)
    ysklearn = reg.predict(X)
    plt.plot(range(lags,lags+len(ysklearn)), ysklearn, 'g')

    intercept = reg.intercept_
    params = reg.coef_[:, None]

    tildeX = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(tildeX), tildeX)), np.transpose(tildeX)), target)[:, None]

    alfa = beta[0]
    beta = beta[1:]

    y_myUnderstanding = (alfa + np.matmul(X, beta))[:, 0]
    print(f'my understanding of ysklearn: {np.abs(y_myUnderstanding - ysklearn).max()}')
    predictionLevel = reg.score(X, target)

    mse_h = 1/len(target) * np.power(y_myUnderstanding - target, 2).sum()
    mse_0 = 1/len(target) * np.power(target - target.mean(), 2).sum()
    my_predictionLevel = 1 - mse_h/mse_0

    print(f'my understanding of prediction level {my_predictionLevel-predictionLevel}')

    modelError = data[lags:] - y
    plt.figure(figsize=(10,5))
    plt.plot(range(lags,lags+len(y)), modelError, 'b')


    MSE = np.power(modelError, 2).mean()
    median = np.median(np.power(modelError, 2))
    print(f'mean-square-error is {MSE}, median-square-error {median}')


    plt.show()