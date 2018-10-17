# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd 
import numpy  as np 
import scipy  as sc
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


print(chr(27) + "[2J")

# Import train/test
dataTrain = pd.read_csv(            "train.csv")
dataTest  = pd.read_csv(             "test.csv")
dataTTest = pd.read_csv("sample_submission.csv")
dataTest  = pd.concat([ dataTest, dataTTest ],axis = 1)

# Name of columns
tr = ['SalePrice']
ca = ['MSSubClass',
      'MSZoning',
      'Street',
      'Alley',
      'LotShape',
      'LandContour',
      'Utilities',
      'LotConfig',
      'LandSlope',
      'Neighborhood',
      'Condition1',
      'Condition2',
      'BldgType',
      'HouseStyle',
      'OverallQual',
      'OverallCond',
      'RoofStyle',
      'RoofMatl',
      'Exterior1st',
      'Exterior2nd',
      'MasVnrType',
      'ExterQual',
      'ExterCond',
      'Foundation',
      'BsmtQual',
      'BsmtCond',
      'BsmtExposure',
      'BsmtFinType1',
      'BsmtFinType2',
      'Heating',
      'HeatingQC',
      'CentralAir',
      'Electrical',
      'BsmtFullBath',
      'BsmtHalfBath',
      'FullBath',
      'HalfBath',
      'BedroomAbvGr',
      'YrSold',
      'SaleType',
      'SaleCondition']
nu = ['LotFrontage',
      'LotArea',
      'OverallQual',
      'OverallCond',
      'YearBuilt',
      'YearRemodAdd',
      'MasVnrArea',
      'BsmtFinSF1',
      'BsmtFinSF2', 
      'BsmtUnfSF',
      'TotalBsmtSF',
      '1stFlrSF',
      '2ndFlrSF',
      'LowQualFinSF',
      'GrLivArea',
      'BsmtFullBath',
      'BsmtHalfBath',
      'FullBath',
      'HalfBath',
      'BedroomAbvGr',
      'KitchenAbvGr',
      'TotRmsAbvGrd',
      'Fireplaces',
      'GarageYrBlt',
      'GarageCars',
      'GarageArea',
      'WoodDeckSF',
      'OpenPorchSF',
      'EnclosedPorch',
      '3SsnPorch',
      'ScreenPorch',
      'PoolArea',
      'MiscVal',
      'MoSold',
      'YrSold',
      'SalePrice']

# Pearson correlation
pearson  = dataTrain[nu].corr('pearson' )
spearman = dataTrain[nu].corr('spearman')
pearson  =  pearson[tr]
spearman = spearman[tr]

# Anova one-way


# Data selection
umb = 0.5
select_pearson  = pearson [ pearson > umb].dropna().abs().sort_values(tr).index.tolist()
select_spearman = spearman[spearman > umb].dropna().abs().sort_values(tr).index.tolist()
select_nu = [x for x in select_spearman if x in select_pearson]

df = dataTrain[ select_nu ]

# Preprocessing
df = (df-df.mean())/df.std()

# Plot
#sns.pairplot(df, x_vars=select_nu[:-1], 
#                 y_vars=select_nu[ -1], 
#                 size=7, aspect=0.7, kind='reg')

# Cross-validation
sdf = df.dropna()
x = sdf[ select_nu[:-1] ].values
y = sdf[ select_nu[ -1] ].values.reshape(-1, 1)
kf = KFold(n_splits=8)


"""
Python implementation
---------------------
"""
MAE = list(); MSE = list(); RMSE = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = LinearRegression()
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    
print('Python implementation:')
print(   'MAE=',np.average(MAE),
       '\tMSE=',np.average(MSE),
      '\tRMSE=',np.average(RMSE))
print('\n')    


"""
Stochastic Gradient Descent
---------------------------
"""
def StochasticGradientDescent(x_train,y_train,x_test):
    
    # Parameters
    alpha     = 0.001
    err       = 1000
    errNorm   = 1000
    threshold = 0.001
    
    theta = np.zeros( len(x_train[0]) )
    
    # Train Loop
    while (errNorm>threshold):
        
        exErr = err
        err   = 0
        for xs, ys in zip(x,y):
            y_pred = theta * xs
            
            #band   = alpha * (ys - y_pred)
            #for i in range( len(theta) ):
            #    theta[i]  = theta[i] + band[i] * xs[i]
            
            theta  = theta + alpha * (ys - y_pred) * xs
            
            err = err + np.sum(np.abs(ys - y_pred))
            
        # Update error
        errNorm = np.abs(exErr - err)/np.abs(err)
        
    return np.dot(x_test,theta)


MAE = list(); MSE = list(); RMSE = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Stochastic Gradient Descent
    y_pred = StochasticGradientDescent(x_train,y_train,
                                       x_test)
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    
print('Stochastic Gradient Descent:')
print(   'MAE=',np.average(MAE),
       '\tMSE=',np.average(MSE),
      '\tRMSE=',np.average(RMSE))
print('\n')    



