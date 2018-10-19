# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd 
import numpy  as np 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

print(chr(27) + "[2J")

# Import train/test
dataTrain = pd.read_csv("train.csv")
dataTest  = pd.read_csv( "test.csv")

# Nan to zero
dataTrain.fillna(0);
dataTest .fillna(0);

# Name of columns
tr = ['SalePrice']

catOrdinal = ['MSZoning','Street','Alley','LandContour',
              'Utilities','LotConfig','Neighborhood','Condition1','Condition2',
              'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
              'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',
              'Electrical','Functional','GarageType','MiscFeature','SaleType',
              'SaleCondition']

catNominal = {'LotShape':['Reg','IR1','IR2','IR3','NA'],
              'LandSlope':['Gtl','Mod','Sev','NA'],
              'ExterQual':['Ex','Gd','TA','Fa','Po','NA'],
              'ExterCond':['Ex','Gd','TA','Fa','Po','NA'],
              'BsmtQual':['Ex','Gd','TA','Fa','Po','NA'],
              'BsmtCond':['Ex','Gd','TA','Fa','Po','NA'],
              'BsmtExposure':['Gd','Av','Mn','No','NA'],
              'BsmtFinType1':['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'],
              'BsmtFinType2':['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'],
              'HeatingQC':['Ex','Gd','TA','Fa','Po','NA'],
              'KitchenQual':['Ex','Gd','TA','Fa','Po','NA'],
              'FireplaceQu':['Ex','Gd','TA','Fa','Po','NA'],
              'GarageQual':['Ex','Gd','TA','Fa','Po','NA'],
              'GarageCond':['Ex','Gd','TA','Fa','Po','NA'],
              'PavedDrive':['Y','P','N'],
              'PoolQC':['Ex','Gd','TA','Fa','NA'],
              'Fence':['GdPrv','MnPrv','GdWo','MnWw','NA']}

nu = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',
      'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
      'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
      'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
      'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars',
      'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
      'ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',
      'SalePrice']



"""
Categorical to Numerical
------------------------
"""

## Ordinal
for cat in catOrdinal:
    
    # Add to data train
    dummy =pd.get_dummies(dataTrain[cat],prefix=cat)
    dataTrain = pd.concat([dataTrain, dummy], axis=1, join_axes=[dataTrain.index])
    
    # Add to data test
    dummy =pd.get_dummies(dataTest[cat],prefix=cat)
    dataTest = pd.concat([dataTest, dummy], axis=1, join_axes=[dataTest.index])
    
    # Add to nu
    nu.extend(dummy.columns.tolist())
    
## Nominal
for cat in catNominal.keys():
    
    # Get domains
    dom = catNominal[cat]
    
    # Add to data train
    catNum = [dom.index(x) for x in dataTrain[cat].fillna('NA')  ]
    dataTrain[cat+'_Num'] = pd.Series(catNum, index=dataTrain.index)
    
    # Add to data test
    catNum = [dom.index(x) for x in dataTest[cat].fillna('NA')  ]
    dataTest[cat+'_Num'] = pd.Series(catNum, index=dataTest.index)
    
    # Add to nu
    nu.append(cat+'_Num')

"""
Data selection
--------------
"""

# Pearson correlation
pearson  = dataTrain[nu].corr('pearson' )
spearman = dataTrain[nu].corr('spearman')
pearson  =  pearson[tr]
spearman = spearman[tr]


# Data selection
umb = 0.05
select_pearson  = pearson [ pearson > umb].dropna().abs().sort_values(tr).index.tolist()
select_spearman = spearman[spearman > umb].dropna().abs().sort_values(tr).index.tolist()
select_nu = [x for x in select_spearman if x in select_pearson]

df = dataTrain[ select_nu ].fillna(0);

# Preprocessing
ymean = df['SalePrice'].mean()
ystd  = df['SalePrice'].std ()
df = (df-df.mean())/df.std()


# Plot
#sns.pairplot(df, x_vars=select_nu[:-1], 
#                 y_vars=select_nu[ -1], 
#                 size=7, aspect=0.7, kind='reg')

# Cross-validation
x = df[ select_nu[:-1] ].values
y = df[ select_nu[ -1] ].values.reshape(-1, 1)
kf = KFold(n_splits=8)

def aic(y, y_pred, p):
#   y: array-like of shape = (n_samples) including values of observed y
#   y_pred: vector including values of predicted y
#   p: int number of predictive variable(s) used in the model
    
    # Calculation
    resid = np.subtract(y_pred, y)
    rss = np.sum(np.power(resid, 2))
    aic_score = len(y)*np.log(rss/len(y)) + 2*p

    return aic_score


def bic(y, y_pred, p):
#   y: array-like of shape = (n_samples) including values of observed y
#   y_pred: vector including values of predicted y
#   p: int number of predictive variable(s) used in the model
    residual = np.subtract(y_pred, y)
    SSE = np.sum(np.power(residual, 2))
    BIC = len(y)*np.log(SSE/len(y)) + p*np.log(len(y))
    
    return BIC
    

"""
Python implementation
---------------------
"""
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()
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
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Python implementation:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    


"""
Batch Gradient Descent
----------------------
"""
def BatchGradientDescent(x_train,y_train,x_test):
    
    # Parameters
    alpha     = 0.001
    err       = 1000
    errNorm   = 1000
    threshold = 0.001
    
    n_predictor = len(x_train[0])
    n_samples   = len(y_train   )
    theta = np.zeros([n_predictor + 1,1])
    
    # Train Loop
    while (errNorm>threshold):
        exErr = err
        err   = 0
        
        # Cost function
        y_pred = np.dot(x_train,theta[:-1]) + theta[:][-1]
        J =  (1.0/n_samples)*  np.dot( x_train.T, y_train - y_pred)
        
        # Theta calculation
        for i in range( n_predictor ):
            theta[i] = theta[i] + alpha *np.dot(x_train[:][i],J)  
        
        # Error
        err = np.sum(np.abs(y_train - y_pred))
        
        # Update error
        errNorm = np.abs(exErr - err)/np.abs(err)
        
    return np.dot(x_test,theta[:][:-1]) + theta[:][-1]
##  ---------------------------------------------------------------------------    

MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Stochastic Gradient Descent
    y_pred = BatchGradientDescent(x_train,y_train,x_test)
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Batch Gradient Descent:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
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
    
    theta = np.zeros( len(x_train[0]) + 1 )
    
    # Train Loop
    while (errNorm>threshold):
        
        exErr = err
        err   = 0
        for xs, ys in zip(x,y):
            xs = np.append(xs,1)
            y_pred = theta * xs
            
            # Theta
            theta  = theta + alpha * (ys - y_pred) * xs
            
            # Error
            err = err + np.sum(np.abs(ys - y_pred))
            
        # Update error
        errNorm = np.abs(exErr - err)/np.abs(err)
        
    return np.dot(x_test,theta[:][:-1]) + theta[:][-1]
##  ---------------------------------------------------------------------------    


MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Stochastic Gradient Descent
    y_pred = StochasticGradientDescent(x_train,y_train,x_test)
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Stochastic Gradient Descent:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
Random Forest Regressor
-----------------------
"""

MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = RandomForestRegressor(random_state=0,n_estimators=100)
    
    # Train
    regr.fit(x_train,y_train)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Random Forest Regressor:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
Gradient Boosting Regressor
---------------------------
"""
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = GradientBoostingRegressor(random_state=0,n_estimators=100)
    
    # Train
    regr.fit(x_train,y_train)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Gradient Boosting Regressor:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
LASSO Regression
----------------
"""
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()
for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = linear_model.Lasso(alpha=0.0001)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('LASSO Regression:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
PLS Regression
--------------
"""
from sklearn.cross_decomposition import PLSRegression
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = PLSRegression(n_components=5)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('PLS Regression:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
Gaussian process regression 
---------------------------
"""
from sklearn.gaussian_process import GaussianProcessRegressor
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = GaussianProcessRegressor(random_state=0)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Gaussian process regression :')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
Huber Regressor
---------------
"""
from sklearn.linear_model import HuberRegressor
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = HuberRegressor()
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Huber Regressor:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
RANSAC Regressor
----------------
"""
from sklearn.linear_model import RANSACRegressor
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = RANSACRegressor(random_state=0)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('RANSAC Regressor:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
Linear Support Vector Regression
--------------------------------
"""
from sklearn.svm import LinearSVR
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = LinearSVR(random_state=0, tol=1e-8)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Linear Support Vector Regression:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    



"""
Nu Support Vector Regression
--------------------------------
"""
from sklearn.svm import NuSVR
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = NuSVR(tol=1e-6)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Nu Support Vector Regression:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    


"""
Epsilon-Support Vector Regression
---------------------------------
"""
from sklearn.svm import SVR
MAE = list(); MSE = list(); RMSE = list(); R2 = list(); AIC = list(); BIC = list()

for train, test in kf.split(x):
    # Select
    x_train = x[train]; y_train = y[train]
    x_test  = x[test ]; y_test  = y[test ]
    
    # Regression
    regr = SVR(tol=1e-6)
    
    # Train
    regr.fit(x,y)
    
    # Test
    y_pred = regr.predict( x_test )
    
    # Metrics
    MAE .append(        metrics.mean_absolute_error(y_test, y_pred) )
    MSE .append(        metrics.mean_squared_error (y_test, y_pred) )
    RMSE.append(np.sqrt(metrics.mean_squared_error (y_test, y_pred)))
    R2  .append(        metrics.r2_score           (y_test, y_pred) )
    
    AIC.append( aic(y_test,y_pred,len(select_nu)-1) )
    BIC.append( bic(y_test,y_pred,len(select_nu)-1) )
    
    
print('Epsilon-Support Vector Regression:')
print('MAE =',np.average(MAE),'\tMSE =',np.average(MSE),'\tRMSE=',np.average(RMSE))
print('R2 =',np.average(R2),'AIC =',np.average(AIC),'\tBIC =',np.average(BIC))
print('\n')    


"""
Data test result
----------------
"""
select_nu.pop()
select_nu.append('Id')

dt = dataTest[ select_nu ].fillna(0)
index = dt[ 'Id' ].values
dt = dt.drop(['Id'], axis=1)


# Preprocessing
dt = (dt-dt.mean())/dt.std()

x_test = dt.values

"""
pca = PCA(copy=True, iterated_power='auto', n_components=20, random_state=None,
          svd_solver='full', tol=0.0, whiten=True)

x_test = pca.fit_transform(x_test)
x_test = x_test[:,:10]
"""


# Regression
regr = GradientBoostingRegressor(random_state=0,n_estimators=100)   # RandomForestRegressor GradientBoostingRegressor
regr.fit(x,y)
y_test = regr.predict( x_test )

y_test = y_test*ystd+ymean

# Out
test = {'Id': index, 'SalePrice': y_test}
test = pd.DataFrame(data=test)
test.to_csv('sample_submission.csv',index=False)