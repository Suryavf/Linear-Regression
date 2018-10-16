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


for i in range(len( select_nu) - 1):
    # Regression
    regr = LinearRegression()
    
    # Select features
    sdf = df[ [select_nu[i] , select_nu[-1]] ].dropna()
    x = sdf[ sdf.columns[0] ].values.reshape(-1, 1)
    y = sdf[ sdf.columns[1] ].values.reshape(-1, 1)
    
    # Train
    regr.fit(x,y)
    
    # Test
    stf = dataTest[ [select_nu[i] , select_nu[-1]] ].dropna()
    x_test = stf[ stf.columns[0] ].values.reshape(-1, 1)
    y_real = stf[ stf.columns[1] ].values.reshape(-1, 1)

    y_pred = regr.predict( x_test )
    
    print( np.sqrt(metrics.mean_squared_error(y_real, y_pred)) )
