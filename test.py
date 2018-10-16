# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import pandas as pd 
import scipy  as sc
from scipy import stats

print(chr(27) + "[2J")

# Import train/test
dataTrain = pd.read_csv("train.csv")
dataTest  = pd.read_csv( "test.csv")

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
crr = dataTrain[nu].corr()
crr = crr[tr]
#crr = crr[:-1] 

# Anova one-way


# Preprocessing
select_nu = crr[crr > 0.5].dropna().abs().sort_values(tr).index.tolist()
df = dataTrain[ select_nu ]
df = (df-df.mean())/df.std()

# Regression
#for i in range(len( df) - 1):
    

slope, intercept, rho, p, std_err = stats.linregress(df[ nu[3] ],
                                                     df[ tr[0] ])


