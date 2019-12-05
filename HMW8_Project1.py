import re
import os
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn
import csv
import seaborn.linearmodels as sns
from sklearn.datasets import load_boston

from sklearn import datasets
boston = datasets.load_boston()

type(boston)

# importing Bunch object
from sklearn.utils import Bunch

# Cheking the dataset 
boston.keys()

# using DESCR (describe to oversee the data attributes, information, etc.)
print(boston['DESCR'])

# creating the data frame and organizing it
pd.DataFrame

pd.DataFrame (data = None, index = None, columns = None, dtype = None, copy = False)

# exploring the data
boston ['data']
boston ['data'].shape
boston ['feature_names']

# loading the data
df = pd.DataFrame(data = boston['data'], columns = boston ['feature_names'])

# exploring the first 5 lines of the dataset
df.head()
# exploring the last 5 lines of the dataset
df.tail()

# Analysing the types of dataset
df.dtypes

#analysing the data distribution

df.describe()
df.count()
df.min()
df.max()
df.median()
df.quantile()
df.mode()
df.notnull()
df.nunique()
df.notna()

# Defining the independent variable is "MEDV" (Median House Value)
df ['MEDV'] = boston ['target']

# rearanging the dataset
y = df.MEDV.copy()
del df ['MEDV']
df = pd.concat((y, df), axis = 1)
df.head()
df.tail()

# exploring the data describe and T methods
df.describe().T

# pairwise correlations
df[cols].corr()


def predictivity(correlations):
    corrs_with_target = correlations.ix[-1][:-1]
    return corrs_with_target[abs(corrs_with_target).argsort()[::-1]]

    print("Pearson's correlation:", file=file)
# eliminating some variables that will not be used to be abble to work with a smaller dataset

for col in ['ZN', 'NOX', 'RAD', 'PTRATIO', 'B']:
    del df [col]
df.head(11)
df.head()

df [cols].corr()
print(corr)

# pairwise correlation heatmap
import seaborn as sns
ax = sns.heatmap(df[cols].corr(), cmap = sns.cubehelix_palette(20, light = 0.95, dark = 0.15))
ax.xaxis.tick_top() 
plt.savefig('C:\\Users\\Rogerio\\Documents\\Big Data Diploma\\CEBD 1100\\Homework\\Homework_8\\figures\\boston-housing-scatter-corr.png', bbox_inches = 'tight', dpi = 300) 
plt.show()
df.head()
df.describe()

# pair of plots
sns.pairplot(df[cols], plot_kws = {'alpha': 0.6}, diag_kws = {'bins': 30}) 
# it was not possible due to the size of data (ValueError: cannot copy sequence with size 506 to array axis with dimension 14)

# regressiona analysis with MEDV as a  function of CRIM and TAX
fig, ax = plt.subplots(1, 2)
sns.regplot ('CRIM', 'MEDV', df, ax = ax [0], scatter_kws = {'alpha': 0.4})
sns.regplot ('TAX', 'MEDV', df, ax = ax [1], scatter_kws = {'alpha': 0.4})
plt.savefig ('C:\\Users\\Rogerio\\Documents\\Big Data Diploma\\CEBD 1100\\Homework\\Homework_8\\figures\\boston-housing-scatter.png', bbox_inches = 'tight', dpi = 300)
print(plt)
plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('- MEDV', '--headers', action='store_true',
                        help="shows the input file's column headers")
    parser.add_argument('-c1', '--column1', action='store', dest='column1', type=str,
                        help="perform a data sanity check for the provided column")
    parser.add_argument('-c2', '--column2', action='store', dest='column2', type=str,
                        help="perform a data sanity check for the provided column")
    args = parser.parse_args()
    boston = load_boston()
    df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    df['target'] = boston['target']
    check_column_names(df, headers=args.headers, column1=args.column1, column2=args.column2)
    plot_data(df, column1=args.column1, column2=args.column2)

from argparseGraph.argparseGraph import argparseGraph as agg

def parsarg():
    parser = argparse.ArgumentParser(description="Options for differents cenarios")
    parser.add_argument("-MEDV", dest="argv1", help="test", type=int)
    parser.add_argument("-ZN", dest="argv2", help="test", type=int, action='append')
    parser.add_argument("-INDUS", dest="argv3", help="test", type=str, default="test3")
    parser.add_argument("-t", dest="argv4", help="test", type=str, default=False)
    parser.add_argument("-a", dest="argv5", help="test", type=str)
    parser.add_argument("-s", dest="argv6", help="test", type=bool)
    parser.add_argument("-d", dest="argv7", help="test", type=str, action='append')
    args = parser.parse_args()

# ploting the relationship among the independent variable MEDV with some independent variables:

from sklearn.datasets import load_boston
boston = load_boston()
features = boston.data.T

plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=boston.target, cmap='viridis')
plt.xlabel(boston.feature_names[0])
plt.ylabel(boston.feature_names[1])
plt.show()

plt.scatter(features[0], features[3], alpha=0.2,
            s=100*features[2], c=boston.target, cmap='viridis')
plt.xlabel(boston.feature_names[0])
plt.ylabel(boston.feature_names[2])
plt.show()

plt.scatter(features[0], features[3], alpha=0.2,
            s=100*features[3], c=boston.target, cmap='viridis')
plt.xlabel(boston.feature_names[0])
plt.ylabel(boston.feature_names[3])
plt.show()

plt.scatter(features[0], features[4], alpha=0.2,
            s=100*features[4], c=boston.target, cmap='viridis')
plt.xlabel(boston.feature_names[0])
plt.ylabel(boston.feature_names[4])
plt.show()

plt.scatter(features[0], features[5], alpha=0.2,
            s=100*features[5], c=boston.target, cmap='viridis')
plt.xlabel(boston.feature_names[0])
plt.ylabel(boston.feature_names[5])
plt.show()

plt.scatter(features[0], features[6], alpha=0.2,
            s=100*features[6], c=boston.target, cmap='viridis')
plt.xlabel(boston.feature_names[0])
plt.ylabel(boston.feature_names[6])
plt.show()


def main():
    parser = argparse.ArgumentParser (description = "")
    parser = argparse_add_argument ()

if __name__ == "__main__":
    main()
