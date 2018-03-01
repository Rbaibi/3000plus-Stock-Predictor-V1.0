## import modules

import pandas as pd
import quandl as Quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("\n")
print('CryptoTrader is a program that predicts the stockmarket using Machine Learning')
print("\n")
y = input("Do you want the full list stock options?(y/n)")

if y == 'y':
    df_wiki = pd.read_csv('WIKI-datasets-codes.csv')
    print(df_wiki)

print("\n")   
print('Your options are:')
print('1 = Choose stockdata to predict')
print('2 = Load Data from local source')
print("\n")
x = input("How would you like to proceed? ")


if x == '1':
    option2 = input("What stock do you want to predict? ")
    wiki= 'WIKI/' + option2
    df = Quandl.get(wiki)

elif x == '2':
    option2 = input("What is the name of the source? ")
    df = pd.read_csv(option2,index_col='Date', parse_dates=True)

else:
    print('please restart and choose 1 of the 3 options')
    


###_____________________________________

#print(df.head())
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

#HML High minus Low Percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
#Daily price change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print(df.head())
#

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_window = int(math.ceil(0.1 * len(df)))#forecast out 1% of the entire length of the dataset
#print(forecast_window)

#use data from former days
df['label'] = df[forecast_col].shift(-forecast_window)


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_window:]
X = X[:-forecast_window]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#clf = svm.SVR()
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
#print(confidence)

'''
#Every kernel for SVM
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
'''

#  Forecasting and Predicting

forecast_set = clf.predict(X_lately)
print('Prediction test:' + str(forecast_set))
print('The forecast window is ' + str(forecast_window) + ' days')
print('The accuracy of this model =' + str(confidence))


# Visualization
 
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-pastel')
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot(c='k')
df['Forecast'].plot(c='r')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



z = input("Do you want to save this model?(y/n) ")

if z == 'y':
    import pickle
    pickle_name = input("Under what name would you like to save? ")
    with open(pickle_name,'wb') as f:
        pickle.dump(clf, f)

else:
    raise SystemExit


