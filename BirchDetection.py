from matplotlib import cm
from  matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.interpolate import CubicSpline, interp1d
from sklearn.cluster import KMeans, Birch, FeatureAgglomeration, BisectingKMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd
#from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
import numpy as np
###################################################################"
'''
cols=['Engine rpm','Lub oil pressure','Fuel pressure','Coolant pressure',
      'lub oil temp','Coolant temp']
pwd="data/"
df = pd.read_csv("data/engine_data.csv" ,delimiter=",");
X = df[cols]
y=df['Engine Condition']
'''
from sklearn.model_selection import train_test_split
df= pd.read_csv('fenetres.csv', delimiter=';')
data = df.values
print('data shape : ',data.shape)

y= df.iloc[: , -1].values
X = df.iloc[:,:-1].values
print('X shape : ',X.shape)
print('Y shape : ',y.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, random_state = None,shuffle=False)

print("X_train shape ",X_train.shape)
print("X_test shape ",X_test.shape)
birch = BisectingKMeans(n_clusters=2 ,random_state=0, max_iter=500)
birch.fit(X_train[y_train==0])
Y_test_prediction =birch.predict(X_test)

perf= classification_report(y_test,Y_test_prediction)
print("All : ",perf)

###############################################################