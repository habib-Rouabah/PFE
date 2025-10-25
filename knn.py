from matplotlib import cm
from  matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.interpolate import CubicSpline, interp1d
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

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, random_state = None)

print("X_train shape ",X_train.shape)
print("X_test shape ",X_test.shape)
classifier = NearestNeighbors(n_neighbors=4)#n_neighbors=2
print(classifier.get_params(deep=True))
classifier.fit(X_train[y_train==0])


distances, indices = classifier.kneighbors(X_test)#, return_distance=False
mindistances= np.amin(distances, axis=1)
maxdistances= np.amax(distances, axis=1)
distances= mindistances
Y_test_prediction = np.where(distances > 3.1, 1, 0)
perf= classification_report(y_test,Y_test_prediction)
print("All : ",perf)
auc = roc_auc_score(y_test,Y_test_prediction)
print("AUC : ",auc)
###############################################################

import matplotlib.pyplot as plt
s = 121
idx = np.array(range(len(X_test)), float)
print("idx : ",idx.shape)
print("distances : ",distances.shape)
plt.scatter(idx, distances,c=y_test, s=50, marker='o', alpha=.4)
plt.ylabel('Distances')
plt.xlabel('Sequence')
plt.grid(True)
plt.show()
