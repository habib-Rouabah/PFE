import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, \
    AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv('fenetres.csv', delimiter=';')
data = df.values
print('data shape : ',data.shape)

y= df.iloc[: , -1].values
X = df.iloc[:,:-1].values
print('X shape : ',X.shape)
print('Y shape : ',y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,shuffle=False)
"""
knn = KNeighborsClassifier(n_neighbors=3)
gbc  = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
xgbc = XGBClassifier(eval_metric='mlogloss')
hgbc = HistGradientBoostingClassifier()
rfc = RandomForestClassifier()
dtc =  DecisionTreeClassifier(max_depth=5)
abc = AdaBoostClassifier()


for name,model in zip(['knn','gbc','xgbc','hgbc','rfc','dtc','abc','svm'],[knn,gbc,xgbc,hgbc,rfc,dtc,abc,svm]):
    print(name)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
"""
