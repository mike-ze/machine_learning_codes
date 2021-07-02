import psycopg2
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

con = psycopg2.connect(dbname= 'dbname', host='database-1.c1dkosdydkhh.ca-central-1.rds.amazonaws.com', port= '5432', user= 'yomikey', password= '11231123')
cur = con.cursor()
cur.execute("SELECT * FROM anomaly_data")
data = np.array(cur.fetchall())
cols = list(map(lambda x: x[0], cur.description))
data = pd.DataFrame(data, columns=cols)
data = data.apply(lambda x: x.str.strip())
cur.close() 
con.close()

# Feature engineering
data = data.fillna({'accessedNodeType': '/batteryService'})
data["accessedNodeType"] = data["accessedNodeType"].astype('category')
data["accessedNodeType_cat"] = data["accessedNodeType"].cat.codes
data = data.fillna({'value': '0'})
data["value"] = data["value"].astype('category')
data["value_cat"] = data["value"].cat.codes
data.loc[data['normality'] != 'normal', 'normality'] = 1
data.loc[data['normality'] == 'normal', 'normality'] = 0
data["normality"] = data["normality"].astype('int8')
data["sourceAddress"] = data["sourceAddress"].astype('category')
data["sourceAddress_cat"] = data["sourceAddress"].cat.codes
data["sourceType"] = data["sourceType"].astype('category')
data["sourceType_cat"] = data["sourceType"].cat.codes
data["sourceLocation"] = data["sourceLocation"].astype('category')
data["sourceLocation_cat"] = data["sourceLocation"].cat.codes
data["destinationServiceAddress"] = data["destinationServiceAddress"].astype('category')
data["destinationServiceAddress_cat"] = data["destinationServiceAddress"].cat.codes
data["destinationServiceType"] = data["destinationServiceType"].astype('category')
data["destinationServiceType_cat"] = data["destinationServiceType"].cat.codes
data["destinationLocation"] = data["destinationLocation"].astype('category')
data["destinationLocation_cat"] = data["destinationLocation"].cat.codes
data["accessedNodeAddress"] = data["accessedNodeAddress"].astype('category')
data["accessedNodeAddress_cat"] = data["accessedNodeAddress"].cat.codes
data["operation"] = data["operation"].astype('category')
data["operation_cat"] = data["operation"].cat.codes

# Train test split
x = data[['sourceType_cat', 'sourceAddress_cat', 'sourceLocation_cat', 'destinationServiceType_cat', 'destinationServiceAddress_cat', 'destinationLocation_cat', 'accessedNodeType_cat', 'accessedNodeAddress_cat', 'operation_cat', 'value_cat']]
y = data[['normality']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# Model: XGBoost without oversamlping
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder =False, learning_rate= 1.5358256467186069, n_estimators = 35, max_depth = 14, verbosity = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
precision, recall, f1score, ___ = precision_recall_fscore_support(y_test, y_pred, average='weighted') 
print("\nAccuracy: %.10f" %(score))
print('Precision: %s' %(precision))
print('Recall: %s' %(recall))
print('F1 score: %s' %(f1score))
