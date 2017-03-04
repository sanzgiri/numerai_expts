from sklearn.decomposition import FastICA, PCA, NMF  
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC  
import numpy as np  
import pandas as pd  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import VotingClassifier  
from sklearn.neural_network import BernoulliRBM  
from sklearn.pipeline import Pipeline  
from sklearn import preprocessing  
from sklearn.calibration import CalibratedClassifierCV  
from sklearn.cluster import MiniBatchKMeans

df = pd.read_csv("/home/asanzgiri/my_numerai/numerai_training_data.csv")
training = df.values[:, :50]  
classes = df.values[:, -1]  
training = preprocessing.scale(training)  
kmeans = MiniBatchKMeans(n_clusters=500, init_size=6000).fit(training)  
labels = kmeans.predict(training)

clusters = {}  
for i in range(0, np.shape(training)[0]):  
    label = labels[i]  
    if label not in clusters:  
        clusters[label] = training[i, :]  
    else:  
        clusters[label] = np.vstack((clusters[label], training[i, :]))

params = {'n_estimators': 1000, 'max_depth': 3, 'subsample': 0.5,  
          'learning_rate': 0.01}  
xgb = XGBClassifier(**params)  
ica = FastICA(10)

icas = {}  
for label in clusters:  
    icas[label] = ica.fit(clusters[label])

factors = np.zeros((np.shape(training)[0], 10))

for i in range(0, np.shape(training)[0]):  
    factors[i, :] = icas[labels[i]].transform(training[i, :].reshape(1, -1))

xgb = xgb.fit(factors, classes)    

tf = pd.read_csv("/home/asanzgiri/my_numerai/numerai_tournament_data.csv")  
forecast = tf.values[:, 1:]  
forecast = preprocessing.scale(forecast)  
labels = kmeans.predict(forecast)

factors = np.zeros((np.shape(labels)[0], 10))  
for i, label in enumerate(labels):  
    factors[i, :] = icas[label].transform(forecast[i, :].reshape(1, -1))

proba = xgb.predict_proba(factors)

of = pd.Series(proba[:, 1], index=tf.values[:, 0].astype(int))

of.to_csv("/home/asanzgiri/my_numerai/predictions.csv", header=['probability'], index_label='t_id')  
