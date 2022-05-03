import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash import Input, Output
from dash import dash_table
from PreProcessing import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import warnings

warnings.catch_warnings()
warnings.simplefilter("ignore")

def mapping(xx):
    dict = {}
    count = -1
    for x in xx:
        dict[x] = count + 1
        count = count + 1
    return dict


for i in ['Gender', 'Type of Travel', 'Class', 'Age_Cat', 'Customer Type', 'Departure Delay', 'Arrival Delay']:
    unique_tag = df_2[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    df_2[i] = df_2[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)


X = df_2.loc[:, ~df_2.columns.isin(['satisfaction'])]

Y = df_2['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# 10.3 Random Forest

# perform training with random forest with all columns specify random forest classifier

clf_rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=2, random_state=42, verbose=1)

# perform training
clf_rf.fit(X_train, y_train)

# make predictions

# prediction on test using all features
y_pred = clf_rf.predict(X_test)
y_pred_score = clf_rf.predict_proba(X_test)

# %%-----------------------------------------------------------------------
# calculate metrics gini model

print("\n")

print("Classification Report for RandomForest: ")
print(classification_report(y_test, y_pred))
print("\n")

print("Accuracy for RandomForest: ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC for RandomForest: ", roc_auc_score(y_test, y_pred_score[:, 1]) * 100)

for feat, importance in zip(df_2.columns, clf_rf.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))
print("\n")

