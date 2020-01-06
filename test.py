import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import svm
from sklearn import ensemble, preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import csv


df = pd.read_csv(
    './merge.csv')
print('one')
one = df[df.target == 1]
one.to_csv('./one.csv', index=False)
print('zero')
zero = df[df.target == 0]
zero.to_csv('./zero.csv', index=False)
