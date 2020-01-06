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

one = pd.read_csv(
    './one.csv')

zero = pd.read_csv(
    './zero.csv')

print('male', df[df.gender == 'male'].shape[0],
      one[one.gender == 'male'].shape[0], zero[zero.gender == 'male'].shape[0])
print('female', df[df.gender == 'female'].shape[0],
      one[one.gender == 'female'].shape[0], zero[zero.gender == 'female'].shape[0])
