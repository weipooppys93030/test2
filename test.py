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

print('chinese', df[df.language == 3].shape[0],
      one[one.language == 3].shape[0], zero[zero.language == 3].shape[0])

print('english', df[df.language == 52].shape[0],
      one[one.language == 52].shape[0], zero[zero.language == 52].shape[0])

print('korean', df[df.language == 31].shape[0],
      one[one.language == 31].shape[0], zero[zero.language == 31].shape[0])

print('japanese', df[df.langage == 17].shape[0],
      one[one.language == 17].shape[0], zero[zero.language == 17].shape[0])

print('taiwanese', df[df.langage == 10].shape[0],
      one[one.language == 10].shape[0], zero[zero.language == 10].shape[0])
