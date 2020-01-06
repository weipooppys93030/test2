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

print('init', df['registration_init_time'].sum(),
      one['registration_init_time'].sum(), zero['registration_init_time'].sum())

print('expiration_date', df['expiration_date'].sum(),
      one['expiration_date'].sum(), zero['expiration_date'].sum())

print('remaining_time', df['remaining_time'].sum(),
      one['remaining_time'].sum(), zero['remaining_time'].sum())
