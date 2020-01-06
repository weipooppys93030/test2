import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import csv


# training and predicting=========================
print('preprocess training data')

# read training data
pretrain = pd.read_csv(
    'train.csv')
# read testing data
pretest = pd.read_csv(
    'test.csv')

# pop ans of training data
ans = pretrain.pop('target')

pretest.pop('id')
# merge train and test to normalize
total = pretrain.append(pretest)

# fillna
total.loc[total['source_system_tab'].isnull(
), 'source_system_tab'] = 'source_system_tab_none'

total.loc[total['source_screen_name'].isnull(
), 'source_screen_name'] = 'source_screen_name_none'

total.loc[total['source_type'].isnull(
), 'source_type'] = 'source_type_none'

print('preprocess of total done')
print('total na', total.isnull().sum().sum())
# get_dummy
total = pd.get_dummies(data=total, columns=[
                       'source_system_tab', 'source_screen_name', 'source_type'])

# read or just use variable
members = pd.read_csv(
    'membersdummy.csv', index_col='msno')

songs = pd.read_csv(
    'songsdummy.csv', index_col='song_id')
print('songs', songs.isnull().sum().sum())


# merge total and songs,members

total.set_index('song_id', inplace=True)
total = total.join(songs)
print('join songs', total.isnull().sum().sum())

total.set_index('msno', inplace=True)
total = total.join(members)
print('join members', total.isnull().sum().sum())

print('already merge songs,members')

total.fillna(value=0, inplace=True)
# scalization
scaler = MinMaxScaler()
print('scale')
scaler.fit(total)


# sep total to train, test
print('split')
train = total[:pretrain.shape[0]]

test = total[pretrain.shape[0]:]


# can modify variable
#clf = linear_model.SGDClassifier(n_jobs=-1, verbose=1)

clf = XGBClassifier(n_jobs=-1)

print('start to train')
clf.fit(train, ans.astype('int'))

print('predicting')
result = clf.predict(test)

print('output result')
# output result
result = pd.DataFrame({'id': [str(
    i) for i in range(0, len(result))], 'target': result}, columns=['id', 'target'])
result.to_csv('result.csv',
              index=False, quoting=2)
