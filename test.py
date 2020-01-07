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

train = pd.read_csv(
    './test.csv')
songs = pd.read_csv(
    './songs.csv')

# songs.pop('genre_ids')
songs.pop('artist_name')
songs.pop('composer')
songs.pop('lyricist')

# members = pd.read_csv(
#    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/members.csv')

members = pd.read_csv(
    './members.csv')


# members.pop('gender')

expiration = members.pop('expiration_date')
last = expiration - members['registration_init_time']
expiration = pd.concat([expiration, last], axis=1)
expiration.columns = ['expiration_date', 'remaining_time']
members = pd.concat([members, expiration], axis=1)


songs.set_index('song_id', inplace=True)
members.set_index('msno', inplace=True)

train.set_index('song_id', inplace=True)
train = train.join(songs)

train.set_index('msno', inplace=True)
train = train.join(members)

train.fillna(value=0, inplace=True)

train.to_csv(
    './test2.csv', index=False)
