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

# songs preprocess
'''
print('run')
songs = pd.read_csv(open(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/songs.csv', newline=''))
print(songs.shape)

songs.pop('artist_name')
songs.pop('composer')
songs.pop('lyricist')
songs.loc[songs['genre_ids'].isnull(), 'genre_ids'] = (-10)
songs.loc[songs['language'].isnull(), 'language'] = (-1)
genre = songs.pop('genre_ids')
language = songs.pop('language')
# genre_ids ,language
languagedummy = pd.get_dummies(language.astype(str), prefix='language_')
#languagedummy.rename(str.join, axis='columns')
genredummmy = genre.str.get_dummies()
listname = []
for col in genredummmy.columns:
    listname.append('genre_' + col)
genredummmy.columns = listname
print('concat')
songs = pd.concat([songs, genredummmy], axis=1)
songs = pd.concat([songs, languagedummy], axis=1)
#songs = pd.Series(['1|2', np.nan, '2|3']).str.get_dummies()
songs.to_csv('/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/songsdummy.csv',
             index=False)
print('done')
'''


# members preprocess
'''
members = pd.read_csv(open(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/members.csv', newline=''))
print(members.shape)
members.loc[members['gender'].isnull(), 'gender'] = 'unknow'
agelist = list(range(-5, 81, 5))


members.loc[members['bd'] > 80, 'bd'] = 0


city = members.pop('city')
bd = members.pop('bd')
gender = members.pop('gender')
via = members.pop('registered_via')
init = members.pop('registration_init_time')
expiration = members.pop('expiration_date')

last = expiration-init
expiration = pd.concat([expiration, last], axis=1)
expiration.columns = ['expiration_date', 'remaining_time']

city = pd.get_dummies(city.astype(str), prefix='city_')
via = pd.get_dummies(via.astype(str), prefix='via_')
gender = pd.get_dummies(gender)

bd = pd.cut(bd, agelist)
bd = pd.get_dummies(bd, prefix='bd')

members = pd.concat([members, city], axis=1)
members = pd.concat([members, bd], axis=1)
members = pd.concat([members, gender], axis=1)
members = pd.concat([members, via], axis=1)
members = pd.concat([members, init], axis=1)
members = pd.concat([members, expiration], axis=1)
#members = pd.concat([members, last], axis=1)

# members.loc['last'] = int(['expiration_date']) - \
#    int(['registration_init_time'])


members.to_csv('/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/membersdummy.csv',
               index=False)
'''
'''
print('run')
pretrain = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/train.csv')
print(pretrain.isnull().sum())

pretest = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/test.csv')
print(pretest.isnull().sum())

ans = pretrain.pop('target')

total = pretrain.append(pretest)
total.loc[total['source_system_tab'].isnull(
), 'source_system_tab'] = 'source_system_tab_none'

total.loc[total['source_screen_name'].isnull(
), 'source_screen_name'] = 'source_screen_name_none'

total.loc[total['source_type'].isnull(
), 'source_type'] = 'source_type_none'

total = pd.get_dummies(total)


members = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/membersdummy.csv', index_col='msno')

songs = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/songs.csv', index_col='song_id')


total.set_index('msno', inplace=True)
total = total.join(members)
total.set_index('song_id', inplace=True)
total = total.join(songs)

scaler = MinMaxScaler()
scaler.fit(total)
'''
'''
print('run')
pretrain = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/testtrain.csv')
print(pretrain.isnull().sum())

pretest = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/testtest.csv')
print(pretest.isnull().sum())

ans = pretrain.pop('target')
yans = pretest.pop('target')

total = pretrain.append(pretest)
total.loc[total['source_system_tab'].isnull(
), 'source_system_tab'] = 'source_system_tab_none'

total.loc[total['source_screen_name'].isnull(
), 'source_screen_name'] = 'source_screen_name_none'

total.loc[total['source_type'].isnull(
), 'source_type'] = 'source_type_none'

total = pd.get_dummies(data=total, columns=[
                       'source_system_tab', 'source_screen_name', 'source_type'])

print(total)

members = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/membersdummy.csv', index_col='msno')
print(members.isna())
songs = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/songsdummy.csv', index_col='song_id')
print(songs.isna())


total.set_index('msno', inplace=True)
total = total.join(members)
total.set_index('song_id', inplace=True)
total = total.join(songs)

print(total.shape)

scaler = MinMaxScaler()
scaler.fit(total)

train = total[:pretrain.shape[0]]
test = total[pretrain.shape[0]:]

clf = linear_model.SGDClassifier(n_jobs=-1, verbose=1)

clf.fit(train, ans.astype('int'))
result = clf.predict(test)
result = pd.DataFrame({'id': [str(
    i) for i in range(0, len(result))], 'target': result}, columns=['id', 'target'])
#print(accuracy_score(yans, result))
result.to_csv('/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/result.csv',
              index=False, quoting=2)
'''
'''
songs = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/songs.csv')
print(songs.isnull().sum())
members = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/members.csv')
print(members.isnull().sum())
train = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/train.csv')
print(train.isnull().sum())
test = pd.read_csv(
    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/test.csv')
print(test.isnull().sum())
'''


# train = pd.read_csv(
#   '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/testtrain.csv')

train = pd.read_csv(
    './train.csv')

train.pop('source_system_tab')
train.pop('source_screen_name')
# train.pop('source_type')


# songs = pd.read_csv(
#    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/songs.csv')

songs = pd.read_csv(
    './songs.csv')

songs.pop('genre_ids')
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

train = train.dropna()

# train.to_csv(
#    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/merge.csv', index=False)

train.to_csv(
    './merge.csv', index=False)

print('dropna')
zero = []
one = []
for index, row in train.iterrows():
    if row['target'] == 1:
        one.append(row)
    else:
        zero.append(row)


one = pd.DataFrame(one, columns=train.columns)


# one.to_csv(
#    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/one.csv', index=False)

one.to_csv(
    './one.csv', index=False)

zero = pd.DataFrame(zero, columns=train.columns)

# zero.to_csv(
#    '/Users/jimmy/Desktop/Python/project/kkbox-music-recommendation-challenge/zero.csv', index=False)

zero.to_csv(
    './zero.csv', index=False)
