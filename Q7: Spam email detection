#Importing needed modules
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

all_emails_dir = []
ok_train_dir = '../input/spam-emails/ok_training'
spam_train_dir = '../input/spam-emails/spam_training'

ok_train_lst = os.listdir(ok_train_dir)
ok_train_lst.sort()

spam_train_lst = os.listdir(spam_train_dir)
spam_train_lst.sort()

for i in ok_train_lst:
    tmp = os.path.join(ok_train_dir, i)
    all_emails_dir.append(tmp)
    
for i in spam_train_lst:
    tmp = os.path.join(spam_train_dir, i)
    all_emails_dir.append(tmp)

random.shuffle(all_emails_dir)

# creating labels from the files' name
y_train = []
for i in all_emails_dir:
    tmp = i.split('/')
    if tmp[-2] == 'spam_training':
        y = 1
    else:
        y = 0
    y_train.append(y)
    
y_train = np.array(y_train)

# cleaning lines in emails and putting them all in a list
all_lines = []
for i in tqdm(all_emails_dir):
    f = open(i)
    for line in f:
        line = line.strip('> >')
        line = line.strip('\n')
        all_lines.append(line)

# using word embedding method
vocab_size = 10000
embedding_dim = 64
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(all_lines)
word_index = tokenizer.word_index

all_files = []
for i in tqdm(all_emails_dir):
    tmp_file = []
    f = open(i)
    for line in f:
        line = line.strip('> >')
        line = line.strip('\n')
        tmp_file.append(line)
    a = tokenizer.texts_to_sequences(tmp_file)
    all_files.append(a)

# finding 1000 most used words in all emails
final_feature = []
num_features = 1000
for a in tqdm(all_files):
    tmp_arr = np.zeros((num_features,))
    for i in range(num_features):
        for line in a:
            if i+1 in line:
                tmp_arr[i] = 1
                break
    final_feature.append(tmp_arr)
x_train = np. array(final_feature)

model = RandomForestClassifier(n_estimators=1000, max_depth=10)
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=10)

test_dir = '../input/spam-emails/test'
test_lst = os.listdir(test_dir)
test_lst.sort()

test_full_dir = []
for i in test_lst:
    test_full_dir.append(os.path.join(test_dir,i))

test_all_files = []
for i in tqdm(test_full_dir):
    tmp_file = []
    f = open(i)
    for line in f:
        line = line.strip('> >')
        line = line.strip('\n')
        tmp_file.append(line)
    a = tokenizer.texts_to_sequences(tmp_file)
    test_all_files.append(a)

test_final_feature = []
num_features = 1000
for a in tqdm(test_all_files):
    tmp_arr = np.zeros((num_features,))
    for i in range(num_features):
        for line in a:
            if i+1 in line:
                tmp_arr[i] = 1
                break
    test_final_feature.append(tmp_arr)
    
x_test = np.array(test_final_feature)

pred = model.predict(x_test)

d = dict()
for num,i in enumerate(test_lst):
    d[i] = pred[num]

tst_a = pd.read_csv('../input/spam-emails/test.csv').values

p = []
for i in tst_a:
    for k,v in d.items():
        if i == k:
            p.append(v)

final_pred = []
for i in p:
    if i == 0:
        final_pred.append('False')
    elif i == 1:
        final_pred.append('True')

tmp_df = pd.read_csv('../input/spam-emails/test.csv')
df = pd.DataFrame()
df['filename'] = tmp_df['filename']
df['prediction'] = final_pred
df.to_csv('./test.csv')

