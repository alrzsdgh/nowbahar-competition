# Needed modules
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# importing train dataset
train_df = pd.read_csv('../input/movie2/train.csv')

# Preparing multiclass labels (rate 4 --> [0,0,0,1,0]
lbl = train_df['Rating'].values
lbl = np.expand_dims(lbl , axis = -1)
enc = OneHotEncoder()
labels = enc.fit_transform(lbl).toarray()

# preprocessing comments by using embeded method to use them to train the model
cms = train_df['Review'].tolist()

vocab_size = 10000
embedding_dim = 64
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 720

# split train/validation data
training_sentences = cms[0:training_size]
testing_sentences = cms[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# a simple DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint_filepath = './model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

num_epochs = 100
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    callbacks=[model_checkpoint_callback],
                    verbose=2)

f_model = tf.keras.models.load_model('./model.h5')


# importing test dataset and preprocessing them
test_df = pd.read_csv('../input/movie2/test.csv')
tcms = test_df['Review'].tolist()

test_sequences = tokenizer.texts_to_sequences(tcms)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_padded = np.array(test_padded) 

pred = f_model.predict(test_padded)

final_pred = []
for i in pred:
    tmp = np.argmax(i)
    final_pred.append(tmp+1)


df = pd.DataFrame()
df['ID'] = test_df['ID']
df['prediction'] = final_pred
df.to_csv('./output.csv')
