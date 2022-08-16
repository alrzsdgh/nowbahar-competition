# Importing needed modules
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Model

# importing dataset
train_dir = '/content/drive/MyDrive/Food-Data/training'
train_folders = os.listdir(train_dir)
train_folders.sort()

all_samples = []
for i in tqdm(train_folders):
  tmp_lst = os.listdir(os.path.join(train_dir, i))
  tmp_lst.sort()
  for j in tmp_lst:
    all_samples.append(os.path.join(train_dir, i, j))

random.shuffle(all_samples)

# extracting labels from file's name
lbl = []
for i in all_samples:
  lbl.append(i.split('/')[-2])

lbl_arr = np.array(lbl)
l = np.expand_dims(lbl_arr, axis = -1)

# creating labels for a multiclass model (ex: egg --> [1,0,0,0,0])
enc = OneHotEncoder()
labels = enc.fit_transform(l).toarray()
print(np.shape(labels))

# make all images the same size
def load_and_resize(s_dir,desired_width = 150,desired_height = 150):
    img = plt.imread(s_dir)
    current_width = img.shape[0]
    current_height = img.shape[1]
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    img = ndimage.zoom(img, (width_factor, height_factor,1), order=1)
    return img

train_data = []
for i in tqdm(all_samples):
  train_data.append(load_and_resize(i))

x_train = np.array(train_data)

# importing the basic model for transfer learning
pre_model = tf.keras.applications.EfficientNetV2B3(include_top=False, input_shape=(150,150,3))

last_layer = pre_model.layers[-1]
last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)                  
x = tf.keras.layers.Dense  (5, activation='softmax')(x)           
model = Model(pre_model.input, x) 
model.summary()

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

checkpoint_filepath = '/content/model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(x_train,labels,
                    epochs=30,
                    callbacks=[model_checkpoint_callback],
                    validation_split = 0.2,
                    verbose = 1)

test_dir = '/content/drive/MyDrive/Food-Data/test'
test_lst = os.listdir(test_dir)
test_lst.sort()

test_data = []
for i in tqdm(test_lst):
  tmp_dir = os.path.join('/content/drive/MyDrive/Food-Data/test' , i)
  test_data.append(load_and_resize(tmp_dir))

test_data = np.array(test_data)

f_model = tf.keras.models.load_model('./model.h5')
test_pred = f_model.predict(test_data)

out_lst = []
for el in test_pred:
  tmp = np.argmax(el)
  if tmp == 0:
    out_pred = 'Egg'
  elif tmp == 1:
    out_pred = 'Fried food'
  elif tmp == 2:
    out_pred = 'Meat'
  elif tmp == 3:
    out_pred = 'Rice'
  elif tmp == 4:
    out_pred = 'Seafood'
  out_lst.append(out_pred)

df = pd.DataFrame()
df['file'] = test_lst
df['prediction'] = out_lst
df.to_csv('/content/sub.csv')
