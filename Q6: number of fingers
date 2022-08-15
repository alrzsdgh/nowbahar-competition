#Importing needed modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image
from tqdm import tqdm

# importing train dataset
train_lst = os.listdir('../input/fingerdigit/train')

# preparing dataset
def prepare_images_and_labels(image_dir):    
    labels = np.zeros((6,1))
    l = int(image_dir.split('.')[-2][-1])
    labels[l] = 1
    img = image.imread(image_dir)
    return img, labels

x_train_lst = []
y_train_lst = []
for i in tqdm(train_lst):
    a , b = prepare_images_and_labels(os.path.join('../input/fingerdigit/train',i))
    x_train_lst.append(a)
    y_train_lst.append(b)
x_train = np.array(x_train_lst)
y_train = np.array(y_train_lst)

x_train = np.expand_dims(x_train,axis=-1)
y_train = np.reshape(y_train , (np.shape(y_train)[0] , np.shape(y_train)[1]))

# Deep CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(7,7), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.Conv2D(64, kernel_size=(7,7), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation = 'softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train,y_train,validation_split = 0.2 , epochs = 10)

test_lst = os.listdir('../input/fingerdigit/test')
test_lst.sort()

def prepare_test_images(image_dir):    
    img = image.imread(image_dir)    
    return img

tmp = []
for i in tqdm(test_lst):
    ID = int(i.split('.')[0])
    tmp.append(ID)
  
tmp.sort()

tst = []
for i in tqdm(tmp):
    a = str(i) + '.png'
    tst.append(a)
    
x_test_lst = []
for i in tqdm(tst):
    a = prepare_test_images(os.path.join('../input/fingerdigit/test',i))
    x_test_lst.append(a)

x_test = np.array(x_test_lst)
x_test = np.expand_dims(x_test , axis=-1)
pred = model.predict(x_test)

final_pred = []
for i in pred:
    final_pred.append(np.argmax(i))

df = pd.DataFrame()

df['ID'] = tmp
df['prediction'] = final_pred
df.to_csv('./output.csv')
