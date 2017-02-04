import numpy as np
import scipy.misc
import cv2
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import merge, Input
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
# In 'th' mode the channel dimension is at index 1 ,in 
# 'tf' mode it is at index 3
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
weightsOutputFile='imagerecog.{epoch:02d}-{val_loss:.3f}.h5'

root_dir = "/home/delhivery"
rel_path = "/Desktop/dataset"
train = pd.read_csv(root_dir + rel_path + "/Train/train.csv")
test = pd.read_csv(root_dir + rel_path + "/Test.csv")
train_size = 40000
test_size = 9000
X_train = np.zeros((train_size, 28, 28))
y_train = np.zeros((train_size, 1))
X_test = np.zeros((test_size, 28, 28))
y_test = np.zeros((test_size, 1))
for image_index in range(train_size):
    img = scipy.misc.imread(
        root_dir + rel_path + "/Train/Images/train/" + train.filename[image_index])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    X_train[image_index] = img
    y_train[image_index] = train.label[image_index]

for image_index in range(test_size):
    img = scipy.misc.imread(
        root_dir + rel_path + "/Train/Images/train/" + train.filename[train_size + image_index])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    X_test[image_index] = img
    y_test[image_index] = train.label[train_size + image_index]

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def inceptionModule(input_img):
    layer1a=Convolution2D(30, 1, 1, border_mode='same',
                            activation='relu')(input_img)
    layer1b=Convolution2D(30, 1, 1, border_mode='same',
                            activation='relu')(layer1a)
    layer1c=MaxPooling2D((2, 2), strides=(
        1, 1), border_mode='same')(layer1b)

    layer2a=Convolution2D(30, 1, 1, border_mode='same',
                            activation='relu')(layer1c)
    layer2b=Convolution2D(30, 3, 3, border_mode='same',
                            activation='relu')(layer2a)
    layer2c=MaxPooling2D((2, 2), strides=(
        1, 1), border_mode='same')(layer2b)

    layer3a=Convolution2D(30, 1, 1, border_mode='same',
                            activation='relu')(layer2c)
    layer3b=Convolution2D(30, 5, 5, border_mode='same',
                            activation='relu')(layer3a)
    layer3c=MaxPooling2D((2, 2), strides=(
        1, 1), border_mode='same')(layer3b)


    tower_1 = Convolution2D(30, 1, 1, border_mode='same',
                            activation='relu')(layer3c)
    tower_2 = Convolution2D(30, 3, 3, border_mode='same',
                            activation='relu')(tower_1)
    tower_3 = Convolution2D(30, 5, 5, border_mode='same',
                            activation='relu')(tower_1)
    tower_4 = MaxPooling2D((3, 3), strides=(
        1, 1), border_mode='same')(layer3c)
    tower_4 = Convolution2D(15, 1, 1, border_mode='same',
                            activation='relu')(tower_4)
    output = merge([tower_1, tower_2, tower_3, tower_4],
                   mode='concat', concat_axis=1)
    return output

inputs = Input(shape=(1, 28, 28))
fire1 = inceptionModule(inputs)
fire2 = Dropout(0.4)(fire1)
fire3 = Flatten()(fire2)
pred1 = Dense(4 * num_classes, activation='relu')(fire3)
pred2 = Dense(2 * num_classes, activation='relu')(pred1)
pred3 = Dense(num_classes, activation='softmax')(pred2)
model = Model(input=inputs, output=pred3)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(weightsOutputFile, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(X_train, y_train, validation_data=(X_test, y_test),callbacks=[checkpointer],
          nb_epoch=10, batch_size=1000, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
model.save(root_dir+rel_path+"/my_model3.h5")
