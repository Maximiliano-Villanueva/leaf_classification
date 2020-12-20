# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils


train = pd.read_csv('data/train.csv.zip')
test = pd.read_csv('data/test.csv.zip')


def encode(train, test):
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)

    train = train.drop(['species', 'id'], axis=1)
    test_ids=test.id
    test = test.drop('id', axis=1)

    return train, labels, test, classes,test_ids

train, labels, test, classes,test_ids = encode(train, test)

"""
standardize train features
"""

scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

"""
split train data into train and validation
"""
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)

for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]
    
nb_features = 64 # number of features per features type (shape, texture, margin)   
nb_class = len(classes)


"""
reshape
"""
# reshape train data
X_train_r = np.zeros((len(X_train), nb_features, 3))
X_train_r[:, :, 0] = X_train[:, :nb_features]
X_train_r[:, :, 1] = X_train[:, nb_features:128]
X_train_r[:, :, 2] = X_train[:, 128:]

# reshape validation data
X_valid_r = np.zeros((len(X_valid), nb_features, 3))
X_valid_r[:, :, 0] = X_valid[:, :nb_features]
X_valid_r[:, :, 1] = X_valid[:, nb_features:128]
X_valid_r[:, :, 2] = X_valid[:, 128:]



# Keras model with one Convolution1D layer
# unfortunately more number of covnolutional layers, filters and filters lenght 
# don't give better accuracy
model = Sequential()
model.add(Convolution1D(512, 1, input_shape=(nb_features, 3)))
model.add(Activation('relu'))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_class))
model.add(Activation('softmax'))


y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

nb_epoch = 15
model.fit(X_train_r, y_train, epochs=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=16)


scaler = StandardScaler().fit(test.values)
scaled_test = scaler.transform(test.values)

test_dataset = np.zeros((len(scaled_test), nb_features, 3))
test_dataset[:, :, 0] = scaled_test[:, :nb_features]
test_dataset[:, :, 1] = scaled_test[:, nb_features:128]
test_dataset[:, :, 2] = scaled_test[:, 128:]

preds_test = model.predict_proba(test_dataset)
print(preds_test)


submission = pd.DataFrame(preds_test, columns=classes)
submission.insert(0, 'id', test_ids)
submission.to_csv('submission.csv', index=False)