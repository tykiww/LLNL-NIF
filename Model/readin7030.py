# https://hub.packtpub.com/generative-adversarial-networks-using-keras/
import h5py
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Reshape, Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose, UpSampling2D, LeakyReLU, Dropout, BatchNormalization
from tensorflow.python import keras
from tensorflow.python.keras.models import Model, Sequential

train_path = 'h5py_train_and_test_split/new_training_data.h5'
test_path = 'h5py_train_and_test_split/new_test_data.h5'

def load_data(train_path,test_path):
    # read in Data
    tests_data = test_path ; train_data = train_path;
    ts_df = h5py.File(tests_data,'r') ; tr_df = h5py.File(train_data,'r') ;
    # Scan for images and grab labels
    y_train = np.array(tr_df['images'])
    X_train = np.array([tr_df['images'][i][0] for i in y_train]) # 15 seconds..
    y_test = np.array(ts_df['images'])
    X_test = np.array([ts_df['images'][i][0] for i in y_test])
    # Reshape images to fit model
    X_train = np.expand_dims(X_train, axis = 3)
    X_test = np.expand_dims(X_test, axis = 3)
    # close h5py
    ts_df.close() ; tr_df.close()
    return (X_train*1.0,y_train),(X_test*1.0,y_test)

(X_train,y_train),(X_test,y_test) = load_data(train_path,test_path)

def normalize(train, test):
    mean, std = train.mean(), test.std()
    train = (train - mean) / std
    test = (test - mean) / std
    return train, test

(X_train, X_test) = normalize(X_train,X_test)
