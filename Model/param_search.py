#####################
## Search Function ##
#####################

# packages
import tables as tb
import numpy as np
import pandas as pd
import random as rd
# Plan of attack..

# Generate a TRAINING dataset that has 3 columns. phi, theta, and p

# 1) First, grab training dataset

ams = tb.open_file(train_path, 'r')
offer = ams.get_node("/images") # image 0

# 2) Grab image object from 0 to 16770

obj = list(offer)

# 3) create 3 lists that grab p theta and phi make dataframe

theta = np.array([obj[x].get_attr('theta') for x in range(0,16770)])
p = np.array([obj[x].get_attr('p') for x in range(0,16770)])
phi = np.array([obj[x].get_attr('phi') for x in range(0,16770)])

param_search = pd.DataFrame({'theta':theta, 'p':p, 'phi':phi})

# 4) Check to see if indexed correctly.

val = rd.randint(0,16770)
one = param_search['theta'][val] == obj[val].get_attr('theta') 
two = param_search['phi'][val] == obj[val].get_attr('phi')
three = param_search['p'][val] == obj[val].get_attr('p')
one*1 + two*1 + three*1 == 3

# 5) write as a csv

    # param_search.to_csv("param_search.csv", index = False)

# 6) Write function that does a search on the first column, then second, then third column. outputs index.

################# Bottom should not depend on anything above #######################
# make sure to have param_search.csv in directory

import tables as tb
import numpy as np
import pandas as pd
from itertools import starmap

# Data to use
df = pd.read_csv("param_search.csv", sep = ",")
# Function to iterate
def ptp_search(p, theta, phi, df):
    idx = (np.abs(df['p']-p)).idxmin()
    df = df.loc[df['p'] == df['p'][idx]]
    idx = (np.abs(df['theta']-theta)).idxmin()
    df = df.loc[df['theta'] == df['theta'][idx]]
    idx = (np.abs(df['phi']-phi)).idxmin()
    df = df.loc[df['phi'] == df['phi'][idx]]
    return df.index.values.tolist()[0]



# 7) Make object iterable
p = np.random.uniform(-10, 10, (int(16770/2), 1)).tolist()
theta = np.random.uniform(0, 90, (int(16770/2), 1)).tolist()
phi = np.random.uniform(0,180, (int(16770/2),1)).tolist()

obj = np.array([ptp_search(pp,t,f, df) for pp,t,f in zip(p,theta,phi)])


# 8) Check on h5 Dataset

# pull the first 100 objects
idx = obj[0:100]

# load dataset.
import h5py
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


# read it in.. come on be faster.
(X_train,y_train),(X_test,y_test) = load_data(train_path,test_path)

# WORKS!!
np.shape(X_train[obj[0:1000]])
