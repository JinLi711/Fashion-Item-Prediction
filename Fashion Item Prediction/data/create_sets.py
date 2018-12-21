# For splitting the train data into images and labels.

import pandas as pd

raw_data = pd.read_csv('train.csv')
images, labels =  raw_data.drop('label', axis=1), raw_data.loc[:,'label']

images.to_csv('images.csv')
labels.to_csv('labels.csv')
