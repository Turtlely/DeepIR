# Joins molecule_descriptors with CAS-IRS into one single dataset, then splits into training and testing

import config
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_path = config.ROOT_PATH+"/data/"

# Load molecule_descriptors
MD = pd.read_csv(data_path+"molecule_descriptors.csv")

# Load CAS-IRS
IRS = pd.read_csv(data_path+"CAS-IRS.csv")

# Join both dataframes by their indicies (CAS numbers)
df = pd.concat([IRS,MD],axis='columns')

# Drop NaN's that were placed in
df.dropna()

# Split into a training set and testing set
train, test = train_test_split(df, test_size=0.2)

# Visualize the distribution of FG's in training and test sets
df.hist(df.columns[4801:])
train.hist(train.columns[4801:])
test.hist(test.columns[4801:])
plt.show()