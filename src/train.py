'''
Trains the neural network on Functional Group dataset (FG)
'''

import pandas as pd
import config

#Import dataset
data_dir = config.ROOT_DIR+'/Datasets/IR-FG.csv'

dataset = pd.read_csv(data_dir,index_col=0,sep='\t')
dataset = dataset.dropna()

#Split into X and y, train and test
X = dataset.iloc[:,:12000]
y = dataset.iloc[:,12000] #CA counting

#Convert numerical count to categorical encoding
y = pd.get_dummies(y)

#Convert to True/False
#y = (y>0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1) #test size will always be 10%
import sys
import modelCA as model
import profileCA as profile

network = model.Network()
network.getSummary()
network.compile(profile.optimizer,profile.loss)

model = network.getModel()

#Training the model
history = model.fit(X,y,validation_split=profile.validation_split,epochs=profile.epochs,batch_size=profile.batch_size)

#save the model
model.save(config.ROOT_DIR+'/models/'+profile.name+'.h5')

hist_df = pd.DataFrame(history.history)

#save model history
hist_df.to_csv(config.ROOT_DIR+'/models/'+profile.name+'_History.csv')
