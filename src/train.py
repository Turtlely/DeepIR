'''
Trains the neural network on Functional Group dataset (FG)
Uses Keras
'''
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import config

#Import dataset
data_dir = config.ROOT_DIR+'/Datasets/IR-FG.csv'

dataset = pd.read_csv(data_dir,index_col=0,sep='\t')
dataset.dropna()

FG_cols = 31

#Split into X and y, train and test
X = dataset.iloc[:,:12000]
y = dataset.iloc[:,-31:]
y = (y>0)
print(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
print(X_train.shape,X_test.shape)
print(y_train.shape, y_test.shape)

import model

network = model.Network()
network.summary()
network.compile()

model = network.getModel()

#Training the model
history = model.fit(X_train,y_train,validation_split=0.1,epochs=10,batch_size=10)

model.save('model2')

score = model.evaluate(X_test, y_test) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.xlabel('epoch')
plt.show()