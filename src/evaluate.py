from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix
import config
import pandas as pd

#model_PATH='/Users/ryantang/Documents/GitHub/DeepIR/models/CAregressor.h5'
model_PATH='/Users/ryantang/Documents/GitHub/DeepIR/models/Carboxylic Acid/CAcounter.h5'
history_PATH='/Users/ryantang/Documents/GitHub/DeepIR/models/Carboxylic Acid/CAcounter_History.csv'
#history_PATH='/Users/ryantang/Documents/GitHub/DeepIR/models/CAregressor_History.csv'
data_dir = config.ROOT_DIR+'/Datasets/IR-FG.csv'


'''import dataset'''
dataset = pd.read_csv(data_dir,index_col=0,sep='\t')
dataset = dataset.dropna()

#Split into X and y, train and test
X = dataset.iloc[:,:12000]
y = dataset.iloc[:,12000] #12000 for CA, 12007 for NH0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.99) #larger test size here

from keras.models import load_model

model = load_model(model_PATH)

predictions = model.predict(
    x=X_test,
    batch_size=10
)

print(model.summary())

'''Evaluations start here'''

#Confusion matrix
from tools import lossHistory, plot_confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

y_test = pd.get_dummies(y_test)
padding = predictions.shape[1]-y_test.shape[1]
if padding >0:
    y_test = np.pad(y_test,([0,0],[0,padding]),'constant') #make sure same shape
else:
    predictions = np.pad(predictions,([0,0],[0,abs(padding)]),'constant') #make sure same shape


print(y_test.shape)
#print(y_test)
print()
print(predictions.shape)
#print(np.round(predictions))



history = pd.read_csv(history_PATH)
lossHistory(history)

mse = mean_squared_error(y_test,predictions)
print('Mean Squared Error: ', mse)

classes = [str(n) for n in range(y_test.shape[1])]
print('Classes: ', classes)
cm = confusion_matrix(y_true=np.array(y_test).argmax(axis=1),y_pred=np.round(predictions.argmax(axis=1)))

eval_report = classification_report(y_true=np.array(y_test).argmax(axis=1),y_pred=np.round(predictions.argmax(axis=1)))
print('Evaluation Report')
print(eval_report)
plot_confusion_matrix(cm=cm,classes=classes,title=f'Confusion Matrix (Carboxylic acid counting)')

plt.show()
