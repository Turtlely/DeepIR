'''Neural network model class for carboxylic acid'''

from keras.layers import Input,Conv1D, MaxPooling1D, concatenate, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.models import Model

class Network:
    def __init__(self):
        In = Input(shape=(12000,1))
        layer_1 = Conv1D(20,1,activation='relu')(In)
        layer_1 = Conv1D(10,3,activation='relu')(layer_1)

        layer_2 = Conv1D(10, 1, activation='relu')(In)
        layer_2 = Conv1D(10, 5, activation='relu')(layer_2)

        layer_3 = MaxPooling1D(3, strides=1, padding='same')(In)
        layer_3 = Conv1D(10, 1, activation='relu')(layer_3)

        mid_1 = concatenate([layer_1, layer_2, layer_3],axis=1)

        flat_1 = Flatten()(mid_1)

        dense_1 = Dense(600, activation='relu')(flat_1)
        dense_2 = Dense(300, activation='relu')(dense_1)
        drop_1 = Dropout(0.6)(dense_2)
        dense_3 = Dense(150, activation='relu')(drop_1)
        output = Dense(5, activation='softmax',kernel_regularizer=l2(0.01))(dense_3)
        #output = Dense(1, activation='linear')(dense_3)

        self.model = Model([In],output)
        
    def compile(self,opt,loss):
        self.model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])

    def getSummary(self):
        print(self.model.summary())

    def getModel(self):
        return self.model

    def drawModel(self):
        pass
    
