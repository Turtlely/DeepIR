#Training profile for Carboxylic acid counting

'''
Aldehyde counting model, Inception V3, counts a maximum of 2 groups
1641 entries in IR-FG.csv
Evaluation results (Thu, July 7,9:44PM): OUTDATED

not finished, dont run
'''

from keras.optimizers import Adam

name = 'AldehydeCounter'
epochs = 150
batch_size=100
validation_split = 0.1
learning_rate = 0.0001
#loss = 'binary_crossentropy'
loss = 'categorical_crossentropy'

optimizer = Adam(learning_rate=learning_rate)


'''
'''