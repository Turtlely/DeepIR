#Training profile for Tertiary amine (NH0) counting

'''
Tertiary Amine counting model, Inception V3, counts a maximum of 4 groups
1641 entries in IR-FG.csv
Evaluation results (Thu, July 7,9:44PM): OUTDATED

'''

from keras.optimizers import Adam

name = 'NH0counter'
epochs = 150
batch_size=8
validation_split = 0.1
learning_rate = 0.00001
loss = 'categorical_crossentropy'

optimizer = Adam(learning_rate=learning_rate)


'''
'''