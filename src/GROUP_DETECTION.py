'''
This script detects the presence of a group inside of a molecule.
'''

#from loss import f1, f1_loss


# Import modules
import config
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout,MaxPooling1D,Flatten, Dense,GlobalMaxPooling1D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sn
import os

'''testing'''




import tensorflow.keras.backend as K
from sklearn.metrics import f1_score

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)








'''INITIALIZATION'''

# Required to allow training to not freeze on first epoch
CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
session = InteractiveSession(config=CONFIG)

# Display GPU's available
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))


'''START'''

def start_run(R,n_runs=200,batch_size=64,train_split=0.975,mu=0.1,o=0.05,min_lr=0.0000001):

    # R Group that should be detected
    #R = 'ALCOHOL'

    # Create data logging directory
    LOG_PATH = config.ROOT_PATH+"/data/"+R+"_RUN/"
    os.mkdir(LOG_PATH)

    '''DATA PREPROCESSING'''

    # Dataset Path:
    irs_path = config.ROOT_PATH+"/data/CAS-IRS.csv"
    md_path = config.ROOT_PATH+"/data/molecule_descriptors.csv"

    # Import CSV's
    irs = pd.read_csv(irs_path, index_col=0)
    md = pd.read_csv(md_path,index_col=0,header=0)

    # Convert into 1 or 0, 1 = present 0 = absent
    md[R].mask(md[R] != 0,1,inplace=True)

    # Make a training / test set that uses a proportion of both classes of data (Has R/Does not have R)

    # split MD into two sets of data.
    #   - withR = molecules that have >0 R groups
    #   - without_R = molecules with 0 R groups

    # Using a train test split of 95-5, select a random 80% out of withR and 80% out of without_R.
    # Combine these two selections together to produce the training set

    # Resulting training set will have 80% of the total molecules with R, as well as 80% of the molecules without R.
    # Should somewhat diminish the ability for the majority class (without R) to overshadow the minority class (with R)

    withR = md[md[R]>0][R]
    without_R = md[md[R]==0][R]

    withR_TRAIN = withR.sample(frac=train_split,random_state=config.RANDOM_SEED)
    without_R_TRAIN = without_R.sample(frac=train_split,random_state=config.RANDOM_SEED)

    withR_TEST = withR.drop(withR_TRAIN.index)
    without_R_TEST = without_R.drop(without_R_TRAIN.index)

    # Combine both sets of data and scramble rows
    # The training dataset should now have about 80% of the initial majority and minority class
    md_TRAIN = irs.join(pd.concat([withR_TRAIN,without_R_TRAIN])).dropna() # NOT shuffled because the data will be further processed in augmentation
    md_TEST = irs.join(pd.concat([withR_TEST,without_R_TEST]).sample(frac=1.0)).dropna() # Shuffled because the test set is pretty much complete

    '''DATA AUGMENTATION'''

    # Duplicate and noise molecule entries that have >0 R group

    # First select 50% of all entries that have >0 R group and select a random 80% of these entries
    noNoise = md_TRAIN.loc[md[R]>0].sample(frac=0.5,random_state=config.RANDOM_SEED) #Does not have noise applied, used to produce duplicates

    # Duplicate to have two copies of these entries

    dup = noNoise.copy() # Will have noise applied, duplicate of noNoise
    dup = dup.loc[dup.index.repeat(2)]
    noise = np.random.normal(mu, o, [dup.shape[0],dup.shape[1]-1]) # Create noise
    noise = noise/np.linalg.norm(noise) # Normalize
    noise = np.pad(noise, ((0,0),(0,1)), 'constant') # Pad right side with 0's

    #Produce noisy data
    withNoise = dup.add(noise)

    #Join to the final dataset, remember to shuffle the data
    md_TRAIN = pd.concat([md_TRAIN,withNoise]).sample(frac=1.0)

    '''MODEL BUILDING'''

    # Standard 1 Dimensional Convolutional Neural Network
    # A CNN is used because it can extract spatial features within the signal, which are directly linked to the presence or absence of certain functional groups.

    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(md_TEST.shape[1]-1,1)))
    model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=13, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    plot_model(model, to_file=f'{LOG_PATH}{R}_Model.png', show_shapes=True, show_layer_names=True)

    # Compile the model using Binary crossentropy and the Adam optimizer
    #model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.compile(loss=f1_loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    '''MODEL TRAINING'''

    # Produce Training X and Y sets
    md_TRAIN_X = md_TRAIN.iloc[:,:-1].values.reshape(md_TRAIN.shape[0],md_TRAIN.shape[1]-1,1)
    md_TRAIN_Y = md_TRAIN.iloc[:,-1].values.reshape(md_TRAIN.shape[0],1,)

    # Produce Testing X and Y sets
    md_TEST_X = md_TEST.iloc[:,:-1].values.reshape(md_TEST.shape[0],md_TEST.shape[1]-1,1)
    md_TEST_Y = md_TEST.iloc[:,-1].values.reshape(md_TEST.shape[0],1,)

    # Log training metrics in a CSV file
    csv_logger = CSVLogger(f'{LOG_PATH}/{R}_Training_Log.csv', append=False, separator=';')

    # Callback to escape the validation loss plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=min_lr)

    # Callback to initiate early stopping
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Fit model to training data
    history = model.fit(md_TRAIN_X,md_TRAIN_Y,epochs = n_runs,batch_size=batch_size,verbose = 1,validation_split=0.1,callbacks=[csv_logger,reduce_lr])	

    print(f"FINISHED TRAINING {R} MODEL")
    print("BEGINNING EVALUATION")

    # Model evaluation on the Test set, may not be necessary as a confusion matrix is generated next
    _, accuracy = model.evaluate(md_TEST_X, md_TEST_Y, batch_size=1, verbose=1)

    '''MODEL EVALUATION'''

    #Predictions of the test set
    y_prediction = model.predict(md_TEST_X)

    #Create confusion matrix
    cm = confusion_matrix(md_TEST_Y, np.round(y_prediction))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{R} Absent",f"{R} Present"])
    disp.plot(cmap=plt.cm.Blues)
    disp.figure_.savefig(f"{LOG_PATH}{R}_CONFUSION_MATRIX", bbox_inches='tight')

    # Calculate Confusion Matrix metrics
    tn, fp, fn, tp = cm.ravel()

    accuracy = ((tp+tn)/(tp+tn+fp+fn))
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1 = (2*(tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))

    # Save metrics to a csv
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
    pd.DataFrame(metrics,index=[0]).to_csv(f"{LOG_PATH}TEST_METRICS.csv")

    '''MODEL SAVING'''
    model.save(f'{LOG_PATH}/{R}_MODEL')

    print(f"FINISHED {R} MODEL")


'''
TODO Add run logging functionality to log all aspects of a run in a spreadsheet. can be used to make data tables and graphs in paper for more details DONE
The model has been successfully tested with detecting alkene groups and alcohol groups. once the model has been ironed out, it can be used for each of the 11 groups of interest.
Analysis of model results is still needed

Metrics to log for each model:
1. Testing precision
2. Testing recall
3. Testing f1 score
4. validation loss DONE
5. validation accuracy DONE
6. training loss DONE
7. training accuracy DONE
8. testing loss 
9. testing accuracy

create an automated pipeline for testing multiple models and generating evaluation analysis
parameterize data augmentation step
visualize saliency
'''