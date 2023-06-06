'''
This script detects the presence of a group inside of a molecule.
'''

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
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sn
import os

'''INITIALIZATION'''

# Required to allow training to not freeze on first epoch
CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
session = InteractiveSession(config=CONFIG)

# Display GPU's available
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))


from scipy.signal import savgol_filter    

# Smoothing function
def smooth(y, n=51):
    #cumsum = np.cumsum(np.insert(y, 0, 0)) 
    #return (cumsum[n:] - cumsum[:-n]) / float(n)
    #box = np.ones(box_pts)/box_pts
    #y_smooth = np.convolve(y, box, mode='same')
    #return y_smooth
    return savgol_filter(y,n,5)


'''START'''

def start_run(R,batch_size,n_runs=1,train_split=0.80,mu=0,o=0.02,min_lr=0.0000001):

    

    '''DATA PREPROCESSING'''

    # Dataset Path:
    irs_path = config.ROOT_PATH+"/data/CAS-IRS.csv"
    md_path = config.ROOT_PATH+"/data/molecule_descriptors.csv"

    # Import CSV's
    irs = pd.read_csv(irs_path, index_col=0,header=None)
    md = pd.read_csv(md_path,index_col=0,header=0)

    #print(irs)

    #print(md)

    # Drop NaN's
    md = md.dropna()
    irs = irs.dropna()

    # Convert into 1 or 0, 1 = present 0 = absent
    md[R].mask(md[R] != 0,1,inplace=True)

    # Make a training / test set that uses a proportion of both classes of data (Has R/Does not have R)

    # split MD into two sets of data.
    #   - withR = molecules that have >0 R groups
    #   - without_R = molecules with 0 R groups

    # Using a train test split of 80-20, select a random 80% out of withR and 80% out of without_R.
    # Combine these two selections together to produce the training set

    # Resulting training set will have 80% of the total molecules with R, as well as 80% of the molecules without R.
    # Should somewhat diminish the ability for the majority class (without R) to overshadow the minority class (with R)

    withR = md[md[R]>0][R]
    without_R = md[md[R]==0][R]

    print(withR)
    print(without_R)
    

    # Test set contains exactly 50 positives and negatives
    withR_TEST = withR.sample(n=50,random_state=config.RANDOM_SEED)
    without_R_TEST = without_R.sample(n=50,random_state=config.RANDOM_SEED)

    # Save the test set ID's so that we can manually check them later. purely for debugging
    print(len(withR_TEST))
    print(len(without_R_TEST))
    #quit()

    withR_TRAIN = withR.drop(withR_TEST.index)
    without_R_TRAIN = without_R.drop(without_R_TEST.index)

    ''' DELETE SOME OF THE MAJORITY CLASS HERE'''

    # Delete half of all entries that have 0 R groups

    # First select 50% of all entries that have 0 R group
    
    # Mean value in the dataset should be around 0.4
    # Count number of positives, negatives
    # Calculate amount of negatives to subtract to get the wanted mean

    numPos = len(withR_TRAIN)
    numNeg = len(without_R_TRAIN)
    desiredMean = 0.3

    amount_remove = -1 * (numPos - desiredMean * (numPos + numNeg))/desiredMean

    # TODO removing data possibly makes the model worse. check that
    # It seems removing data actually does make the model better

    # Occasionally it will try to remove too many
    try:
        without_R_TRAIN = without_R_TRAIN.drop(without_R_TRAIN.sample(n=int(np.abs(amount_remove))).index)
        pass
    except Exception as e:
        print(e)
        # Do not remove any


    # Combine both sets of data and scramble rows
    # The training dataset should now have about 80% of the initial majority and minority class
    md_TRAIN = irs.join(pd.concat([withR_TRAIN,without_R_TRAIN])).dropna() # NOT shuffled because the data will be further processed in augmentation
    md_TEST = irs.join(pd.concat([withR_TEST,without_R_TEST]).sample(frac=1.0)).dropna() # Shuffled because the test set is pretty much complete

    print(pd.concat([withR_TEST,without_R_TEST]).describe())
    quit()
    '''DATA AUGMENTATION'''

    
    # TODO adding noise possibly makes the model worse. check that
    # It seems adding noise makes the model perform a bit worse. (f1 scores are moved closer to 0.925)

    # Duplicate and noise molecule entries that have >0 R group

    # First select 100% of all entries that have >0 R group and select a random 80% of these entries
    #noNoise = md_TRAIN.loc[md[R]>0].sample(frac=1.0,random_state=config.RANDOM_SEED) #Does not have noise applied, used to produce duplicates
    noNoise = md_TRAIN.loc[md[R]>0].sample(frac=1,random_state=config.RANDOM_SEED) #Does not have noise applied, used to produce duplicates

    # Duplicate to have two copies of these entries

    dup = noNoise.copy() # Will have noise applied, duplicate of noNoise
    '''Automate this to generate the right amount of data'''
    
    #print(dup)

    dup = dup.loc[dup.index.repeat(1)] 

    #print(dup)
    #plt.plot(np.linspace(600,3700,6200),dup.iloc[100,:-1].to_list(),zorder=2)



    noise = np.random.normal(mu, o, [dup.shape[0],dup.shape[1]-1]) # Create noise
    #noise = noise/np.linalg.norm(noise) # Normalize
    noise = np.pad(noise, ((0,0),(0,1)), 'constant') # Pad right side with 0's

    #Produce noisy data
    #withNoise = dup.add(noise).clip(0,1)
    
    
    #print(noise)


    withNoise = pd.concat([dup.iloc[:,:-1].apply(smooth, axis=1,result_type='expand'),dup.iloc[:,-1]],axis=1)
    
    #print(list(withNoise.columns[:-1]))
    
    withNoise.columns = [x+1 for x in list(withNoise.columns)[:-1]] + [R]
    
    #print(withNoise.columns)

    #Join to the final dataset, remember to shuffle the data
    md_TRAIN = pd.concat([md_TRAIN,withNoise]).sample(frac=1.0)
    # NO NOISE ADDED. this makes the model better it seems
    
    #print(withNoise)

    #plt.plot(np.linspace(600,3700,6200),withNoise.iloc[100,:-1].to_list(),zorder=1)
    #plt.show()
    #print(md_TRAIN)
    #quit()
    '''MODEL BUILDING'''

    # Standard 1 Dimensional Convolutional Neural Network
    # A CNN is used because it can extract spatial features within the signal, which are directly linked to the presence or absence of certain functional groups.
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(md_TEST.shape[1]-1,1)))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    

    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Create data logging directory
    LOG_PATH = config.ROOT_PATH+"/data/"+R+"_RUN/"
    os.mkdir(LOG_PATH)


    plot_model(model, to_file=f'{LOG_PATH}{R}_Model.png', show_shapes=True, show_layer_names=True)

    # Compile the model using Binary crossentropy and the Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

    '''MODEL TRAINING'''

    # Produce Training X and Y sets
    md_TRAIN_X = md_TRAIN.iloc[:,:-1].values.reshape(md_TRAIN.shape[0],md_TRAIN.shape[1]-1,1)
    md_TRAIN_Y = md_TRAIN.iloc[:,-1].values.reshape(md_TRAIN.shape[0],1,)

    # Split training X and Y into validation sets
    x_train, x_val, y_train, y_val = train_test_split(md_TRAIN_X,md_TRAIN_Y, test_size = 0.2, random_state = config.RANDOM_SEED)


    # Produce Testing X and Y sets
    md_TEST_X = md_TEST.iloc[:,:-1].values.reshape(md_TEST.shape[0],md_TEST.shape[1]-1,1)
    md_TEST_Y = md_TEST.iloc[:,-1].values.reshape(md_TEST.shape[0],1,)

    # Log training metrics in a CSV file
    csv_logger = CSVLogger(f'{LOG_PATH}/{R}_Training_Log.csv', append=False, separator=';')

    # Callback to escape the validation loss plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=min_lr)

    # Callback to initiate early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,restore_best_weights=True)


    print(md_TRAIN[R].describe())

    md_TRAIN.iloc[:,-1].to_csv("model_input.csv")


    # Fit model to training data
    #history = model.fit(md_TRAIN_X,md_TRAIN_Y,epochs = n_runs,batch_size=batch_size,verbose = 1,validation_split=0.1,callbacks=[csv_logger,reduce_lr,early_stop])	
    history = model.fit(x_train,y_train,epochs = n_runs,batch_size=batch_size,verbose = 1,validation_data=(x_val,y_val),callbacks=[csv_logger,reduce_lr,early_stop])	

    # Model evaluation on the Test set, may not be necessary as a confusion matrix is generated next
    _, accuracy = model.evaluate(md_TEST_X, md_TEST_Y, batch_size=1, verbose=1)

    '''MODEL EVALUATION'''

    #Predictions of the test set
    y_prediction = model.predict(md_TEST_X)

    # Generate ROC and calculate AUC
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    fpr, tpr, thresholds = roc_curve(md_TEST_Y,y_prediction.ravel())
    auc = auc(fpr, tpr)
    
    # Calcualte G-Mean for each threshold
    gmeans = np.sqrt(tpr*(1-fpr))
    ix = np.argmax(gmeans)
    opt_threshold = thresholds[ix]
    print("Optimal Threshold: ", opt_threshold)

    # Clear plot so we can draw the ROC plot
    plt.clf()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(f"{LOG_PATH}{R}_roc_curve")

    # Clear plot to draw confusion matrix
    plt.clf()

    #Create confusion matrix using the optimal thresholds
    cm = confusion_matrix(md_TEST_Y, [0 if i < opt_threshold else 1 for i in y_prediction])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{R} Absent",f"{R} Present"])
    disp.plot(cmap=plt.cm.Blues)
    disp.figure_.savefig(f"{LOG_PATH}{R}_CONFUSION_MATRIX", bbox_inches='tight')

    # Calculate Confusion Matrix metrics
    tn, fp, fn, tp = cm.ravel()

    accuracy = ((tp+tn)/(tp+tn+fp+fn))
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    f1 = (2*(tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))

    # Clear plot
    plt.clf()

    # Save metrics to a csv
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Optimal Threshold": opt_threshold, "AUC": auc}
    pd.DataFrame(metrics,index=[0]).to_csv(f"{LOG_PATH}TEST_METRICS.csv")

    '''MODEL SAVING'''
    model.save(f'{LOG_PATH}/{R}_MODEL')

    print(f"FINISHED {R} MODEL")