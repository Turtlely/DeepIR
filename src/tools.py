#Evaluation tools
import matplotlib.pyplot as plt
import itertools
import numpy as np

    
def lossHistory(history):
    fig, ax = plt.subplots()
    ax.plot(history['loss'],label='loss')
    ax.plot(history['val_loss'],label='val_loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.legend(loc='upper right')
    fig.tight_layout()

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
        '''
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        '''
        fig, ax = plt.subplots()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes, rotation=45)
        ax.set_yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')