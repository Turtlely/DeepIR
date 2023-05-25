# Utility file for generating statistics plots

import os
import config
import pandas as pd
import matplotlib.pyplot as plt

# Path to folder containing run data for all 11 models
RUN_PATH = "/home/ryant/Documents/GitHub/DeepIR-2/data/FINAL FINAL RUN"
md_path = "/home/ryant/Documents/GitHub/DeepIR-2/data/molecule_descriptors.csv"

# First generate all loss/accuracy plots
# Iterate through each folder, extract data from R_Training_Log.csv, plot
md = pd.read_csv(md_path,index_col=0,header=0)

for group in md.columns:
    training_log = pd.read_csv(RUN_PATH+f"/{group}_RUN/{group}_Training_Log.csv",sep=";")
    epoch = training_log['epoch']
    accuracy = training_log['accuracy']
    loss = training_log['loss']
    val_accuracy = training_log['val_accuracy']
    val_loss = training_log['val_loss']

    # Generate loss plot
    fig, ax = plt.subplots()
    ax.plot(epoch,loss)
    ax.plot(epoch,val_loss)
    fig.savefig(f"{group}_LOSS.png")

    # Generate accuracy plot


# Generate a distribution of molecules used to train each model
