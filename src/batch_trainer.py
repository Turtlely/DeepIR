# This program trains a model for each group of interest automatically, and generates log data for each run

from GROUP_DETECTION import start_run
import pandas as pd
import config

# List of functional groups to train from
md_path = config.ROOT_PATH+"/data/molecule_descriptors.csv"
md = pd.read_csv(md_path,index_col=0,header=0)


for col in md.columns:
    print(f"TRAINING {col} MODEL")
    start_run(col,batch_size=16)


#start_run("ALKANE",n_runs=100,batch_size=64)

