#NIST URL
NIST_URL = "https://webbook.nist.gov/cgi/cbook.cgi"

#Random seed
RANDOM_SEED=42

#Root path
import os
ROOT_PATH = str(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)))

# Test set size = 100, this means there will be 50 positive and 50 negatives
TEST_SET_SIZE=100

# Desired mean of the training dataset
DESIRED_MEAN = 0.3

# Learning Rate
LEARNING_RATE = 0.0001

# Validation set split
VALIDATION_SIZE = 0.2

# Final Threshold values
thresholds = {'ALCOHOL': 0.6088,
            'ALDEHYDE':0.5660,
            'KETONE':0.3511,
            'ETHER':0.8924,
            'NITRO':0.8792,
            'ACYLHALIDE':0.5692,
            'NITRILE':0.7687,
            'ALKENE':0.3319,
            'ALKANE':0.9870,
            'ESTER':0.2299,
            'PAMINE':0.4537,
            'SAMINE':0.0197,
            'TAMINE':0.0923,
            'ARENE':0.2332,
            'CARBOXYLIC_ACID':0.0079,
            'AMIDE':0.2775}