import pandas as pd
import config

# CAS-IRS Statistics
'''
# Dataset Path:
irs_path = config.ROOT_PATH+"/data/CAS-IRS.csv"

# Import CSV's
irs = pd.read_csv(irs_path, index_col=0,header=None)

# Drop NaN's
irs = irs.dropna()

print(irs)

'''

# molecular_descriptors Statistics

md_path = config.ROOT_PATH+"/data/molecule_descriptors.csv"

md = pd.read_csv(md_path,index_col=0,header=0)

md = md.dropna()

print(md)
