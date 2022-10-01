#Convert the species list to a csv file

import config
import pandas as pd

PATH = config.ROOT_PATH+"/data/"

df = pd.read_csv (PATH+"species.txt", sep='\t')
df.columns = ["NAME", "FORMULA", "CAS"]

#Drop all rows that are missing a CAS
df = df[df.CAS.notnull()]
df = df.dropna()

#Save as a new csv file
df.to_csv (PATH+"species.csv",index=None)
