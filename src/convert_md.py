# Utility script, you probably dont need to use this.
# This did some processing work when running experiments, but is now pretty much obsolete.

import config
import pandas as pd

# Dataset Path:
md_path = config.ROOT_PATH+"/data/molecule_descriptors.csv"

# Import CSV's
md = pd.read_csv(md_path,index_col=0,header=0)

# Drop NaN's
md = md.dropna()

# Combine amides together
md['AMIDE'] = md['PAMIDE'] + md['SAMIDE'] + md['TAMIDE']

# Drop other amides
md = md.drop('PAMIDE',axis=1)
md = md.drop('SAMIDE',axis=1)
md = md.drop('TAMIDE',axis=1)

# Drop disulfides
md = md.drop('DSULFIDE',axis=1)

for col in md.columns:
    # Convert into 1 or 0, 1 = present 0 = absent
    md[col].mask(md[col] != 0,1,inplace=True)

md.to_csv(config.ROOT_PATH+"/data/molecule_descriptors_converted_dropped.csv")
