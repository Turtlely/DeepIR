import numpy as np
import pandas as pd
import seaborn as selections
import matplotlib.pyplot as plt
import config
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# didnt work well :(

'''DATA PREPROCESSING'''


'''
# Dataset Path:
irs_path = config.ROOT_PATH+"/data/CAS-IRS.csv"

# Import CSV's
irs = pd.read_csv(irs_path, index_col=0,header=None)

# Drop NaN's
irs = irs.dropna()

print(irs.shape)

# PCA 

scaler =  StandardScaler()
scaler.fit(irs)
irs_scaled = scaler.transform(irs)

pca = PCA(n_components=100,random_state=2020)
pca.fit(irs_scaled)
X_pca = pca.transform(irs_scaled)

print(X_pca.shape)

print(X_pca)

df = pd.DataFrame(data=X_pca[0:,0:], index=list(irs.index.values))

print(df.shape)
print(df.head())

# Plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
'''

def apply_PCA(irs,n=100):
    # Drop NaN's
    irs = irs.dropna()

    scaler =  StandardScaler()
    scaler.fit(irs)
    irs_scaled = scaler.transform(irs)

    pca = PCA(n_components=100,random_state=2020)
    pca.fit(irs_scaled)
    X_pca = pca.transform(irs_scaled)
    df = pd.DataFrame(data=X_pca[0:,0:], index=list(irs.index.values))

    return df