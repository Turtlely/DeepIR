# Information on the IRS dataset

import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

irs = pd.read_csv(config.ROOT_PATH+'/data/CAS-IRS.csv')
md = pd.read_csv(config.ROOT_PATH+'/data/molecule_descriptors-WITH_NITRILE+ALKENE+ALKANE.csv',index_col=0)

#for some reason the OH's are filled with NA's, will need to redo those later
# It seems we have NO nitriles, this must be a SMARTS error

md.drop(md.columns[len(md.columns)-1], axis=1, inplace=True)
md = md.dropna()

#print(irs.describe())
print(md.describe())

plt.xticks = [0,1,2,3,4,5,6,7,8,9,10]


# Create a count plot for each functional group

for (columnName, columnData) in md.iteritems():
    print('Column Name : ', columnName)
    ax = sns.countplot(x=columnData,palette=['#ff2d00'])
    ax.bar_label(ax.containers[0])
    #plt.show()
    plt.savefig(f'{columnName}.png')









'''
CAS-IRS information:
8568 molecule entries
    - 80% Training set, 6854 entries
    - 20% test set, 1713 entries

X axis data:
    - Wavenumbers in cm^-1
Y axis data:
    - Absorption between 0 and 1

Normalized data between 0 and 1 (min = 0 max = 1)




molecule_descriptors information:
9 groups represented
    - ALCOHOL, ALDEHYDE,'AMIDE','KETONE','ETHER','NITRO', 'ACYLHALIDE', 'NITRILE', 'PSAMINE'

8125 molecule entries

Statistics on each group represented:
    - ALCOHOL:
        Mean: 0.311     Std: 0.549
        Min: 0      Q1: 0       Med: 0      Q3: 1       Max: 4
        Percent representation:
    - ALDEHYDE:
        Mean: 0.025      Std: 0.159
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 2
        Percent representation:
    - AMIDE:
        Mean: 0.025     Std: 0.197
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 4
        Percent representation:
    - KETONE
        Mean: 0.103     Std: 0.343
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 4
        Percent representation:
    - ETHER
        Mean: 0.337     Std: 0.668
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 6
        Percent representation:
    - NITRO
        Mean: 0.058     Std: 0.262
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 2
        Percent representation:
    - ACYLHALIDE
        Mean: 0.012     Std: 0.119
        Min: 0      Q1: 0       Med:0        Q3: 0      Max: 3
        Percent representation:
    - NITRILE (looks like NONE of our molecules have a nitrile group)
        Mean: 0     Std: 0
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 0
        Percent representation:
    - PRIMARY OR SECONDARY AMINES, EXCLUDING AMIDES:
        Mean: 0.105     Std: 0.344
        Min: 0      Q1: 0       Med: 0      Q3: 0       Max: 4
        Percent representation:
'''