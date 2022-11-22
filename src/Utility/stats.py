# Information on the IRS dataset

import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#irs = pd.read_csv(config.ROOT_PATH+'/data/CAS-IRS.csv')
C = pd.read_csv(config.ROOT_PATH+'/data/COMBINED_IRS_MD.csv')
#md = pd.read_csv(config.ROOT_PATH+'/data/molecule_descriptors-WITH_NITRILE+ALKENE+ALKANE.csv',index_col=0)

#md.drop(md.columns[len(md.columns)-1], axis=1, inplace=True)
#md = md.dropna()

print(C.describe())
print(C)
quit()
print(md.describe(include='all'))


plt.xticks = [0,1,2,3,4,5,6,7,8,9,10]

# Create a count plot for each functional group

for (columnName, columnData) in md.iteritems():
    print('Column Name : ', columnName)
    ax = sns.countplot(x=columnData,palette=['#ff2d00'])
    ax.set_title(columnName)
    ax.set_xlabel(f"Number of detected {columnName} groups")
    ax.set_ylabel("Frequency")
    ax.bar_label(ax.containers[0])
    plt.savefig(config.ROOT_PATH + '/data/Statistics/'f'{columnName}.png')


'''
CAS-IRS information:
8568 molecule entries
    - 80% Training set, 6854 entries
    - 20% test set, 1713 entries

X axis data:
    - Wavenumbers in cm^-1
Y axis data:
    - Absorption between 0 and 1
'''