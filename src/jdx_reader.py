#Utility for reading a jdx file
import config
from jcamp import JCAMP_reader,JCAMP_calc_xsec
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from rdkit import Chem
import requests

#Path to the folder with IRS data
JDX_PATH = config.ROOT_PATH+"/data/IRS/"

# URL to NIST Chemistry Webbook
URL = "https://webbook.nist.gov/cgi/cbook.cgi"

#File name that is to be looked at
FILE_NAME = '136112-69-1'

'''
#Load in IRS file
jcamp_dict = JCAMP_reader(JDX_PATH+FILE_NAME+".jdx")
JCAMP_calc_xsec(jcamp_dict)

#Normalize function
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if jcamp_dict['yunits'].upper() != 'ABSORBANCE':
    print("Error, not in absorbance")
    quit()

#Normalize data
y = NormalizeData(jcamp_dict['y'])
x = jcamp_dict['wavenumbers']

#Interpolate between 500 and 3000 cm^-1
#Interpolation range can be changed
f = interpolate.interp1d(x, y)
newx = np.linspace(500,3000,2500)
newy = f(newx)
'''


# Request parameters
params = {}

# Modify this to choose what functional groups will be included in a molecule descriptor
substructs_smarts = {'ALCOHOL': '[#6][OH]',
                    'ALDEHYDE':'[CX3H1](=O)[#6]',
                    'CARBOXYLIC ACID':'[CX3](=O)[OX2H1]',
                    'AMIDE':'[NX3][CX3](=[OX1])[#6]',
                    'KETONE':'[#6][CX3](=O)[#6]',
                    'ETHER':'[OD2]([#6])[#6]',
                    'NITRO':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
                    'ACYLHALIDE':'[CX3](=[OX1])[F,Cl,Br,I]',
                    'NITRILE':'[NX1]#[CX2]',
                    'PSAMINE':'[NX3;H2,H1;!$(NC=O)]',
                    'NITRO':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
                    'BENZENE':'c1ccccc1'}

# substructs_smarts contents that have been turned into mol datatypes for processing
substructs = {}

fg = []

# For each group in substructs_smarts, convert it to a mol datatype and copy it to substructs
for i in substructs_smarts:
    substructs[i] = Chem.MolFromSmarts(substructs_smarts[i])

# Assemble parameters for request
params['GetInChI'] = 'C' + FILE_NAME

# Get functional group information
inchi = requests.get(URL, params=params).text

if inchi == "":
    print("Could not get an InChI, May need to be manually done")

try:
    # Convert inchi to molecule
    mol = Chem.MolFromInchi(inchi, treatWarningAsError=True)   

    # Reset the temporary dictionary
    functional_groups = {}

    # Find number of functional group matches
    for i in substructs:
        match = len(mol.GetSubstructMatches(substructs[i]))
        functional_groups[i] = match
    
    # Add to molecule descriptor to functional group log
    fg.append(functional_groups)

    # Creating each compound's picture, optional, used for testing
    # Draw.MolToFile(mol,config.ROOT_PATH+"/data/images/"+cas+'.png')

except Exception as e:
    print(e)
    print(inchi)

print(fg)

quit()
#Plot
plt.plot(x, y)
plt.plot(newx, newy,c='red')
plt.title(jcamp_dict['title'])
plt.xlabel('cm^-1')
plt.ylabel('Absorbance')
#plt.xticks(newx)
plt.grid()
plt.show()