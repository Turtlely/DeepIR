# Create molecular_descriptors.csv
# This is a spreadsheet indexed by individual molecule CAS ID's, where each column corresponds to a functional group.
# If a functional group is present in a molecule, the corresponding cell will be filled with 1, otherwise it will be 0

# Imports
import config
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Fragments
import numpy as np
from rdkit.Chem import Draw

# CSV path
CSV_PATH = config.ROOT_PATH+"/data/CAS-IRS.csv"

# NIST url
URL = "https://webbook.nist.gov/cgi/cbook.cgi"

# Load in csv
df = pd.read_csv(CSV_PATH,header=None,index_col=0,usecols=[0])

# Get all the CAS id's
CAS_list = df.index.values.tolist()

# How many molecules we're working with
print(len(CAS_list))

# Request parameters
params = {}

# List of molecule descriptors for each molecule
fg = []

# Produce the dictionary of functional groups to be screened
# SMARTS strings from here
# https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html#C

# Modify this to choose what functional groups will be included in a molecule descriptor

substructs_smarts = {'ALCOHOL': '[OX2H][CX3,!O]',
                    'ALDEHYDE':'[CX3H1](=O)[#6]',
                    'PAMIDE':'[NX3;H2][CX3](=[OX1])[#6]',
                    'SAMIDE':'[#6][NX3;H1][CX3](=[OX1])[#6]',
                    'TAMIDE':'[#6][NX3]([#6])[CX3](=[OX1])[#6]',
                    'KETONE':'[#6][CX3](=O)[#6]',
                    'ETHER':'[OD2]([#6])[#6]',
                    'NITRO':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
                    'ACYLHALIDE':'[CX3](=[OX1])[F,Cl,Br,I]',
                    'NITRILE':'[NX1]#[CX2]',
                    'ALKENE':'[CX3]=[CX3]',
                    'ALKANE':'[CX4;H0,H1,H2,H4]',
                    'ESTER':'[#6][CX3](=O)[OX2H0][#6]',
                    'PRIMARY_AMINE':'[#7;H2;X3][#6]',
                    'SECONDARY_AMINE':'[#6][#7;H1;X3][#6]',
                    'TERTIARY_AMINE':'[#6][#7;X3]([#6])[#6]',
                    'ARENE':'c',
                    'CARBOXYLIC_ACID':'[#6,H][CX3](=O)[OX2H1]'}

# substructs_smarts contents that have been turned into mol datatypes for processing
substructs = {}

# For each group in substructs_smarts, convert it to a mol datatype and copy it to substructs
for i in substructs_smarts:
    substructs[i] = Chem.MolFromSmarts(substructs_smarts[i])

# Search through every molecule listed in the IR spectra
for n, cas in enumerate(CAS_list):
    # Logging purposes
    print(n, cas)

    # Assemble parameters for request
    params['GetInChI'] = 'C' + cas

    # Fetch InChI
    inchi = requests.get(URL, params=params).text

    print(inchi)
    # If an InChI is not retrieved, put a NaN in the place, and then drop all nan's later
    if inchi == "":
        functional_groups = {}
        fg.append(functional_groups)
        print("NaN Found")
        continue
    
    # Try to find functional group hits, count, and log
    # Errors may occur
    try:
        # Convert inchi to molecule
        mol = Chem.MolFromInchi(inchi,treatWarningAsError=True)   

        # Reset the temporary dictionary
        functional_groups = {}

        # Find number of functional group matches
        for i in substructs:
            # Return 1 if a functional group is found, otherwise 0
            if len(mol.GetSubstructMatches(substructs[i])) != 0:
                functional_groups[i] = 1
            else:
                functional_groups[i] = 0

        # Add to molecule descriptor to functional group log
        fg.append(functional_groups)
        
        # Creating each compound's picture, optional, used for testing
        Draw.MolToFile(mol,config.ROOT_PATH+"/data/images/"+cas+'.png')

    except Exception as e:
        print(e)
        print(inchi)

        functional_groups = {}
        fg.append(functional_groups)
        continue


# Create the dataframe to store molecule descriptors
md = pd.DataFrame(fg,index=CAS_list)

md.index.name='CAS'

# Save the dataframe as a csv
md.to_csv(config.ROOT_PATH+"/data/molecule_descriptors.csv")
