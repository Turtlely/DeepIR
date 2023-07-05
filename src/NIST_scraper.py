#Scrape IRS from NIST database
import requests
import config
import pandas as pd

PATH = config.ROOT_PATH
NIST_URL = config.NIST_URL

#Scrape a list of molecules from the NIST
df = pd.read_csv(PATH+"/data/species.csv")
CAS_LIST = df['CAS']

# Directory to save the IR spectra in
IR_dir = PATH+"/data/IRS/"

# For each CAS ID in the list
for index,cas_id in enumerate(CAS_LIST,start=0):
    
    # GET request parameters
    params={'JCAMP': 'C'+cas_id, 'Type': 'IR', 'Index': 0}	

    # Make a GET request
    response = requests.get(NIST_URL, params=params)

    # If no spectrum is found, skip this CAS ID
    if response.text == '##TITLE=Spectrum not found.\n##END=\n':
        continue
    
    # Create a .jdx file from the results of the GET request
    with open(IR_dir+cas_id+'.jdx','wb') as spec:
        spec.write(response.content)

    # Print the number of the molecule we are at
    print(index)