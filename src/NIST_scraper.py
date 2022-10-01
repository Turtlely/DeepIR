#Scrape IRS from NIST database
import requests
import config
import pandas as pd

PATH = config.ROOT_PATH
NIST_URL = config.NIST_URL

#Scrape a list of
df = pd.read_csv(PATH+"/data/species.csv")
CAS_LIST = df['CAS']

IR_dir = PATH+"/data/IRS/"

for index,cas_id in enumerate(CAS_LIST,start=0):
    params={'JCAMP': 'C'+cas_id, 'Type': 'IR', 'Index': 0}	
    response = requests.get(NIST_URL, params=params)

    if response.text == '##TITLE=Spectrum not found.\n##END=\n':
        continue
    
    with open(IR_dir+cas_id+'.jdx','wb') as spec:
        spec.write(response.content)
    print(index)