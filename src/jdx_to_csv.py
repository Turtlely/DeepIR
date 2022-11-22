#Load all IRS data into a csv, 4800 columns and n rows
#Each row represents one molecule, and will be tagged with a CAS as the first entry in each row
import pandas as pd
import config
import os
from jcamp import JCAMP_reader,JCAMP_calc_xsec
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import interpolate
import numpy as np

#Path to the IRS folder
IRS_PATH = config.ROOT_PATH+"/data/IRS/"

#Encoded path for iterating
directory = os.fsencode(IRS_PATH)

#List to store the data
data = []

n=0

#Function to normalize spectra
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#Iterate through every file in the directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jdx"): 
        #Try catch because some files will not be interpolatable
        try:
            print(n)
            #Load in the IR spectrum file
            raw = JCAMP_reader(str(directory)+"/"+(filename))

            # Do this so that the data is consistent, we will be using absorption and wavenumber (1/cm)
            JCAMP_calc_xsec(raw)

            # Ensure that the data is in absorbance
            if raw['yunits'].lower() != 'absorbance':
                print("Error, not in absorbance")
                continue
            
            # Make sure that the spectrum data is there
            if raw['npoints'] == 0:
                continue

            #Normalize absorption data
            y = NormalizeData(raw['y'])
            x = raw['wavenumbers']

            #Interpolate between 1000 and 3400 cm^-1, with 4800 data points
            f = interpolate.interp1d(x, y)
            newx = np.linspace(1000,3400,4800)
            newy = f(newx)

            #Scrape CAS Identification number
            CAS = raw['cas registry no']

            #add molecular entry in, remember to add in the CAS identification number
            data.append([CAS]+ newy.tolist())
            n+=1
        except Exception as e:
            #Log errors, most of these will be interpolation errors
            print("Error ",e)
            continue
    else:
        #Skip non jdx files, there should be none in the directory though
        continue

#Print number of entries
print("number of entrys", len(data))

#Save data to a csv file
np.savetxt(config.ROOT_PATH+"/data/CAS-IRS.csv", data,delimiter =", ",fmt='%s')