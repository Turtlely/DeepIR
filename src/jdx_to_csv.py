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

# Counter for how many molecules have been processed
n=0

#Iterate through every file in the directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jdx"): 

        #Try catch because some files will not be interpolatable
        try:
            print(n)
            #Load in the IR spectrum file

            raw = JCAMP_reader(str(directory)[2:-1]+(filename))

            # Make sure that the spectrum data is there. Otherwise, skip to the next molecule
            if raw['npoints'] == 0:
                continue

            # Data preprocessing that the data is consistent, we will be using transmittance and wavenumber (1/cm)

            # Convert x data to 1/cm
            if (raw['xunits'].lower() in ('1/cm', 'cm-1', 'cm^-1')):
                raw['wavenumbers'] = raw['x']    
            elif (raw['xunits'].lower() in ('micrometers', 'um', 'wavelength (um)')):
                raw['wavenumbers'] = 10000.0 / raw['x']
            elif (raw['xunits'].lower() in ('nanometers', 'nm', 'wavelength (nm)')):
                x = raw['x'] / 1000.0
                raw['wavenumbers'] = 10000.0 / x
            else:
                print("Error in converting x values.")
                quit()

            # Ensure that the data is in % absorbance
            if raw['yunits'].lower() == 'absorbance':

                print("Error, in absorbance units. Converting to absorbance % ")
                 # Convert to transmittance %

                # Formula is transmittance % = 10^(-absorbance units)
                # Subtract by 1 to get absorbance %

                raw['y'] = 1-10**(-raw['y'])
                #raw['yunits'] = 'ABSORBANCE'

                # Correct for unphysical values
                raw['y'][raw['y'] > 1.0] = 1
                raw['y'][raw['y'] < 0.0] = 0
            
            elif raw['yunits'].lower() == 'transmittance':
                # Convert % Transmittance to % absorbance
                raw['y'] = 1-raw['y']

                # Correct for unphysical values
                raw['y'][raw['y'] > 1.0] = 1
                raw['y'][raw['y'] < 0.0] = 0

            else:
                print("Unknown y unit. Aborting...")
                continue

            #set %absorbance data
            y = raw['y']
            x = raw['wavenumbers']

            #Interpolate between 1000 and 3400 cm^-1, with 4800 data points                        

            f = interpolate.interp1d(x, y)
            #newx = np.linspace(1300,3700,4800)
            #newx = np.linspace(1000,3700,5400)
            newx = np.linspace(600,3700,6200)
            #newx = np.linspace(600,4000,6800)
            newy = f(newx)

            #Scrape CAS Identification number
            CAS = raw['cas registry no']

            #add molecular entry in, remember to add in the CAS identification number
            data.append([CAS]+ newy.tolist())

            # Update molecule number counter
            n+=1

        except Exception as e:
            #Log errors, most of these will be interpolation errors
            print("Error ",e)
            continue
    else:
        #Skip non jdx files, there should be none in the directory though
        continue

#Print number of entries
print("number of entries", len(data))

#Save data to a csv file
np.savetxt(config.ROOT_PATH+"/data/CAS-IRS-600-4000.csv", data,delimiter =", ",fmt='%s')