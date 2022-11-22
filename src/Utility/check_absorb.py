# This program ensures that every IR spectra file is in units of absorption and wavenumber

# Do this so that the data is consistent, we will be using absorption and wavenumber (1/cm)


import config
import os
from jcamp import JCAMP_reader,JCAMP_calc_xsec

# Path to IRS folder

PATH = config.ROOT_PATH+'/data/IRS'

for filename in os.listdir(PATH):
    f = os.path.join(PATH, filename)
    # checking if it is a file
    if os.path.isfile(f):
        raw = JCAMP_reader(str(PATH)+"/"+(filename))

        JCAMP_calc_xsec(raw)

        # Ensure that the data is in absorbance
        if raw['yunits'].lower() != 'absorbance':
            print(f"Error, {filename} not in absorbance")
