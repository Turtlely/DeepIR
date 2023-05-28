# This script highlights parts of the IR spectrum that are responsible for a models predictions
# This script runs predictions on indiviudal molecules

import config
import tensorflow as tf
from tensorflow.keras.models import load_model
from jcamp import JCAMP_reader,JCAMP_calc_xsec
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.cm as cm

# Required to allow training to not freeze on first epoch
CONFIG = ConfigProto()
CONFIG.gpu_options.allow_growth = True
session = InteractiveSession(config=CONFIG)

# Display GPU's available
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Model Path
model_PATH = config.ROOT_PATH + "/data/ALCOHOL_RUN/ALCOHOL_MODEL"

# Ask user for the CAS number
import sys
CAS= sys.argv[1]

# Spectrum Path
IRS_PATH = config.ROOT_PATH + "/data/IRS/"+CAS+".jdx"

# Load in the model
model = load_model(model_PATH)

# Load in IRS file
jcamp_dict = JCAMP_reader(IRS_PATH)

# Convert x units to 1/cm.
if (jcamp_dict['xunits'].lower() in ('1/cm', 'cm-1', 'cm^-1')):
    jcamp_dict['wavenumbers'] = jcamp_dict['x']          ## note that array() always performs a copy
elif (jcamp_dict['xunits'].lower() in ('micrometers', 'um', 'wavelength (um)')):
    jcamp_dict['wavenumbers'] = 10000.0 / jcamp_dict['x']
elif (jcamp_dict['xunits'].lower() in ('nanometers', 'nm', 'wavelength (nm)')):
    x = jcamp_dict['x'] / 1000.0
    jcamp_dict['wavenumbers'] = 10000.0 / x
else:
    print("Error in converting x values.")
    quit()

# Update xunits variable
jcamp_dict['xunits'] = "1/CM"

# Convert y units to percentages

# If units are in transmittance %, fix unphysical values and move on
# If units are in absorbance, convert to transmittance %, fix unphysical values, and move on

if jcamp_dict['yunits'].upper() == "TRANSMITTANCE":
    # Correct for unphysical values
    jcamp_dict['y'][jcamp_dict['y'] > 1.0] = 1
    jcamp_dict['y'][jcamp_dict['y'] < 0.0] = 0

elif jcamp_dict['yunits'].upper() == "ABSORBANCE":
    # Convert to transmittance %
    # Formula is transmittance % = 10^(-absorbance units)
    jcamp_dict['y'] = 10**(-jcamp_dict['y'])

    # Correct for unphysical values
    jcamp_dict['y'][jcamp_dict['y'] > 1.0] = 1
    jcamp_dict['y'][jcamp_dict['y'] < 0.0] = 0

else:
    print("Unknown y unit. Aborting...")
    quit()

# Interpolation stuff
# Convert % Transmittance to % Absorbance, which the model was trained for
y = 1-jcamp_dict['y'] 
x = jcamp_dict['wavenumbers']
jcamp_dict['yunits'] = '% ABSORBANCE'

# Interpolate between 1000 and 3400 cm^-1

# Interpolation range can be changed
f = interpolate.interp1d(x, y)

# Set interpolation range to be for the entire spectrum
newx = np.linspace(1300,3700,4800)
newy = f(newx)

# Plot transmittance

plt.plot(x,y,c='black')

# Reverse the x axis for easier reading
plt.xlim(max(x), min(x))

newy = newy.reshape(1,4800,1)

# Print a summary of the model architecture
print(model.summary())

# Model Prediction
y_pred = model.predict(newy)

print("RESULTS FOR",jcamp_dict['title'])
print("Model output: ",y_pred)

if y_pred >= 0.5:
    print("Model Decision: Presence predicted")
else:
    print("Model Decision: Presence not predicted")


# Saliency map generation
images = tf.Variable(newy, dtype=float)

# Generate gradient
with tf.GradientTape() as tape:
    pred = model(images, training=False)

# Calculate the gradient
grads = tape.gradient(pred, images)

# Absolute value of the gradient
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=2)[0]

# normalize the gradient to range between 0 and 1
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

# Smoothing function to highlight peaks in model attention
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# Plot the saliency map
for xc in range(len(newx)):
    color=plt.cm.Blues(smooth(grad_eval**2,50))
    #color=plt.cm.Blues(smooth(grad_eval,50))

    plt.axvline(x=newx[xc],ymin=0,ymax=1,color=color[xc],zorder=-1)

# Borders of where the model looks
plt.axvline(x=newx[0],ymin=0,ymax=1,color='red',ls='--')
plt.axvline(x=newx[-1],ymin=0,ymax=1,color='red',ls='--')

# Generate plot of the transmission spectrum
plt.title(f"{jcamp_dict['title']} {jcamp_dict['yunits']} Spectrum")
plt.xlabel(jcamp_dict["xunits"])
#plt.show()
plt.savefig(f"{config.ROOT_PATH}/data/saliency_maps/{CAS}.png")