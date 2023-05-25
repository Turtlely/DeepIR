# This script highlights parts of the IR spectrum that are responsible for a models predictions

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

# File Paths
model_PATH = "/home/ryant/Documents/GitHub/DeepIR-2/data/ALCOHOL_RUN/ALCOHOL_MODEL"
IRS_PATH = "/home/ryant/Documents/GitHub/DeepIR-2/data/IRS/67-56-1.jdx"

# Load in the model
model = load_model(model_PATH)

# Load in IRS file
jcamp_dict = JCAMP_reader(IRS_PATH)


#JCAMP_calc_xsec(jcamp_dict)



# Normalize function
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



# Convert x units to 1/cm.

'''
# For now, if the units are not 1/cm, do not use that file
if jcamp_dict['xunits'] != "1/CM":
    print(jcamp_dict['xunits'])
    print("ERROR!")
'''

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

jcamp_dict['xunits'] = "1/CM"

# Convert y units to percentages

# If units are in transmittance %, fix unphysical values and move on
# If units are in absorbance, convert to transmittance %, fix unphysical values, and move on
# Dont deal with absorbance actually, i cant figure out how to convert from AU to transmittance percent...

if jcamp_dict['yunits'].upper() == "TRANSMITTANCE":
    # Correct for unphysical values
    jcamp_dict['y'][jcamp_dict['y'] > 1.0] = 1
    jcamp_dict['y'][jcamp_dict['y'] < 0.0] = 0

elif jcamp_dict['yunits'].upper() == "ABSORBANCE":
    # dont deal with absorbance i hate this
    print("y units in absorbance")
    quit()
    # Convert to transmittance %
    # Formula is transmittance % = 10^(2-absorbance units)
    jcamp_dict['y'] = NormalizeData(np.power(2-jcamp_dict['y'],10))
    jcamp_dict['yunits'] = 'TRANSMITTANCE'

    '''
    # Correct for unphysical values
    jcamp_dict['y'][jcamp_dict['y'] > 1.0] = 1
    jcamp_dict['y'][jcamp_dict['y'] < 0.0] = 0
    '''

'''
if jcamp_dict['yunits'].upper() != 'ABSORBANCE':
    print("Error, not in absorbance")
    print(type(jcamp_dict['y']))

    # if the units are in transmittance, just convert to absorbance
    if jcamp_dict['yunits'].upper() == 'TRANSMITTANCE':
        print("Converting to percent transmittance")
        jcamp_dict['y'] = 1-jcamp_dict['y']

    else:
        print(f"Unknown y unit of {jcamp_dict['yunits']}. Quitting")
        quit()
'''

'''
# Normalize data
y = NormalizeData(jcamp_dict['y'])
x = jcamp_dict['wavenumbers']
'''

y = jcamp_dict['y']
x = jcamp_dict['wavenumbers']

print(y)

# Interpolate between 1000 and 3400 cm^-1
# Interpolation range can be changed
f = interpolate.interp1d(x, y)
newx = np.linspace(1000,3400,4800)
newy = f(newx)

# Plot transmittance

plt.plot(x,y)
plt.plot(newx,newy)

# Reverse the x axis for easier reading
plt.xlim(max(x), min(x))

newy = newy.reshape(1,4800,1)

# Print a summary of the model architecture
print(model.summary())

# Model Prediction
y_pred = model.predict(newy)
print(y_pred)


# Saliency map generation

images = tf.Variable(newy, dtype=float)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    #class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    #print(np.reshape(pred,[]))
    #print(class_idxs_sorted)
    #quit()
    #loss = pred[0][class_idxs_sorted[0]]
    #loss = np.reshape(pred,[])
    loss = pred

# Calculate the gradient
grads = tape.gradient(loss, images)

# Absolute value of the gradient
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=2)[0]

# normalize the gradient to range between 0 and 1
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)


#plt.plot(grad_eval)
#plt.scatter(newx.flatten(),newy.flatten(),c=grad_eval,cmap='viridis',s=3,zorder=5)

# Plot the saliency map
plt.scatter(newx.flatten(),grad_eval,c=grad_eval,cmap='viridis',s=10*grad_eval,zorder=5)

# Generate plot of the transmission spectrum
plt.title(f"{jcamp_dict['title']} {jcamp_dict['yunits']} Spectrum")
plt.xlabel(jcamp_dict["xunits"])
#plt.colorbar()
plt.show()