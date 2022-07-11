model_PATH='/Users/ryantang/Documents/GitHub/DeepIR/models/NH0counter.h5'
IR_PATH = '/Users/ryantang/Documents/GitHub/DeepIR/tests/aniline.jdx'

from keras.models import load_model, Model
from jcamp import JCAMP_calc_xsec, JCAMP_reader
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

model = load_model(model_PATH)

data = JCAMP_reader(IR_PATH)
JCAMP_calc_xsec(data)

#Interpolate spectra between 600 and 3500cm^-1
x = np.linspace(600,3500,num=12000)
spectrum=[]
if(data['yunits'].lower()=='absorbance'):
    f = interp1d(data['wavenumbers'],data['y'],bounds_error=False,fill_value=data['y'][-1])
    spectrum = f(x)

elif (data['yunits'].lower() == 'transmittance'):
    f = interp1d(data['wavenumbers'],data['absorbance'],bounds_error=False,fill_value=data['y'][-1])
    spectrum = f(x)
else:
    print("Incompatible units")


data = pd.DataFrame(columns=np.arange(0,12000))
data.loc[0]=spectrum

X = data.iloc[:,:12000]

print(X)
print("")

#pred = np.round(model.predict(X))

#pred_df = pd.DataFrame(pred,columns=np.arange(0,6))

#print("Tertiary amines detected: ",pred_df.idxmax(axis=1)[0])

model.summary()

feature_map = Model(inputs=model.inputs,outputs=model.layers[5].output)
#feature_map.summary()
feature_maps = feature_map.predict(X,batch_size=1)

print(feature_maps.shape)

for x in range(feature_maps.shape[2]):
    for y in range(feature_maps.shape[0]):
        plt.plot(feature_maps[y,:,x],color='red',linewidth=2)

plt.plot(spectrum,label='Spectrum')

#plt.plot(feature_maps)
plt.legend(loc="upper right")
plt.show()
