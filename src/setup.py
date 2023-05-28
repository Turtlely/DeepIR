# Sets up the required directory structure for this project

import os
import config

# Make the directory for data to go

os.mkdir(config.ROOT_PATH+"/data")
os.mkdir(config.ROOT_PATH+"/data/IRS")
os.mkdir(config.ROOT_PATH+"/data/saliency_maps")