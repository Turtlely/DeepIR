# DeepIR, Machine learning applied to IR spectroscopy

DeepIR is a research project focused on accurately characterizing molecular properties through IR spectroscopy.

## Installation

This project may be directly cloned from the repository
>$ git clone https://github.com/Turtlely/DeepIR

More specific dependencies will be added later, but here is the general list:
- Numpy
- Pandas
- RDkit
- Scipy
- Keras and Tensorflow

Any of the more recent versions of these dependencies should work fine, but this has not been tested.

## Usage

Inside the "src" directory, will be all the source files for this project.

You may have to make a few directories inside "src" yourself.
- A "models" directory to hold finished models
- A "Datasets" directory to hold the datasets used in this project. These can be found on the associated kaggle profile.

"config.py" contains the path to this directory, this will not need to be changed.

### Dataset Processing

The "Dataset Processing" directory contains scripts responsible for producing the datasets used in this project.
-  IR Scraper scrapes the NIST for ".jdx" spectral data files, and puts them into a datasets directory.
- SMILES Scraper scrapes pubchem for SMILES strings, and puts the resulting file into the datasets folder.
- jdx to dataframe converts the folder of ".jdx" files made by IR Scraper, into a csv file containg interpolated spectral data from 600-3500 cm^-1
- Functional Group Dataset converts the csv file of SMILES strings made by "SMILES Scraper" into a csv file containing a list of the number of each of 31 functional groups documented by this project.
- IR-FG Dataset combines the "Functional Group Dataset" csv file and the "IR Scraper" csv file, into a single csv file that relates the IR spectrum (X) to the functional group dataset (Y)
- IR-FG Stats is a program that documents some statistics about the IR-FG dataset.

The "Training Profiles" directory contains the training profiles for each of the models made so far. 
- These training profiles may have to be moved out of this directory and into the "src" directory to be used.

### Training the model

"Tools.py" is a file that contains a few evaluation tools (Confusion matrices) that are used later.

"Train.py" is the program that will actually train a model, just set the path to the "IR-FG" dataset and set the program to import your wanted model and profile, and it will produce a ".h5" model in the "models" directory.

"Evaluate.py" is the program that runs evaluations on a selected model. You have to set the model path manually, and it will return a confusion matrix as well as a few other evaluation metrics.

"Predict.py" is a program that will actually use a model to make real world predictions. This is a fun tool to play around and test out models with.

### Statistics and Evaluation

As previously mentioned, this project provides a few albiet limited tools to evaluate models, as well as tools to analyze the datasets used.

Inside the "plots_and_statistics" directory, are generated bar charts depicting the composition of the "IR-FG" dataset. This may prove to be useful in analyzing which groups are not well represented or overrepresented.

## Dataset Information
Data for this project was scraped from the NIST Chemistry Webbook.

The accompanying datasets for this project can be found at https://www.kaggle.com/turtlely


