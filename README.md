# DeepIR, Application of 1-D CNN's to IR Spectral Analysis.
 
This repository accompanies the paper "Elucidating Functional Group Presence by Analyzing IR Spectra with 1-Dimensional Convolutional Neural Networks"

## Overview
We present DeepIR, a novel approach to automated IR spectral analysis using a 1 Dimensional Convolutional Neural Network (1-D CNN).

### Setup
To set up this repository on your machine, do the following:
1. cd into the ``src`` directory where all scripts will be found.

2. Install the required libraries and set up tensorflow in a conda enviroment.
	- Run ``pip install -r requirements.txt``
3. Run `setup.py`
	- This will create a directory for the datasets and models to be stored in

Once these steps have been completed, setup is now complete.

To test the code in this repository for yourself, you will need the datasets to train the models on.
You can either get the datasets yourself using the below instructions, or you can download pre-generated datasets from ScienceDB at https://www.scidb.cn/s/22yAVv. This saves a lot of time (a few hours)

### Downloading the datasets yourself
1. Download the NIST Species List
	- This file may be found at https://webbook.nist.gov/chemistry/download/. 
	- Download and extract the contents into the `data` directory
2. Run `species_list_to_csv.py`
	- This will convert the species list into a csv file
3. Run ``NIST_scraper.py``
	- This will scrape the NIST database for the IR spectrum files for all of the molecules contained in ``species.csv``, putting them inside ``data/IRS``.
4. Run ``jdx_to_csv.py``
	- This will read the contents of ``data/IRS`` and generate ``CAS-IRS.csv`` file
5. Run ``fragmentation.py``
	- This will read the contents of `CAS-IRS.csv` and generate `molecular_descriptors.csv`,  a file containing the results of fragmenting molecules to find functional groups

### Downloading pre-generated datasets

Head to this link and download the two following datasets into the ``\data`` folder: "CAS-IRS.csv" and "molecular_descriptors.csv". 

### Training

DeepIR trains individual models that are specialized to detect specific groups.
To train a complete set of models for every functional group currently supported, run `batch_trainer.py`. The results of training will be put inside the `data` directory with each model getting an individual directory.
- Statistics from training are also put inside the model directory. These include:
	- Confusion Matrix, Model topology, training log, ROC curve, and test metrics.

To specify custom training parameters such as epoch count, see `start_run` inside of `GROUP_DETECTION.py` 

Additional training parameters may be found inside `config.py`.

### Testing the models

Once these models have been trained, you may want to test them on individual or a set of molecules. For this, you can use either `batch_test.py` or `saliency.py`.

`batch_test.py` is well suited for testing a model on a set of molecules. Simply give the program a list of molecules specified by their CAS identifiers and the program will run through each molecule to generate predictions and saliency maps.

`saliency.py` works well for testing a model on an individual molecule. Here is an example of how to test a model on Ethanol (CAS_ID = 64-17-5) using the Alcohol model.
- `python saliency.py ALCOHOL 64-17-5`

You may find the resulting saliency maps in `data/saliency_maps`.
- Here, blue bars indicate the model has put extra attention at the specified location on the spectrum. Usually this indicates that the model has learnt the position that a peak should be located for a corresponding functional group.

Finally, you can screen a molecule for all functional groups supported using `predictor.py`. Here is an example of how to use this program with ethanol (CAS_ID = 64-17-5).
- `python predictor.py 64-17-5`

Please note that you will need fully trained models before using `predictor.py`, `saliency.py`, or `batch_test.py`.
