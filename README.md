# DeepIR, Application of 1-D CNN's to FTIR Spectral Analysis.
 

This repository accompanies the paper "FTIR Spectral Analysis using 1 Dimensional Convolutional Neural Networks"

## Overview
We present DeepIR, a novel approach to automated Fourier Transform IR (FTIR) spectral analysis using a 1 Dimensional Convolutional Neural Network (1-D CNN).

To set up this repository on your machine, do the following:
1. Install the required libraries and set up tensorflow in a conda enviroment.
	- Run ``pip install -r requirements.txt``
2. Run `setup.py`
	- This will create a directory for the datasets and models to be stored in
3. Download the NIST Species List
	- This file may be found at https://webbook.nist.gov/chemistry/download/. 
	- Download and extract the contents into the `data` directory
4. Run `species_list_to_csv.py`
	- This will convert the species list into a csv file
5. Run ``NIST_scraper.py``
	- This will scrape the NIST database for the IR spectrum files for all of the molecules contained in ``species.csv``, putting them inside ``data/IRS``.
6. Run ``jdx_to_csv.py``
	- This will read the contents of ``data/IRS`` and generate ``CAS-IRS.csv`` file
7. Run ``fragmentation.py``
	- This will read the contents of `CAS-IRS.csv` and generate `molecular_descriptors.csv`,  a file containing the results of fragmenting molecules to find functional groups

Once these steps have been completed, setup is now complete.

DeepIR trains individual models that are specialized to detect specific groups.
To train a complete set of models for every functional group currently supported, run `batch_trainer.py`. The results of training will be put inside the `data` directory with each model getting an individual directory.
- Statistics from training are also put inside the model directory. These include:
	- Confusion Matrix, Model topology, training log, ROC curve, and test metrics.

To specify custom training parameters such as epoch size and learning rate, see `start_run` inside of `GROUP_DETECTION.py` 
- Two parameters of this function that you may not be familiar with are `mu` and `o`.
	- DeepIR utilizes data augmentation to make up for disproportionate representation of certain groups within the NIST dataset. This is done by adding noise sampled from a normal distribution to a random selection of the IR spectra. `mu` and `o` specify the mean and standard deviation of the normal distribution respectively.

Once these models have been trained, you may want to test them on individual or a set of molecules. For this, you can use either `batch_test.py` or `saliency.py`.

`batch_test.py` is well suited for testing a model on a set of molecules. Simply give the program a list of molecules specified by their CAS identifiers and the program will run through each molecule to generate predictions and saliency maps.

`saliency.py` works well for testing a model on an individual molecule. Here is an example of how to test a model on Ethanol (CAS_ID = 64-17-5)
- `python saliency.py 64-17-5`

Regardless, you will have to select a model to test by editing `model_PATH` within `saliency.py`. This will be the path of the folder containing the output of training. For example, it may be:
- `/data/ALCOHOL_RUN/ALCOHOL_MODEL`

You may find the resulting saliency maps in `data/saliency_maps`.
- Here, blue bars indicate the model has put extra attention at the specified location on the spectrum. Usually this indicates that the model has learnt the position that a peak should be located for a corresponding functional group.