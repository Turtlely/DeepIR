# DeepIR, Machine learning applied to IR spectroscopy

DeepIR is a research project focused on accurately characterizing molecular properties through IR spectroscopy.

As of 6/27/2022, only a dataset has been prepared under this project.

## Installation

This project may be directly cloned from the repository
>$ git clone https://github.com/Turtlely/DeepIR

## Usage
Currently, all you can do is download the dataset (*dataset.csv*)

If you wish to, you can curate your own dataset of infared spectra or molecular adjacency matrices by modifying *"dataScraper.ipynb"*.

Programs are written in Jupyter Notebook

## Dataset Information
Data for this project was scraped from the NIST Chemistry Webbook.

The "Datasets" directory contains data used in this project.
- dataset.csv is a dataset of over 10k molecules that contains:
 1. IR absorbance measurements from 600 to 3500cm^-1.
 Each IR spectrum contains 12,000 collection points, produced via interpolation of experimental data.
 
 2. Flattened bond adjacency matrices. These have already been padded with zeros, and should all be of the same shape.
- Mol_Folder is a folder containing *.mol* files for various compounds in the NIST database.
- IR_spectra is a folder containing IR spectrum data for various compounds, in JCAMP *(.jdx)* file format
- "species.txt" is a text file which was downloaded from the NIST Chemistry Webbook. It contains a list of every compound in the NIST compound database
      
