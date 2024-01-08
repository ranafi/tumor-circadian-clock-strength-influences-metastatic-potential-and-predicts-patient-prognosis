# Scripts for Tumor circadian clock strength influences metastatic potential and predicts patient prognosis
CYCLOPS 2.0 used in Li et al. 2023, "Tumor circadian clock strength influences metastatic potential and predicts patient prognosis in Luminal A breast cancer."

## CYCLOPS.jl
The "CYCLOPS.jl" file contains all the code to pre-process data, train on processed data, and calculate cosinors of best fit for all transcripts using CYCLOPS sample phase predictions. This module contains both CYCLOPS model algorithms: the original CYCLOPS ('Order'), and the improved CYCLOPS 2.0 ('Covariates').

## Scripts for Human Breast Cancer
Contains the script used to run "CYCLOPS.jl" on female human breast samples from TCGA, GTEx, and the University of Manchester, UK.

## Mortality within 5 Years
Contains the script used to model patient mortality as a function of clinical data, transcript expression, and CYCLOPS sample magnitude.

## Scripts for Benchmarking
Contains scripts used to calculate rhythmic and non-rhythmic parameters of BA11 data (Chen et al. 2016) and Lung data (Bossé et al. 2012). Calculated parameters were used to generate synthetic data with varying levels of batch influence, sample collection time bias, or both. CYCLOPS 2.0 was benchmarked against CYCLOPS with ComBat adjusted data (method described in Johnson et al. 2007).
