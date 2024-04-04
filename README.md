# Status of the main repository used for the resilience project

The purpose of this readme is to depict the status of the repositry, as well as updates, upgrades, and overall ideas to improve the implementation of the work done.


## Status

### Data    

*Remapping*  
It is currently done for 5 CPM models (CNRM,ETH,HCLIMcom,ICTP,MOHC)
Problems with:
- KIT: problem with corner coordinates
- KNMI: original files seems to be HCLIMcom and not KNMI
- CMCC: problem with missing coordinates

*Data Collection*

> Reanalysis
- CMCC VHR-REA 2.2km (precipitation done for triveneto). Problem wth size of files for entire north Italy
- SPHERA 0.2°(Currently getting them from INES)
- COSMO-REA2: all grib files for TP,u,v,T°,Gusts form 2007-2018. Missing 3 days specified in the email

> Projection


## To Do 03/02/2023

*Data Collection*

> Reanalysis
- CMCC VHR-REA: Download other files
- COSMO-REA2: regrid the grib files to common grid
> Projection
- CMCC: Download data from dds
- CORDEX: Download  RCP4.5


## Ideas

Using generative adversiaral net (GAN) to create credible and reliable climate model realizations


> **21/04/2023**

# **Time needed to run code**

## Bias correct the data for QDM,EQM
*./bias_corr.py*
~ 75 minutes

## Compute metrics from corrected data
*./bc_metrics.py*
~ 32 minutes  

## Evaluate the bias correction methods, by evaluating the metrics used  
*./evaluate_bc.py*
~ 2 minutes

