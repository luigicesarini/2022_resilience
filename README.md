# Status of the main repository used for the resilience project

The purpose of this readme is to depict the status of the repositry, as well as updates, upgrades, and overall ideas to improve the implementation of the work done.


## Status

### Data    

*Remapping*  
Done for 8 CPM models (CNRM,ETH,HCLIMcom,ICTP,MOHC,KIT,KNMI,CMCC), precipitation and wind module.
For:
- Historical
- Evaluation
- Rcp 2.6, 8.5

And for evaluation also done for some WRF model

*Bias Correction*

Done for 8 CPM models (CNRM,ETH,HCLIMcom,ICTP,MOHC,KIT,KNMI,CMCC)
For:
- Historical
- Evaluation
- Rcp 2.6, 8.5

For precipitation with the exception of Evaluation also done for wind module.





*Data Collection*

> Reanalysis
- CMCC VHR-REA 2.2km (precipitation done for triveneto). Problem wth size of files for entire north Italy
- SPHERA 0.2°(Currently getting them from INES)
- COSMO-REA2: all grib files for TP,u,v,T°,Gusts form 2007-2018. Missing 3 days specified in the email. Very hard projection systme. I cant get out of it.

> Projection



> **23/09/2024**
Times are a little dated, might have variation
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

