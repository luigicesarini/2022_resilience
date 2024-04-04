#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python
"""
Script containing bias methods
"""
import os
import numpy as np 
import xarray as xr 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings('ignore')

os.chdir("/mnt/beegfs/lcesarini//2022_resilience/")


class EvaluatorBiasCorrection:
    """
    Class used to evaluate bias correction methods
    Parameters
    ----------
    obs : np.ndarray, defaults to None
        Name of the reference object used to compute the bias
    """
    
    def __init__(self,obs,mod) -> None:


        self.obs = obs
        self.mod = mod


    def PBias(self):
        """
        Method used to compute the bias
        
        Parameters
        ----------
        obs : np.ndarray, defaults to None
            Name of the reference object used to compute the bias

        mod : np.ndarray, defaults to None
            Name of the corrected object 

        
        Returns
        -------
        returns numpy array containing the percentage bias between the 
        reference object and thte corrected object

        Examples
        --------

        bsc=EvaluatorBiasCorrection(ref,mod)\n
        x=bsc.PBias()\n
        plt.plot(x)
        """

        if not isinstance(self.obs,np.ndarray):
            obs = np.array(self.obs).reshape(-1,1)

        if not isinstance(self.mod,np.ndarray):
            mod = np.array(self.mod).reshape(-1,1)

        bias = ((mod.reshape(-1,1) - obs) / obs) * 100

        bias = np.where(np.isfinite(bias),bias,np.nan)

        return bias


    def EQM(self):
        """
        Apply quantile mapping
        Parameters
        ----------
        key : str, defaults to None
            Groupby key, which selects the grouping column of the target.  

        Returns
        -------
        

        Examples
        --------
        
        
        """
        ecdf_obs=ECDF(self.obs.reshape(-1))
        ecdf_mod=ECDF(self.mod.reshape(-1))

        return  ecdf_obs,ecdf_mod

    def pltECDF(self, ecdf_obs, ecdf_mod, thr:float = 0.2):
        """
        Plots the empiricl cumulative distribution function for the observation and the data from model
        Parameters
        ----------
        thr : float, defaults to None
            Threshold used to remove drizzle effects. Modelled rain probability equal to the same  
            probability of observing 0.2 mm/hr
        
        ecdf_obs: ECDF of the observation as returned from EQM()
        
        ecdf_mod: ECDF of the mdoel as returned from EQM()
        
        Returns
        -------
        
        

        Examples
        --------
        
        
        """

        threshold_model=np.percentile(self.obs[self.obs > 0], ecdf_obs(0.2))

        plt.plot(ecdf_mod.x,
                 ecdf_mod.y,'-r',label='mod')
        plt.plot(ecdf_obs.x,
                 ecdf_obs.y,'-g',label='obs')
        # plt.plot(ecdf(ds_sliced_thr[ds_sliced_thr > 0])[0],
        #          ecdf(ds_sliced_thr[ds_sliced_thr > 0])[1],'-r',label='mod')
        # plt.plot(ecdf(prec_o_over[prec_o_over > 0])[0],
        #          ecdf(prec_o_over[prec_o_over > 0])[1],'-g',label='obs')
        plt.legend()
        plt.savefig("ecdf.png")    
        plt.close()



if __name__=="__main__":

    a = np.ones(shape=(100,1))+np.random.uniform(0,0.1,(100,1))
    b = np.ones(shape=(100,1))

    corrector = EvaluatorBiasCorrection(a,b)
    print(corrector.Bias().mean())    
    print(corrector.EQM())    