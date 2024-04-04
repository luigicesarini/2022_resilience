#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python
import os 
import pandas as pd
import numpy as np
import xarray as xr
from .missing_years import findNA

def fillNA(a,b):
    """
    Functions that fills na wer
    """
    return a+b




def return_obj_taylor(LABELS=True,MARKERS=True):
    """
    Function that returns the labels and markers for the taylor diagram

    Parameters
    ----------
    LABELS : bool, optional
        If True, returns the labels, by default True
    MARKERS : bool, optional
        If True, returns the markers, by default True

    Returns
    -------
    lab : list
        list of labels
    mark : dict
        dictionary of markers

    Examples
    --------
    >>> from resilience.utils import return_obj_taylor
    >>> lab,mark = return_obj_taylor(LABELS=True,MARKERS=True)
    
    
    """

    SIZE_MARKER=8

    lab  = [
        "DJF_SPHERA",
        "DJF_STA",
        "DJF_STA_SPHERA",
        "MAM_SPHERA",
        "MAM_STA",
        "MAM_STA_SPHERA",
        "JJA_SPHERA",
        "JJA_STA",
        "JJA_STA_SPHERA",
        "SON_SPHERA",
        "SON_STA",
        "SON_STA_SPHERA",
        "Reference"
            ]

    mark = {
            "DJF_SPHERA": {
                "labelColor": "black",
                "symbol": "o",
                "size": SIZE_MARKER,
                "faceColor": "b",
                "edgeColor": "b",
            },
            "DJF_STA": {
                "labelColor": "k",
                "symbol": "o",
                "size": SIZE_MARKER,
                "faceColor": "g",
                "edgeColor": "g",
            },

            "DJF_STA_SPHERA": {
                "labelColor": "black",
                "symbol": "o",
                "size": SIZE_MARKER,
                "faceColor": "m",
                "edgeColor": "m",
            },
            "MAM_SPHERA": {
                "labelColor": "k",
                "symbol": "^",
                "size": SIZE_MARKER,
                "faceColor": "b",
                "edgeColor": "b",
            },
            "MAM_STA": {
                "labelColor": "black",
                "symbol": "^",
                "size": SIZE_MARKER,
                "faceColor": "g",
                "edgeColor": "g",
            },
            "MAM_STA_SPHERA": {
                "labelColor": "k",
                "symbol": "^",
                "size": SIZE_MARKER,
                "faceColor": "m",
                "edgeColor": "m",
            },
            "JJA_SPHERA": {
                "labelColor": "black",
                "symbol": "s",
                "size": SIZE_MARKER,
                "faceColor": "b",
                "edgeColor": "b",
            },
            "JJA_STA": {
                "labelColor": "k",
                "symbol": "s",
                "size": SIZE_MARKER,
                "faceColor": "g",
                "edgeColor": "g",
            },
            "JJA_STA_SPHERA": {
                "labelColor": "black",
                "symbol": "s",
                "size": SIZE_MARKER,
                "faceColor": "m",
                "edgeColor": "m",
            },
            "SON_SPHERA": {
                "labelColor": "k",
                "symbol": "P",
                "size": SIZE_MARKER,
                "faceColor": "b",
                "edgeColor": "b",
            },
            "SON_STA": {
                "labelColor": "black",
                "symbol": "P",
                "size": SIZE_MARKER,
                "faceColor": "g",
                "edgeColor": "g",
            },
            "SON_STA_SPHERA": {
                "labelColor": "k",
                "symbol": "P",
                "size": SIZE_MARKER,
                "faceColor": "m",
                "edgeColor": "m",
            },
            "Reference": {
                "labelColor": "k",
                "symbol": "+",
                "size": SIZE_MARKER,
                "faceColor": "r",
                "edgeColor": "r",
            },
    }
    if (LABELS) & (MARKERS):
        return lab,mark
    elif (LABELS) & (not MARKERS):
        return mark
    elif (not LABELS) & (MARKERS):
        return lab




def return_obj_taylor_sph(LABELS=True,MARKERS=True):
    """
    Function that returns the labels and markers for the taylor diagram
    for the entire area covered by SPHERA

    Parameters
    ----------
    LABELS : bool, optional
        If True, returns the labels, by default True
    MARKERS : bool, optional
        If True, returns the markers, by default True

    Returns
    -------
    lab : list
        list of labels
    mark : dict
        dictionary of markers

    Examples
    --------
    >>> from resilience.utils import return_obj_taylor
    >>> lab,mark = return_obj_taylor(LABELS=True,MARKERS=True)
    
    
    """

    SIZE_MARKER=8

    lab  = [
        "DJF",
        "MAM",
        "JJA",
        "SON",
        "Reference"
            ]

    mark = {
            "DJF": {
                "labelColor": "black",
                "symbol": "o",
                "size": SIZE_MARKER,
                "faceColor": "b",
                "edgeColor": "k",
            },

            "MAM": {
                "labelColor": "k",
                "symbol": "^",
                "size": SIZE_MARKER,
                "faceColor": "r",
                "edgeColor": "k",
            },

            "JJA": {
                "labelColor": "black",
                "symbol": "s",
                "size": SIZE_MARKER,
                "faceColor": "g",
                "edgeColor": "k",
            },

            "SON": {
                "labelColor": "k",
                "symbol": "P",
                "size": SIZE_MARKER,
                "faceColor": "m",
                "edgeColor": "k",
            },
            "Reference": {
                "labelColor": "k",
                "symbol": "+",
                "size": SIZE_MARKER,
                "faceColor": "r",
                "edgeColor": "r",
            },
    }
    if (LABELS) & (MARKERS):
        return lab,mark
    elif (LABELS) & (not MARKERS):
        return mark
    elif (not LABELS) & (MARKERS):
        return lab





if __name__ == "__main__":
    print("Test filling NAs function")
    name_station="TN_0186"
    dates = np.array(pd.read_csv(f"data/dates/{name_station}.csv")['date'],
                     dtype=np.datetime64)

    dates_missing = findNA(array_date=dates)


    print(dates_missing)