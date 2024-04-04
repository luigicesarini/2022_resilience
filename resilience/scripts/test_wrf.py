#! /home/lcesarini/miniconda3/envs/colorbar/bin/python

import sys 

print(type(sys.argv[1]))
# import pygrib
# import os
# import numpy as np
# import urllib
# import matplotlib.pyplot as plt

# import sys
# sys.path.append('/uufs/chpc.utah.edu/common/home/u0553130/pyBKB_v2')

# URL = 'https://pando-rgw01.chpc.utah.edu/hrrr/sfc/20180425/hrrr.t00z.wrfsfcf00.grib2'
# FILE = URL.split('/')[-1]
# print('File URL: %s' % URL)
# print('File Name: %s' % FILE)

# urllib.request.urlretrieve(URL, FILE)

# grbs = pygrib.open(FILE)

# for WRF data processing (`xwrf` Dataset and DataArray accessor)
# import xarray as xr
# import xwrf
# # for unit conversion (`pint` DataArray accessor)
# import pint_xarray
# # for `dask`-accelerated computation
# from distributed import LocalCluster, Client
# # for numerics
# import numpy as np
# # for visualization
# import holoviews as hv
# import hvplot
# import hvplot.xarray
# import intake

# hv.extension('bokeh') # set plotting backend
# xr.set_options(display_style="text")
# cat = intake.open_catalog("https://raw.githubusercontent.com/xarray-contrib/xwrf-data/main/catalogs/catalog.yml")
# ssp5_ds = cat["xwrf-sample-ssp585"].to_dask()
# ssp5_ds

# ssp5_ds = ssp5_ds.xwrf.postprocess()
# import intake_xarray
# import zarr

# zarr.open(cat["xwrf-sample-ssp585"],engine='pynio').to_dask()
# intake_xarray.xzarr.ZarrSource("/home/lcesarini/tp3_max_baseline.nc",engine='store').to_dask()