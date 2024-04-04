#! /home/lcesarini/miniconda3/envs/colorbar/bin/python
import os
import argparse
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import metview as mv
import seaborn as sns
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import matplotlib as mpl
from random import sample
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from shapely.geometry import mapping
from cartopy import feature as cfeature
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import *

os.chdir("/home/lcesarini/2022_resilience/")

PLOT_ECDF=True
PLOT_EPDF=False

PF="/mnt/data/RESTRICTED/CARIPARO"
# shp_triveneto = gpd.read_file("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
# shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]

# cmap_f,cmap_i,cmap_q=get_palettes()
# lvl_f,lvl_i,lvl_q=get_levels()

eth_or=xr.open_dataset(f"{PF}/datiDallan/CPM_ETH_Italy_2000-2009_pr_hour.nc").load()
eth_re=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/*.nc").load()
eth_nn=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/nn/*.nc").load()
# eth_la=xr.open_mfdataset(f"{PF}/DATA_FPS/ECMWF-ERAINT/ETH/CPM/pr/laf/*.nc").load()

eth_or_tri=eth_or.where((eth_or.lon > 10.38) & (eth_or.lon < 13.1) & (eth_or.lat > 44.7) & (eth_or.lat < 47.1), drop=True)
eth_re_tri=eth_re.where((eth_re.lon > 10.38) & (eth_re.lon < 13.1) & (eth_re.lat > 44.7) & (eth_re.lat < 47.1), drop=True)
eth_nn_tri=eth_nn.where((eth_nn.lon > 10.38) & (eth_nn.lon < 13.1) & (eth_nn.lat > 44.7) & (eth_nn.lat < 47.1), drop=True)
# eth_la_tri=eth_la.where((eth_la.lon > 10.38) & (eth_la.lon < 13.1) & (eth_la.lat > 44.7) & (eth_la.lat < 47.1), drop=True)
del eth_re,eth_or,eth_nn#,eth_la_tri

print("Finished loading the data")

"""METADATA Stations"""
meta_station=pd.read_csv("meta_station_updated_col.csv",index_col=0)
meta_station

reg=ccrs.RotatedPole(pole_latitude=43.0, pole_longitude=-170.0)
XY=reg.transform_points(ccrs.CRS("WGS84"),meta_station.lon,meta_station.lat)[:,:2]

a_99_stat=np.sort(eth_or_tri.pr.sel(rlon=XY[:,0],rlat=XY[:,1],method='nearest').isel(time=eth_or_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
b_99_stat=np.sort(eth_re_tri.pr.sel(lon=[x for x in meta_station.lon],lat=[y for y in meta_station.lat],method='nearest').isel(time=eth_re_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
c_99_stat=np.sort(eth_nn_tri.pr.sel(lon=[x for x in meta_station.lon],lat=[y for y in meta_station.lat],method='nearest').isel(time=eth_nn_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))

print("Finished selecting cells with only stations")

q99=eth_or_tri.pr.quantile(q=0.99).item()

if PLOT_ECDF:   
    print("Plotting ECDF")


    f,ax=plt.subplots(1,2,figsize=(10,5),constrained_layout=True, squeeze=True)
    ax[0].plot(a_99_stat[a_99_stat>q99],
            (np.arange(0,a_99_stat[a_99_stat>q99].shape[0])+1)/a_99_stat[a_99_stat>q99].shape[0],
            marker="d",markersize=5,
            color="blue",label='Original',
            alpha=0.5)
    ax[0].plot(b_99_stat[b_99_stat>q99],
            (np.arange(0,b_99_stat[b_99_stat>q99].shape[0])+1)/b_99_stat[b_99_stat>q99].shape[0],
            marker="d",markersize=5,
            color="darkgreen",label='Conservative',
            alpha=0.5)
    ax[0].plot(c_99_stat[c_99_stat>q99],
            (np.arange(0,c_99_stat[c_99_stat>q99].shape[0])+1)/c_99_stat[c_99_stat>q99].shape[0],
            marker="d",markersize=5,
            color="red",label='NNeighbor',
            alpha=0.5)
    ax[0].set_title(f"ECDFs of the 174 cells containing stations")
    ax[0].ticklabel_format(axis='y',style='scientific')
    ax[0].legend()
    # ax[0].set_ylabel("Cumulative probability")
    ax[0].set_xlabel("Rainfall [mm/h]")

    # plt.savefig(f"figures/ecdf_orig_vs_remap_99_station.png")
    # plt.close()

    ax[1].plot(a_99_stat[a_99_stat>q99],
            (np.arange(0,a_99_stat[a_99_stat>q99].shape[0])+1)/a_99_stat[a_99_stat>q99].shape[0],
            marker="d",markersize=5,
            color="blue",label='Original',
            alpha=0.5)
    ax[1].plot(b_99_stat[b_99_stat>q99],
            (np.arange(0,b_99_stat[b_99_stat>q99].shape[0])+1)/b_99_stat[b_99_stat>q99].shape[0],
            marker="d",markersize=5,
            color="darkgreen",label='Conservative',
            alpha=0.5)
    ax[1].plot(c_99_stat[c_99_stat>q99],
            (np.arange(0,c_99_stat[c_99_stat>q99].shape[0])+1)/c_99_stat[c_99_stat>q99].shape[0],
            marker="d",markersize=5,
            color="red",label='NNeighbor',
            alpha=0.5)
    ax[1].set_title(f"Zoom on the extremes")
    # ax.vlines(x=106.88115,ymin=0,ymax=1)
    ax[1].ticklabel_format(axis='y',style='sci')
    ax[1].legend(loc='lower right')
    ax[1].set_ylabel("Cumulative probability")
    ax[1].set_xlabel("Rainfall [mm/h]")
    ax[1].set_xlim((70,110))
    ax[1].set_ylim((0.99998,1.000004))
    plt.savefig(f"figures/ecdf_orig_vs_remap_99_station_with_zoom.png")
    plt.close()



"""
PLOTTING THE TWO EXTREMES
"""

arr_ind=np.argwhere(eth_or_tri.pr.values > 100)#np.unravel_index(np.argwhere(eth_or_tri.pr.values > 100),shape=eth_or_tri.pr.shape)


y_first,x_first=arr_ind[0,1:]
y_secon,x_secon=arr_ind[1,1:]

a_99_s=np.sort(eth_or_tri.pr.isel(rlon=x_first,rlat=y_first,time=eth_or_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
b_99_s=np.sort(eth_re_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=x_first,rlat=y_first).lon.item(),lat=eth_or_tri.pr.isel(rlon=x_first,rlat=y_first).lat.item(),method='nearest').\
               isel(time=eth_re_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
c_99_s=np.sort(eth_nn_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=x_first,rlat=y_first).lon.item(),lat=eth_or_tri.pr.isel(rlon=x_first,rlat=y_first).lat.item(), method='nearest').\
               isel(time=eth_nn_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))

a_99_s2=np.sort(eth_or_tri.pr.isel(rlon=x_secon,rlat=y_secon,time=eth_or_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
b_99_s2=np.sort(eth_re_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=x_secon,rlat=y_secon).lon.item(),lat=eth_or_tri.pr.isel(rlon=x_secon,rlat=y_secon).lat.item(),method='nearest').\
               isel(time=eth_re_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))
c_99_s2=np.sort(eth_nn_tri.pr.sel(lon=eth_or_tri.pr.isel(rlon=x_secon,rlat=y_secon).lon.item(),lat=eth_or_tri.pr.isel(rlon=x_secon,rlat=y_secon).lat.item(), method='nearest').\
               isel(time=eth_nn_tri["time.year"].isin([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009])).values.reshape(-1))



f,ax=plt.subplots(1,2,figsize=(10,5),constrained_layout=True, squeeze=True)
ax[0].plot(a_99_s[a_99_s>q99],
        (np.arange(0,a_99_s[a_99_s>q99].shape[0])+1)/a_99_s[a_99_s>q99].shape[0],
        marker="^",markersize=5,
        color="blue",label='Original',
        alpha=0.85)
ax[0].plot(b_99_s[b_99_s>q99],
        (np.arange(0,b_99_s[b_99_s>q99].shape[0])+1)/b_99_s[b_99_s>q99].shape[0],
        marker="*",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax[0].plot(c_99_s[c_99_s>q99],
        (np.arange(0,c_99_s[c_99_s>q99].shape[0])+1)/c_99_s[c_99_s>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax[0].set_title(f"ECDF of cyan circle")
ax[0].legend(loc='lower right')
ax[0].set_xlim((20,110))
ax[0].set_ylim((0.98,1.0004))
ax[0].set_ylabel("Cumulative probability")
ax[0].set_xlabel("Rainfall [mm/h]")


ax[1].plot(a_99_s2[a_99_s2>q99],
        (np.arange(0,a_99_s2[a_99_s2>q99].shape[0])+1)/a_99_s2[a_99_s2>q99].shape[0],
        marker="^",markersize=5,
        color="blue",label='Original',
        alpha=0.85)
ax[1].plot(b_99_s2[b_99_s2>q99],
        (np.arange(0,b_99_s2[b_99_s2>q99].shape[0])+1)/b_99_s2[b_99_s2>q99].shape[0],
        marker="*",markersize=5,
        color="darkgreen",label='Conservative',
        alpha=0.5)
ax[1].plot(c_99_s2[c_99_s2>q99],
        (np.arange(0,c_99_s2[c_99_s2>q99].shape[0])+1)/c_99_s2[c_99_s2>q99].shape[0],
        marker="d",markersize=5,
        color="red",label='NNeighbor',
        alpha=0.5)
ax[1].set_title(f"ECDF magenta circle")
ax[1].legend(loc='lower right')
ax[1].set_xlim((20,110))
ax[1].set_ylim((0.98,1.0004))
ax[1].set_xlabel("Rainfall [mm/h]")
plt.savefig(f"figures/ecdf_orig_vs_remap_station_2.png")
plt.close()

"""
QQ-PLOT
"""
extr_q=np.arange(0.9999,1,0.00001)
extr_Q=[0.5,0.7,0.85,0.9,0.95,0.99]
def get_nearest(remapped_ds,lon_st,lat_st):
    X_,Y_=convert_coords(X=lon_st,Y=lat_st,
                        reg=ccrs.RotatedPole(pole_longitude=-170,pole_latitude=43),
                        rot=ccrs.CRS("WGS84"),
                )
    cell_re=remapped_ds.sel(lon=lon_st,lat=lat_st,method='nearest')
    cell_or=eth_or_tri.sel(rlon=X_,rlat=Y_,method='nearest')
    return cell_or,cell_re


extr_q_orig,extr_q_rcon,extr_q_r_nn=[],[],[]
extr_Q_orig,extr_Q_rcon,extr_Q_r_nn=[],[],[]
max_orig,max_rcon,max_r_nn=[],[],[]

for idx,(lon_st,lat_st) in tqdm(enumerate(zip(meta_station.lon,meta_station.lat))):

    original,rcon=get_nearest(eth_re_tri,lon_st,lat_st)
    r_nn=get_nearest(eth_nn_tri,lon_st,lat_st)[1]
    # rlaf=get_nearest(eth_la_tri,lon_st,lat_st)[1]



    extr_q_orig.append(np.quantile(original.pr.values.reshape(-1),q=extr_q))
    extr_q_rcon.append(np.quantile(rcon.pr.values.reshape(-1),q=extr_q))
    extr_q_r_nn.append(np.quantile(r_nn.pr.values.reshape(-1),q=extr_q))

    extr_Q_orig.append(np.quantile(original.pr.values.reshape(-1),q=extr_Q))
    extr_Q_rcon.append(np.quantile(rcon.pr.values.reshape(-1),q=extr_Q))
    extr_Q_r_nn.append(np.quantile(r_nn.pr.values.reshape(-1),q=extr_Q))

    max_orig.append(np.nanmax(original.pr.values.reshape(-1)))
    max_rcon.append(np.nanmax(rcon.pr.values.reshape(-1)))
    max_r_nn.append(np.nanmax(r_nn.pr.values.reshape(-1)))


min_diagonal=np.floor(np.min([np.min(extr_q_orig),np.max(max_orig),np.min(max_rcon),np.max(max_r_nn)]))
max_diagonal=np.ceil(np.max([np.min(extr_q_orig),np.max(max_orig),np.min(max_rcon),np.max(max_r_nn)]))

q_bias_cons=(np.array(extr_q_orig)-np.array((extr_q_rcon)))/np.array(extr_q_orig)
q_bias_nn=(np.array(extr_q_orig)-np.array((extr_q_r_nn)))/np.array(extr_q_orig)

Q_bias_cons=(np.array(extr_Q_orig)-np.array((extr_Q_rcon)))/np.array(extr_Q_orig)
Q_bias_nn  =(np.array(extr_Q_orig)-np.array((extr_Q_r_nn)))/np.array(extr_Q_orig)

M_bias_cons=(np.array(max_orig)-np.array((max_rcon)))/np.array(max_orig)
M_bias_nn=(np.array(max_orig)-np.array((max_r_nn)))/np.array(max_orig)

max_bias_cons=np.nanmax(np.abs((np.array(extr_q_orig)-np.array((extr_q_rcon)))/np.array(extr_q_orig)))
max_bias_nn=np.nanmax(np.abs((np.array(extr_q_orig)-np.array((extr_q_r_nn)))/np.array(extr_q_orig)))

MAX_bias_cons=np.nanmax(np.abs((np.array(max_orig)-np.array((max_rcon)))/np.array(max_orig)))
MAX_bias_nn=np.nanmax(np.abs((np.array(max_orig)-np.array((max_r_nn)))/np.array(max_orig)))

mean_bias_cons=np.nanmean(np.abs((np.array(extr_q_orig)-np.array((extr_q_rcon)))/np.array(extr_q_orig)))
mean_bias_nn=np.nanmean(np.abs((np.array(extr_q_orig)-np.array((extr_q_r_nn)))/np.array(extr_q_orig)))


MEAN_bias_cons=np.nanmean(np.abs((np.array(max_orig)-np.array((max_rcon)))/np.array(max_orig)))
MEAN_bias_nn=np.nanmean(np.abs((np.array(max_orig)-np.array((max_r_nn)))/np.array(max_orig)))

fig,axs=plt.subplots(1,1,
                        figsize=(8,8),constrained_layout=True, squeeze=True)

ax=axs
ax.scatter(extr_q_orig,extr_q_rcon,
                # "-",
                marker="d",#markersize=5,
                color="blue",label='Conservative',
                alpha=0.5)
ax.scatter(extr_q_orig,extr_q_r_nn,
                # "-",
                marker="*",#markersize=5,
                color="red",label='NNeighbour',
                alpha=0.5)
# ax.scatter(extr_q_orig,extr_q_rlaf,
#                 # "-",
#                 marker="d",#markersize=5,
#                 color="brown",label='LAF',
#                 alpha=0.5)
ax.scatter(max_orig,max_rcon,
                # "-",
                marker="d",#markersize=5,
                color="cyan",label='Max Conservative',
                alpha=0.5)
ax.scatter(max_orig,max_r_nn,
                # "-",
                marker="*",#markersize=5,
                color="magenta",label='Max NNeighbour',
                alpha=0.5)
# ax.text(x=20,y=70,
# 		s=
# f"""
# Max bias orig-cons:{max_bias_cons*100:.2f}%
# Max bias orig-nn:{max_bias_nn*100:.2f}%
# Mean bias m orig-cons:{mean_bias_cons*100:.2f}%
# Mean bias orig-nn:{mean_bias_nn*100:.2f}%
# MAX bias orig-cons:{MAX_bias_cons*100:.2f}%
# MAX bias orig-nn:{MAX_bias_nn*100:.2f}%
# MEAN orig-cons:{MEAN_bias_cons*100:.2f}%
# MEAN orig-nn:{MEAN_bias_nn*100:.2f}%
# """,
# 		backgroundcolor='azure')
ax.text(x=70,y=20,
		s=
f"""
Max bias orig-cons:{max_bias_cons*100:.2f}%
Max bias orig-nn:{max_bias_nn*100:.2f}%
Mean bias m orig-cons:{mean_bias_cons*100:.2f}%
Mean bias orig-nn:{mean_bias_nn*100:.2f}%
MAX bias orig-cons:{MAX_bias_cons*100:.2f}%
MAX bias orig-nn:{MAX_bias_nn*100:.2f}%
MEAN orig-cons:{MEAN_bias_cons*100:.2f}%
MEAN orig-nn:{MEAN_bias_nn*100:.2f}%
""",
		backgroundcolor='azure')
ax.plot([min_diagonal, max_diagonal], [min_diagonal, max_diagonal])
# ax.scatter(15, 15,color="red")
ax.set_title(f"quantile considered:\n{extr_q*100}\nand Max value")
ax.legend()
ax.set_xlabel("Original [mm/h]")
ax.set_ylabel("Remap [mm/h]")
plt.suptitle(f"QQ-plot of all stations")
plt.savefig(f"figures/qqplot_original_remapcon.png")
plt.close()



arr_conc=np.concatenate([
    np.array(q_bias_nn),
    np.array(Q_bias_nn),
    np.array(M_bias_nn).reshape(-1,1),
    np.array(q_bias_cons),
    np.array(Q_bias_cons),
    np.array(M_bias_cons).reshape(-1,1)
    ],axis=1)

f,ax=plt.subplots(figsize=(12,8))
ax.boxplot([q_bias_nn.reshape(-1),q_bias_cons.reshape(-1)],labels=['NN','Conservative'])
ax.set_title("Boxplot of biases for extreme quantiles q99.99-q99.999")
plt.savefig("figures/boxplot_rel_bias_remap_cons.png")
plt.close()

f,ax=plt.subplots(figsize=(12,8))
ax.boxplot([M_bias_nn.reshape(-1),M_bias_cons.reshape(-1)],labels=['NN','Conservative'])
ax.set_title("Boxplot of biases for Maximum value of each station")
plt.savefig("figures/boxplot_rel_bias_remap_nn.png")
plt.close()

df_box=pd.DataFrame(arr_conc,
                    columns=[f'q{Q*100:.3f}_nn' for Q in extr_Q]+[f'q{q*100:.3f}_nn' for q in extr_q]+['max_nn']+[f'q{Q*100:.3f}_cons' for Q in extr_Q]+[f'q{q*100:.3f}_cons' for q in extr_q]+['max_cons'])
# df_box['col']='Heavy Prec, JJA'
fig=plt.figure(figsize=(14,8))
ax=plt.axes()
# print(df_box.melt())
sns.set(style="darkgrid")
sns.boxplot(y="value", 
            x="variable",
            data=df_box.melt(), 
            # palette=["green","red","blue"], 
            width=0.25,
            ax=ax)
# ax.hlines(y=-5, xmin=-1,xmax=12,linestyles='dashed',color='red')
# ax.hlines(y=25, xmin=-1,xmax=12,linestyles='dashed',color='red')
ax.set_xlabel(f"")
ax.set_ylabel(f"Relative Bias [%]")
ax.set_ylim(-0.5,0.5)
ax.set_xticklabels([f'q{Q*100:.3f}_nn' for Q in extr_Q]+[f'q{q*100:.3f}_nn' for q in extr_q]+['max_nn']+[f'q{Q*100:.3f}_cons' for Q in extr_Q]+[f'q{q*100:.3f}_cons' for q in extr_q]+['max_cons'],rotation=-25)
plt.title("Bias Conservative vs NNeighbour",pad=20,fontsize=20)
plt.savefig("figures/boxplot_rel_bias_remap.png")
plt.close()
    


if PLOT_EPDF:
    print("Plotting EPDF")
    prec_thr=70

    f,ax=plt.subplots(figsize=(9,6))
    ax.hist(a_99_stat[a_99_stat > prec_thr ],np.arange(min(a_99_stat[a_99_stat > prec_thr ]), max(a_99_stat[a_99_stat > prec_thr ]) + 2.5, 2.5), density=True,fc=(0.2,0.2,0,0.25),ec="green",label='Original')
    ax.hist(b_99_stat[b_99_stat > prec_thr ],np.arange(min(b_99_stat[b_99_stat > prec_thr ]), max(b_99_stat[b_99_stat > prec_thr ]) + 2.5, 2.5), density=True,fc=(0,0,1,0.25),ec="blue",label='conservative')
    ax.hist(c_99_stat[c_99_stat > prec_thr ],np.arange(min(c_99_stat[c_99_stat > prec_thr ]), max(c_99_stat[c_99_stat > prec_thr ]) + 2.5, 2.5), density=True,fc=(1,0,0,0.25),ec="red",label='NNeighbour')
    ax.set_title(f"Histogram entire area")
    ax.set_xlabel("Rainfall [mm/hr]")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(f"figures/pdf_orig_vs_remap_only_station.png")
    plt.close()




cdo gridarea /mnt/data/lcesarini/ncks_eth.nc /mnt/data/lcesarini/area_eth.nc