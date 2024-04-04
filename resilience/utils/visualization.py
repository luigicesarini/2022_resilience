#! /mnt/beegfs/lcesarini//miniconda3/envs/detectron/bin/python

import os
import rioxarray
import numpy as np 
import xarray as xr 
import pandas as pd
import seaborn as sns
from glob import glob
from tqdm import tqdm
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt 
from cartopy import feature as cfeature

from resilience.utils.retrieve_data import get_unlist

import warnings
warnings.filterwarnings('ignore')

os.chdir("/mnt/beegfs/lcesarini//2022_resilience/")

shp_triveneto = gpd.read_file("/mnt/beegfs/lcesarini//2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")
shp_triveneto = shp_triveneto[np.isin(shp_triveneto.NAME_1,["Veneto","Trentino-Alto Adige"])]


def get_palettes():
    """
    Utility to retrive the palettes for frequency,intensity and heavy prec.
    The function gives the opportunity to retrive any compination 
    of the 3 palette.
    """
    cmap_freq = (mpl.colors.ListedColormap([
                                        '#ECF7FE',
                                        '#B1DFFA',
                                        '#36BCFF', 
                                        '#508D5E',
                                        '#55CB70',
                                        '#E5E813',
                                        '#E8AB13',
                                        '#E85413',
                                        '#E82313'
                                        ])
            .with_extremes(over='#AB0202', under='#D8EEFA'))

    cmap_inte = (mpl.colors.ListedColormap(['#ECF7FE',
                                            '#B1DFFA',
                                            '#36BCFF', 
                                            '#508D5E',
                                            '#55CB70',
                                            '#88F7A1',
                                            '#E5E813',
                                            '#E8AB13',
                                            '#E85413',
                                            '#E82313'
                                            ])
            .with_extremes(over='#AB0202', under='#D8EEFA'))
    cmap_q = (mpl.colors.ListedColormap(['#B1DFFA',
                                        '#36BCFF', 
                                        '#508D5E',
                                        '#55CB70',
                                        '#E5E813',
                                        '#E8AB13',
                                        '#E85413',
                                        '#E82313'
                                        ])
            .with_extremes(over='#AB0202', under='#D8EEFA'))
    

    return cmap_freq,cmap_inte,cmap_q

def get_levels():
    return np.arange(0.04,0.29,0.03),np.arange(0.3,3.31,0.3),np.arange(2,19,2)


proj = ccrs.PlateCarree()

rot = ccrs.RotatedPole(pole_longitude=-170.0, 
                       pole_latitude=43.0, 
                       central_rotated_longitude=0.0, 
                       globe=None)

def plot_panel(nrow,ncol,
              list_to_plot,
              name_fig,
              list_titles='Any title',
              levels=[9],
              suptitle="Frequency for JJA",
              name_metric="Frequency",
              SET_EXTENT=True,
              cmap='rainbow'
              ):
    """
    
    Plots panel of the given xarray datasets
    
    Parameters
    ----------
    list_to_plot : list, defaults to None
                  list of the dataarray to plot 
    name_fig: 

    list_titles: 

    levles: either an int indicating the number of intervals or a list of values to break to palette into.
    Returns
    -------



    Examples
        --------
    
    """
    fig,axs=plt.subplots(nrow,ncol,
                        figsize=(16,12),constrained_layout=True, squeeze=True,
                        subplot_kw={"projection":proj})

    if (nrow > 1) or (ncol > 1):
        ax=axs.flatten()
    else:
        ax=axs


    for i,model in enumerate(list_to_plot):
        # if i in [0,1,2]:
        #     pcm=model.plot.\
        #         pcolormesh(x="lon",y="lat",ax=ax[i],
        #                    add_colorbar=False,
        #                 #    cbar_kwargs={"shrink":0.85},
        #                    levels=np.arange(0.04,0.28,0.03),
        #                    cmap="rainbow",
        #                    #norm=norm
        #                    )
        # else:
        if (nrow > 1) or (ncol > 1) :
            pcm=model.plot.pcolormesh(ax=ax[i],
                                    add_colorbar=False,
                                    #   cbar_kwargs={"shrink":0.85},
                                    levels=levels[i],
                                    cmap=cmap[i],#"rainbow",
                                        # norm=norm
                                    )

            shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red')
            ax[i].add_feature(cfeature.BORDERS)
            ax[i].coastlines()
            if SET_EXTENT:
                ax[i].set_extent([10.2,13.15,44.6,47.15])
            ax[i].set_title(f"{list_titles[i]}")
            gl = ax[i].gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )


        else:
            pcm=model.plot.pcolormesh(ax=ax,
                        add_colorbar=False,
                        #   cbar_kwargs={"shrink":0.85},
                        levels=levels[i],
                        cmap=cmap[i],#"rainbow",
                            # norm=norm
                        )

            shp_triveneto.boundary.plot(ax=ax,edgecolor='red')
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines()
            ax.set_title(f"{list_titles[i]}")
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )

            # cbar=fig.colorbar(pcm, ax=ax, extend='both', orientation='horizontal')

    cbar=fig.colorbar(pcm, ax=ax[:] if (nrow > 1) or (ncol > 1) else ax, 
                      extend='both', 
                      orientation='vertical',
                      shrink=0.7)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_ylabel(name_metric,fontsize=25,rotation=90)
    fig.suptitle(suptitle, fontsize=20)
    plt.savefig(f"/mnt/beegfs/lcesarini//2022_resilience/figures/{name_fig}.png")
    plt.close()

def plot_slice_model(ds,gds,proj,rot):
    """
    Plot one slice of CP model and the overlay of the station
    """
    plt.figure(figsize=(16,12))
    ax = plt.axes(projection=proj)
    ds.isel(time=9).pr.plot.pcolormesh(
                                        ax=ax, transform=rot, x="rlon", y="rlat", 
                                        add_colorbar=True, cmap= "RdBu"
                                    )
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='--')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.RIVERS)
    ax.set_xlim([10, 13])
    ax.set_ylim([44.5,47.2])
    gds.plot(ax=ax, column = "max_tp")
    gl = ax.gridlines(
        draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--'
    )
    plt.savefig("test.png")


def plot_panel_rotated(figsize,nrow,ncol,
                       list_to_plot,
                       name_fig,
                       list_titles='Any title',
                       levels=[9],
                       suptitle="Frequency for JJA",
                       name_metric=["Frequency"],
                       SET_EXTENT=True,
                       cmap=['rainbow'],
                       proj=ccrs.PlateCarree(),
                       transform=ccrs.PlateCarree(),
                       SAVE=True
                    ):
    """
    
    Plots panel of the given xarray datasets
    
    Parameters
    ----------
    list_to_plot : list, defaults to None
                  list of the dataarray to plot 
    name_fig: 

    list_titles: 

    levels: either an int indicating the number of intervals or a list of values to break to palette into.
    Returns
    -------



    Examples
        --------
    
    """
    fig,axs=plt.subplots(nrow,ncol,
                        figsize=figsize,constrained_layout=True, squeeze=True,
                        subplot_kw={"projection":proj})
    # Adjust the horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    if (nrow > 1) or (ncol > 1):
        ax=axs.flatten()
    else:
        ax=axs


    for i,model in enumerate(list_to_plot):
        # if i in [0,1,2]:
        #     pcm=model.plot.\
        #         pcolormesh(x="lon",y="lat",ax=ax[i],
        #                    add_colorbar=False,
        #                 #    cbar_kwargs={"shrink":0.85},
        #                    levels=np.arange(0.04,0.28,0.03),
        #                    cmap="rainbow",
        #                    #norm=norm
        #                    )
        # else:
        if (nrow > 1) or (ncol > 1) :
            pcm=model.plot.pcolormesh(ax=ax[i],
                                    add_colorbar=False,
                                    #   cbar_kwargs={"shrink":0.85},
                                    levels=levels[i],
                                    cmap=cmap[i],#"rainbow",
                                    transform=transform                                    
                                    )

            # shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red',transform=proj,linewidth=1)
            ax[i].add_feature(cfeature.BORDERS)
            ax[i].coastlines()
            if SET_EXTENT:
                ax[i].set_extent([10.2,13.15,44.6,47.15])
            ax[i].set_title(f"{list_titles[i]}")
            gl = ax[i].gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
            )
            gl.xlabels=False
            if i in np.arange(ncol*nrow-ncol,ncol*nrow):
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='neither', ticks=levels[i],boundaries=levels[i],
                                    orientation='horizontal',
                                    shrink=1)
                cbar.ax.tick_params(labelsize=10)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom',labelpad=45)
                ax[i].add_feature(cfeature.BORDERS)
                # ax[i].add_feature(cfeature.STATES)

        else:
            pcm=model.plot.pcolormesh(ax=ax,
                        add_colorbar=False,
                        #   cbar_kwargs={"shrink":0.85},
                        levels=levels[i],
                        cmap=cmap[i],#"rainbow",
                        transform=transform                                    
                        )

            shp_triveneto.boundary.plot(ax=ax,edgecolor='red')
            # ax.add_geometries(shp_triveneto['geometry'], crs=proj)
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines()
            ax.set_title(f"{list_titles[i]}")
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )
            if SET_EXTENT:
                ax.set_extent([10.2,13.15,44.6,47.15])

            gl.xlines=None
            gl.xlabels_top=None
            gl.xlabels_bottom=None

            # cbar=fig.colorbar(pcm, ax=ax, extend='both', orientation='horizontal')
            if i in [0,1,2]:
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='both', 
                                    orientation='vertical',
                                    shrink=0.85)
                cbar.ax.tick_params(labelsize=30)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom')
    fig.suptitle(suptitle, fontsize=12)
    if SAVE:
        plt.savefig(f"/mnt/beegfs/lcesarini//2022_resilience/figures/{name_fig}.png")
        plt.close()
    else:
        plt.show()


def plot_panel_continous(figsize,nrow,ncol,
                       list_to_plot,
                       name_fig,
                       list_titles='Any title',
                    #    levels=[9],
                       suptitle="Frequency for JJA",
                       name_metric=["Frequency"],
                       SET_EXTENT=True,
                       cmap=['rainbow'],
                       proj=ccrs.PlateCarree(),
                       transform=ccrs.PlateCarree(),
                       SAVE=True,
                       vmin=None,vmax=None
                    ):
    """
    
    Plots panel of the given xarray datasets
    
    Parameters
    ----------
    list_to_plot : list, defaults to None
                  list of the dataarray to plot 
    name_fig: 

    list_titles: 

    levels: either an int indicating the number of intervals or a list of values to break to palette into.
    Returns
    -------



    Examples
        --------
    
    """
    fig,axs=plt.subplots(nrow,ncol,
                        figsize=figsize,constrained_layout=True, squeeze=True,
                        subplot_kw={"projection":proj})
    # Adjust the horizontal and vertical spacing between subplots
    plt.subplots_adjust(
        # left=0, bottom=0, right=0.1, top=0.1, 
        wspace=0, hspace=0
        )

    if (nrow > 1) or (ncol > 1):
        ax=axs.flatten()
    else:
        ax=axs
    
    if vmin is None:
        vmin=[np.nanmin(mdl.values) * 1.25 for mdl in list_to_plot]
    if vmax is None:
        vmax=[np.nanmax(mdl.values) * 0.75 for mdl in list_to_plot]

    for i,model in enumerate(list_to_plot):
        # if i in [0,1,2]:
        #     pcm=model.plot.\
        #         pcolormesh(x="lon",y="lat",ax=ax[i],
        #                    add_colorbar=False,
        #                 #    cbar_kwargs={"shrink":0.85},
        #                    levels=np.arange(0.04,0.28,0.03),
        #                    cmap="rainbow",
        #                    #norm=norm
        #                    )
        # else:
        if (nrow > 1) or (ncol > 1) :
            pcm=model.plot.pcolormesh(ax=ax[i],
                                    add_colorbar=False,
                                    #   cbar_kwargs={"shrink":0.85},
                                    # levels=levels[i],
                                    vmin=vmin[i],
                                    vmax=vmax[i],
                                    cmap=cmap[i],#"rainbow",
                                    transform=transform                                    
                                    )

            # shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red',transform=proj,linewidth=1)
            ax[i].add_feature(cfeature.BORDERS)
            ax[i].coastlines()
            if SET_EXTENT:
                ax[i].set_extent([10.2,13.15,44.6,47.15])
            ax[i].set_title(f"{list_titles[i]}")
            # gl = ax[i].gridlines(
            #     draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
            # )
            # gl.xlabels=False
            if i in np.arange(ncol*nrow-ncol,ncol*nrow):
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    # extend='neither', ticks=levels[i],boundaries=levels[i],
                                    orientation='horizontal',
                                    shrink=0.75)
                cbar.ax.tick_params(labelsize=10)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom',labelpad=45)
                ax[i].add_feature(cfeature.BORDERS)
                # ax[i].add_feature(cfeature.STATES)

        else:
            pcm=model.plot.pcolormesh(ax=ax,
                        add_colorbar=False,
                        #   cbar_kwargs={"shrink":0.85},
                        # levels=levels[i],
                        cmap=cmap[i],#"rainbow",
                        transform=transform                                    
                        )

            shp_triveneto.boundary.plot(ax=ax,edgecolor='red')
            # ax.add_geometries(shp_triveneto['geometry'], crs=proj)
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines()
            ax.set_title(f"{list_titles[i]}")
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )
            if SET_EXTENT:
                ax.set_extent([10.2,13.15,44.6,47.15])

            gl.xlines=None
            gl.xlabels_top=None
            gl.xlabels_bottom=None

            # cbar=fig.colorbar(pcm, ax=ax, extend='both', orientation='horizontal')
            if i in [0,1,2]:
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='both', 
                                    orientation='vertical',
                                    shrink=0.85)
                cbar.ax.tick_params(labelsize=30)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom')
    fig.suptitle(suptitle, fontsize=12)

    if SAVE:
        plt.savefig(f"/mnt/beegfs/lcesarini//2022_resilience/figures/{name_fig}.png")
        plt.close()
    else:
        plt.show()

def plot_panel_fixbar(figsize,nrow,ncol,
                       list_to_plot,
                       name_fig,
                       list_titles='Any title',
                    #    levels=[9],
                       suptitle="Frequency for JJA",
                       name_metric=["Frequency"],
                       SET_EXTENT=True,
                       cmap=['rainbow'],
                       proj=ccrs.PlateCarree(),
                       transform=ccrs.PlateCarree(),
                       SAVE=True,
                       vmin=None,vmax=None
                    ):
    """
    
    Plots panel of the given xarray datasets
    
    Parameters
    ----------
    list_to_plot : list, defaults to None
                  list of the dataarray to plot 
    name_fig: 

    list_titles: 

    levels: either an int indicating the number of intervals or a list of values to break to palette into.
    Returns
    -------



    Examples
        --------
    
    """
    fig,axs=plt.subplots(nrow,ncol,
                        figsize=figsize,constrained_layout=True, squeeze=True,
                        subplot_kw={"projection":proj})
    # Adjust the horizontal and vertical spacing between subplots
    plt.subplots_adjust(
        # left=0, bottom=0, right=0.1, top=0.1, 
        wspace=0, hspace=0
        )

    if (nrow > 1) or (ncol > 1):
        ax=axs.flatten()
    else:
        ax=axs
    
    if vmin is None:
        vmin=[np.nanmin(mdl.values) * 1.25 for mdl in list_to_plot]
    if vmax is None:
        vmax=[np.nanmax(mdl.values) * 0.75 for mdl in list_to_plot]
    
    for i,model in enumerate(list_to_plot):
        # if i in [0,1,2]:
        #     pcm=model.plot.\
        #         pcolormesh(x="lon",y="lat",ax=ax[i],
        #                    add_colorbar=False,
        #                 #    cbar_kwargs={"shrink":0.85},
        #                    levels=np.arange(0.04,0.28,0.03),
        #                    cmap="rainbow",
        #                    #norm=norm
        #                    )
        # else:
        if (nrow > 1) or (ncol > 1) :
            pcm=model.plot.pcolormesh(ax=ax[i],
                                    add_colorbar=False,
                                    #   cbar_kwargs={"shrink":0.85},
                                    # levels=levels[i],
                                    norm=mpl.colors.Normalize(vmin=vmin[i],vmax=vmax[i]),
                                    cmap=cmap[i],#"rainbow",
                                    transform=transform                                    
                                    )

            # shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red',transform=proj,linewidth=1)
            ax[i].add_feature(cfeature.BORDERS)
            ax[i].coastlines()
            if SET_EXTENT:
                ax[i].set_extent([10.2,13.15,44.6,47.15])
            ax[i].set_title(f"{list_titles[i]}")
            # gl = ax[i].gridlines(
            #     draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
            # )
            # gl.xlabels=False
            if i in np.arange(ncol*nrow-ncol,ncol*nrow):
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='both', #ticks=levels[i],boundaries=levels[i],
                                    orientation='horizontal', format="{x:.0f}",
                                    shrink=0.75)
                cbar.ax.tick_params(labelsize=10)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom',labelpad=45)
                ax[i].add_feature(cfeature.BORDERS)
                # ax[i].add_feature(cfeature.STATES)

        else:
            pcm=model.plot.pcolormesh(ax=ax,
                        add_colorbar=False,
                        #   cbar_kwargs={"shrink":0.85},
                        # levels=levels[i],
                        cmap=cmap[i],#"rainbow",
                        transform=transform                                    
                        )

            shp_triveneto.boundary.plot(ax=ax,edgecolor='red')
            # ax.add_geometries(shp_triveneto['geometry'], crs=proj)
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines()
            ax.set_title(f"{list_titles[i]}")
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )
            if SET_EXTENT:
                ax.set_extent([10.2,13.15,44.6,47.15])

            gl.xlines=None
            gl.xlabels_top=None
            gl.xlabels_bottom=None

            # cbar=fig.colorbar(pcm, ax=ax, extend='both', orientation='horizontal')
            if i in [0,1,2]:
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='both', 
                                    orientation='vertical',
                                    shrink=0.85)
                cbar.ax.tick_params(labelsize=30)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom')
    fig.suptitle(suptitle, fontsize=12)

    if SAVE:
        plt.savefig(f"/mnt/beegfs/lcesarini//2022_resilience/figures/{name_fig}.png")
        plt.close()
    else:
        plt.show()


def plot_diff(x,y,dates,
              name_x,name_y,name_station,
              return_fig=True):
    
    """
    PLOT DIFFERENCES

    hist_m - obs
    era5_o - obs

    x and y must be shape (len(obs),1).

    PARAMETERS
    @x: timeseries from which are evaluated the differences
    @y: reference timeseries
    @name_x: name of the product form whcih is taken the timeseries to evaluate
    @name_y: name of the product used as reference
    @name_station: Name of the station that the plot is for. Needed for plot in the title
    @return_fig: bool if True returns the figure plot, if flase saves only the file
    """    

    if (len(x.shape) == 1) | (len(y.shape) == 1):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

    fig = plt.figure(figsize=(16,12))
    plt.plot(dates,(x-y),'og', label=f'{name_x}-{name_y}', figure=fig)
    plt.legend(fontsize=20)
    plt.suptitle(f"Station: {name_station}", fontsize=25, figure=fig)
    plt.title(f"Differences between {name_x} and {name_y}", fontsize=20, figure=fig)
    plt.ylabel(f"Difference in precipitation [mm]", fontsize=20, figure=fig)
    # plt.xlim(('2004-08-29T09:00:00','2005-08-29T14:00:00'))
    plt.savefig(f"differences_{name_x}_{name_y}.png")
    plt.close()
    
    if return_fig:
        return fig


def plot_ts(list_ts:list,dates:np.datetime64,
           names:list,name_station:str,
           return_fig=True):
    
    """
    PLOT TIME SERIES

    PARAMETERS

    @list_ts: list of object that contains a timeseries
    @dates: list of dates of the period to plot
    @names: names of the object from which the time series are extracted
    @name_station: name of the station under evaluation
    """
    fig = plt.figure(figsize=(16,12))
    [plt.plot(dates,ts, '-', label=f'{name_ts}', figure=fig) for ts,name_ts in zip(list_ts,names)]
    plt.legend(fontsize=20)
    plt.title(f"Station: {name_station}", fontsize=25)
    plt.ylabel(f"Precipitation [mm]", fontsize=20)
    # plt.xlim(('2004-08-29T09:00:00','2005-08-29T14:00:00'))
    plt.savefig(f"timeseries_{name_station}.png")
    plt.close()

    if return_fig:
        return fig

def plot_bias(list_ts:list,dates:np.datetime64,
              names:list,name_station:str,
              return_fig=True):
    
    """
    PLOT BIAS

    PARAMETERS

    @list_ts: list of object that contains a timeseries
    @dates: list of dates of the period to plot
    @names: names of the object from which the time series are extracted
    @name_station: name of the station under evaluation
    """
    fig = plt.figure(figsize=(16,12))
    [plt.plot(dates,ts, 'o', label=f'{name_ts}', figure=fig) for ts,name_ts in zip(list_ts,names)]
    plt.legend(fontsize=20)
    plt.title(f"Station: {name_station}", fontsize=25)
    plt.ylabel(f"Precipitation [mm]", fontsize=20)
    # plt.xlim(('2004-08-29T09:00:00','2005-08-29T14:00:00'))
    plt.savefig(f"ts_bias_{name_station}.png")
    plt.close()

    if return_fig:
        return fig

def scatter(prec_o_over, era_station,ds_sliced,name_station):
    fig,ax = plt.subplots(figsize=(16,12))
    plt.plot(np.sort(prec_o_over[prec_o_over > 0],axis=0), 
             np.sort(era_station.tp.values.reshape(-1,1)[prec_o_over > 0],axis=0),'o', label='era5')
    plt.plot(np.sort(prec_o_over[prec_o_over > 0], axis=0),
             np.sort(ds_sliced.pr.values.reshape(-1,1)[prec_o_over > 0],axis=0),'o', label='model')
    plt.legend(fontsize=20)
    ax.axline([0,0],[1,1])
    ax.set_xlim(np.nanmin(prec_o_over[prec_o_over > 0]),np.nanmax(prec_o_over[prec_o_over > 0]))
    ax.set_ylim(np.array([era_station.tp.values.reshape(-1,1)[prec_o_over > 0],
                          ds_sliced.pr.values.reshape(-1,1)[prec_o_over > 0]]).min(),
                np.array([era_station.tp.values.reshape(-1,1)[prec_o_over > 0],
                          ds_sliced.pr.values.reshape(-1,1)[prec_o_over > 0]]).max())
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Sorted observation [mm]", fontsize=20)
    plt.ylabel("Sorted modelled [mm]", fontsize=20)
    plt.savefig(f"scatter_{name_station}.png")
    plt.close()

def plot_heatmap(path_heatmap_txt):
    """
    Utility to plot a heatmap using seaborn
    """
    cmap = (mpl.colors.ListedColormap(['#7E1104',
                                    '#E33434', 
                                    '#F58080', 
                                    '#F8BCBC', 
                                    '#FBE2E2', 
                                    'white',
                                    '#D4F7FA',
                                    '#90DEF8',
                                    '#7BB2ED',
                                    '#262BBD',
                                    '#040880'
                                    ]))
    array_heatmap=np.loadtxt(path_heatmap_txt)
    df=pd.DataFrame(array_heatmap,columns=['Freq','Int','Quantile'],index=['SON','DJF','MAM','JJA'])
    pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5,linecolor="black")
    pcm.set(xlabel="", ylabel="")
    pcm.xaxis.tick_top()
    plt.savefig(f"figures/{os.path.basename(path_heatmap_txt)}.png")
    plt.close()


def plot_boxplot(list_to_concatenate,
                 names_to_concatenate,
                 title,filename,SAVE=False):
        df_box=pd.DataFrame(np.concatenate([mdl.reshape(-1,1) for mdl in list_to_concatenate],
                                axis=1),
                            columns=[name for name in names_to_concatenate])
        # df_box['col']='Heavy Prec, JJA'
        fig=plt.figure(figsize=(14,8))
        ax=plt.axes()
        # print(df_box.melt())
        sns.set(style="darkgrid")
        sns.boxplot(y="value", 
                    x="variable",
                    data=df_box.melt()[~pd.isnull(df_box.melt().value)], 
                    palette=["green","red",'blue'], 
                    notch=True,
                    width=0.25,
                    flierprops={"marker": "*"},
                    medianprops={"color": "coral"},
                    ax=ax)
        ax.hlines(y=-5, xmin=-1,xmax=3,linestyles='dashed',color='red')
        ax.hlines(y=25, xmin=-1,xmax=3,linestyles='dashed',color='red')
        ax.set_xlabel(f"")
        ax.set_ylabel(f"Relative Bias [%]")
        ax.set_ylim(-100,100)
        # ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
        plt.title(title,pad=20,fontsize=20)
        if SAVE:
            plt.savefig(f"figures/{filename}.png")
            plt.close()
        else:
            plt.show()


def plot_bin_hist(list_of_len,list_of_max,list_of_avg,
                  bins_max,bins_avg,bins_duration,
                  name_model='SPHERA',ev='Precipitation',
                  SAVE=True,palette="rainbow",
                  ):
    """
    
    Parameters
    ----------
    list_of_len : list, defaults to None
        a list with the duration of the events

    list_of_max : list, defaults to None
        a list with the max value of each event

    list_of_avg : list, defaults to None
        a list with the average value of each event

    name_model : str, defaults to None
        Name of the model 

    ev : str, defaults to None
        environmental variable to plot. Either "precipitation","wind", or "combined

    SAVE : bool, defaults to None
        True if you want to save it to disk.


    Returns
    -------

    A plto
    

    Examples
    --------


    """
    duration=get_unlist(list_of_len)
    avg_inte=get_unlist(list_of_avg)
    max_inte=get_unlist(list_of_max)

    H, yedges, xedges = np.histogram2d(max_inte,
                                    duration,
                                    bins=[bins_max,bins_duration]);

    H2, yedges2, xedges2 = np.histogram2d(avg_inte,
                                        duration,
                                        bins=[bins_avg,bins_duration]);
    # print(xedges)
    #np.log([0.2,5,10,20,50,100,170])

    # Plot histogram using pcolormesh
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,6), sharey=False)
    if ev =="Precipitation":
        pcm=ax1.pcolormesh(xedges, np.log(yedges), np.where(H==0,np.nan,H)/len(duration), cmap=palette)
        ax1.set_ylim(np.log(np.min(max_inte)), np.log(np.max(max_inte)))
        ax1.set_xlim(np.min(duration), np.min([np.max(duration),8]))
        ax1.set_ylabel('Peak Intensity')
        ax1.set_xlabel('Duration')
        ax1.set_title('')
        ax1.set_xticks(np.arange(1,11,2))
        ax1.set_xticklabels(np.arange(1,11,2))
        ax1.set_yticks(np.log([5,10,20,50,100,170]))
        ax1.set_yticklabels([5,10,20,50,100,170])
        # ax1.grid()
        cbar = plt.colorbar(pcm)
        # Set colorbar label
        cbar.set_label('', rotation=270, labelpad=20)

        pcm2=ax2.pcolormesh(xedges2, np.log(yedges2), (np.where(H2==0,np.nan,H2)/len(duration)), cmap=palette)
        ax2.set_ylim(np.log(np.min(avg_inte)), np.log(np.max(avg_inte)))
        ax2.set_xlim(np.min(duration), np.min([np.max(duration),8]))
        ax2.set_ylabel('Mean Intensity')
        ax2.set_xlabel('Duration')
        ax2.set_title('')
        # ax2.set_yticks(np.arange(0,100,2.5))
        # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
        ax2.set_xticks(np.arange(1,11,2))
        ax2.set_xticklabels(labels=np.arange(1,11,2))
        ax2.set_yticks(np.log([5,10,20,50,100,170]))
        ax2.set_yticklabels([5,10,20,50,100,170])
    elif ev == 'Wind':
        pcm=ax1.pcolormesh(xedges,(yedges), np.where(H==0,np.nan,H)/len(duration), cmap=palette)
        ax1.set_ylim((np.min(max_inte)), (np.max(max_inte)))
        # ax1.set_ylim((np.min(max_inte)), 20)
        ax1.set_xlim(np.min(duration), np.max(duration))
        # ax1.set_xlim(np.min(duration), 20)
        ax1.set_ylabel('Peak Intensity')
        ax1.set_xlabel('Duration')
        ax1.set_title('')
        # ax1.set_xticks(np.arange(0,10,2))
        # ax1.set_xticklabels(np.arange(0,10,2))
        # ax1.set_yticks(([5,10,20,50,100,170]))
        # ax1.set_yticklabels([5,10,20,50,100,170])
        # ax1.grid()
        cbar = plt.colorbar(pcm)
        # Set colorbar label
        cbar.set_label('', rotation=270, labelpad=20)

        pcm2=ax2.pcolormesh(xedges2, (yedges2), (np.where(H2==0,np.nan,H2)/len(duration)), cmap=palette)
        ax2.set_ylim((np.min(avg_inte)), (np.max(avg_inte)))
        ax2.set_xlim(np.min(duration), np.max(duration))
        ax2.set_ylabel('Mean Intensity')
        ax2.set_xlabel('Duration')
        ax2.set_title('')
        # ax2.set_yticks(np.arange(0,100,2.5))
        # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
        # ax2.set_xticks(np.arange(0,10,2))
        # ax2.set_xticklabels(labels=np.arange(0,10,2))
        # ax2.set_yticks(([5,10,20,50,100,170]))
        # ax2.set_yticklabels([5,10,20,50,100,170])
    # ax2.grid()
    cbar = plt.colorbar(pcm2)
    # Set colorbar label
    cbar.set_label('', rotation=270, labelpad=20)
    plt.suptitle(f"{name_model} for {ev}")
    if SAVE:
        plt.savefig(f"/mnt/beegfs/lcesarini//{name_model}_{ev}.png")
    else:
        plt.show()



if __name__=="__main__":
    path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"

    meta_station = pd.read_csv("meta_station_updated_col.csv")

    gds=gpd.GeoDataFrame(meta_station,geometry=gpd.points_from_xy(meta_station[["lon"]],
                            meta_station["lat"], 
                            crs="EPSG:4326"))    
    
    ds=xr.open_dataset(path_model+"/CPM_ETH_MPI_historical_Italy_1996-2005_pr_hour.nc")

def plot_panel_rotated(figsize,nrow,ncol,
                       list_to_plot,
                       name_fig,
                       list_titles='Any title',
                       levels=[9],
                       suptitle="Frequency for JJA",
                       name_metric=["Frequency"],
                       SET_EXTENT=True,
                       cmap=['rainbow'],
                       proj=ccrs.PlateCarree(),
                       transform=ccrs.PlateCarree(),
                       SAVE=True
                    ):
    """
    
    Plots panel of the given xarray datasets
    
    Parameters
    ----------
    list_to_plot : list, defaults to None
                  list of the dataarray to plot 
    name_fig: 

    list_titles: 

    levels: either an int indicating the number of intervals or a list of values to break to palette into.
    Returns
    -------



    Examples
        --------
    
    """
    fig,axs=plt.subplots(nrow,ncol,
                        figsize=figsize,constrained_layout=True, squeeze=True,
                        subplot_kw={"projection":proj})
    # Adjust the horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    if (nrow > 1) or (ncol > 1):
        ax=axs.flatten()
    else:
        ax=axs


    for i,model in enumerate(list_to_plot):
        # if i in [0,1,2]:
        #     pcm=model.plot.\
        #         pcolormesh(x="lon",y="lat",ax=ax[i],
        #                    add_colorbar=False,
        #                 #    cbar_kwargs={"shrink":0.85},
        #                    levels=np.arange(0.04,0.28,0.03),
        #                    cmap="rainbow",
        #                    #norm=norm
        #                    )
        # else:
        if (nrow > 1) or (ncol > 1) :
            pcm=model.plot.pcolormesh(ax=ax[i],
                                    add_colorbar=False,
                                    #   cbar_kwargs={"shrink":0.85},
                                    levels=levels[i],
                                    cmap=cmap[i],#"rainbow",
                                    transform=transform                                    
                                    )

            # shp_triveneto.boundary.plot(ax=ax[i],edgecolor='red',transform=proj,linewidth=1)
            ax[i].add_feature(cfeature.BORDERS)
            ax[i].coastlines()
            if SET_EXTENT:
                ax[i].set_extent([10.2,13.15,44.6,47.15])
            ax[i].set_title(f"{list_titles[i]}")
            gl = ax[i].gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--',
            )
            gl.xlabels=False
            if i in np.arange(ncol*nrow-ncol,ncol*nrow):
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='neither', ticks=levels[i],boundaries=levels[i],
                                    orientation='horizontal',
                                    shrink=1)
                cbar.ax.tick_params(labelsize=10)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom',labelpad=45)
                ax[i].add_feature(cfeature.BORDERS)
                # ax[i].add_feature(cfeature.STATES)

        else:
            pcm=model.plot.pcolormesh(ax=ax,
                        add_colorbar=False,
                        #   cbar_kwargs={"shrink":0.85},
                        levels=levels[i],
                        cmap=cmap[i],#"rainbow",
                        transform=transform                                    
                        )

            shp_triveneto.boundary.plot(ax=ax,edgecolor='red')
            # ax.add_geometries(shp_triveneto['geometry'], crs=proj)
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines()
            ax.set_title(f"{list_titles[i]}")
            gl = ax.gridlines(
                draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--'
            )
            if SET_EXTENT:
                ax.set_extent([10.2,13.15,44.6,47.15])

            gl.xlines=None
            gl.xlabels_top=None
            gl.xlabels_bottom=None

            # cbar=fig.colorbar(pcm, ax=ax, extend='both', orientation='horizontal')
            if i in [0,1,2]:
                cbar=fig.colorbar(pcm, ax=ax[i] if (nrow > 1) or (ncol > 1) else ax, 
                                    extend='both', 
                                    orientation='vertical',
                                    shrink=0.85)
                cbar.ax.tick_params(labelsize=30)
                cbar.ax.set_ylabel(name_metric[i],fontsize=10,rotation=0,loc='bottom')
    fig.suptitle(suptitle, fontsize=12)
    if SAVE:
        plt.savefig(f"/mnt/beegfs/lcesarini//2022_resilience/figures/{name_fig}.png")
        plt.close()
    else:
        plt.show()

def plot_diff(x,y,dates,
              name_x,name_y,name_station,
              return_fig=True):
    
    """
    PLOT DIFFERENCES

    hist_m - obs
    era5_o - obs

    x and y must be shape (len(obs),1).

    PARAMETERS
    @x: timeseries from which are evaluated the differences
    @y: reference timeseries
    @name_x: name of the product form whcih is taken the timeseries to evaluate
    @name_y: name of the product used as reference
    @name_station: Name of the station that the plot is for. Needed for plot in the title
    @return_fig: bool if True returns the figure plot, if flase saves only the file
    """    

    if (len(x.shape) == 1) | (len(y.shape) == 1):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

    fig = plt.figure(figsize=(16,12))
    plt.plot(dates,(x-y),'og', label=f'{name_x}-{name_y}', figure=fig)
    plt.legend(fontsize=20)
    plt.suptitle(f"Station: {name_station}", fontsize=25, figure=fig)
    plt.title(f"Differences between {name_x} and {name_y}", fontsize=20, figure=fig)
    plt.ylabel(f"Difference in precipitation [mm]", fontsize=20, figure=fig)
    # plt.xlim(('2004-08-29T09:00:00','2005-08-29T14:00:00'))
    plt.savefig(f"differences_{name_x}_{name_y}.png")
    plt.close()
    
    if return_fig:
        return fig


def plot_ts(list_ts:list,dates:np.datetime64,
           names:list,name_station:str,
           return_fig=True):
    
    """
    PLOT TIME SERIES

    PARAMETERS

    @list_ts: list of object that contains a timeseries
    @dates: list of dates of the period to plot
    @names: names of the object from which the time series are extracted
    @name_station: name of the station under evaluation
    """
    fig = plt.figure(figsize=(16,12))
    [plt.plot(dates,ts, '-', label=f'{name_ts}', figure=fig) for ts,name_ts in zip(list_ts,names)]
    plt.legend(fontsize=20)
    plt.title(f"Station: {name_station}", fontsize=25)
    plt.ylabel(f"Precipitation [mm]", fontsize=20)
    # plt.xlim(('2004-08-29T09:00:00','2005-08-29T14:00:00'))
    plt.savefig(f"timeseries_{name_station}.png")
    plt.close()

    if return_fig:
        return fig

def plot_bias(list_ts:list,dates:np.datetime64,
              names:list,name_station:str,
              return_fig=True):
    
    """
    PLOT BIAS

    PARAMETERS

    @list_ts: list of object that contains a timeseries
    @dates: list of dates of the period to plot
    @names: names of the object from which the time series are extracted
    @name_station: name of the station under evaluation
    """
    fig = plt.figure(figsize=(16,12))
    [plt.plot(dates,ts, 'o', label=f'{name_ts}', figure=fig) for ts,name_ts in zip(list_ts,names)]
    plt.legend(fontsize=20)
    plt.title(f"Station: {name_station}", fontsize=25)
    plt.ylabel(f"Precipitation [mm]", fontsize=20)
    # plt.xlim(('2004-08-29T09:00:00','2005-08-29T14:00:00'))
    plt.savefig(f"ts_bias_{name_station}.png")
    plt.close()

    if return_fig:
        return fig

def scatter(prec_o_over, era_station,ds_sliced,name_station):
    fig,ax = plt.subplots(figsize=(16,12))
    plt.plot(np.sort(prec_o_over[prec_o_over > 0],axis=0), 
             np.sort(era_station.tp.values.reshape(-1,1)[prec_o_over > 0],axis=0),'o', label='era5')
    plt.plot(np.sort(prec_o_over[prec_o_over > 0], axis=0),
             np.sort(ds_sliced.pr.values.reshape(-1,1)[prec_o_over > 0],axis=0),'o', label='model')
    plt.legend(fontsize=20)
    ax.axline([0,0],[1,1])
    ax.set_xlim(np.nanmin(prec_o_over[prec_o_over > 0]),np.nanmax(prec_o_over[prec_o_over > 0]))
    ax.set_ylim(np.array([era_station.tp.values.reshape(-1,1)[prec_o_over > 0],
                          ds_sliced.pr.values.reshape(-1,1)[prec_o_over > 0]]).min(),
                np.array([era_station.tp.values.reshape(-1,1)[prec_o_over > 0],
                          ds_sliced.pr.values.reshape(-1,1)[prec_o_over > 0]]).max())
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Sorted observation [mm]", fontsize=20)
    plt.ylabel("Sorted modelled [mm]", fontsize=20)
    plt.savefig(f"scatter_{name_station}.png")
    plt.close()

def plot_heatmap(path_heatmap_txt):
    """
    Utility to plot a heatmap using seaborn
    """
    cmap = (mpl.colors.ListedColormap(['#7E1104',
                                    '#E33434', 
                                    '#F58080', 
                                    '#F8BCBC', 
                                    '#FBE2E2', 
                                    'white',
                                    '#D4F7FA',
                                    '#90DEF8',
                                    '#7BB2ED',
                                    '#262BBD',
                                    '#040880'
                                    ]))
    array_heatmap=np.loadtxt(path_heatmap_txt)
    df=pd.DataFrame(array_heatmap,columns=['Freq','Int','Quantile'],index=['SON','DJF','MAM','JJA'])
    pcm = sns.heatmap(df, annot=True,cmap=cmap,linewidths=.5,linecolor="black")
    pcm.set(xlabel="", ylabel="")
    pcm.xaxis.tick_top()
    plt.savefig(f"figures/{os.path.basename(path_heatmap_txt)}.png")
    plt.close()


def plot_boxplot_wind(list_to_concatenate,
                      names_to_concatenate,
                      title,filename,SAVE=False):
        df_box=pd.DataFrame(np.concatenate([mdl.reshape(-1,1) for mdl in list_to_concatenate],
                                axis=1),
                            columns=[name for name in names_to_concatenate])
        # df_box['col']='Heavy Prec, JJA'
        fig=plt.figure(figsize=(14,8))
        ax=plt.axes()
        # print(df_box.melt())
        sns.set(style="darkgrid")
        sns.boxplot(y="value", 
                    x="variable",
                    data=df_box.melt()[~pd.isnull(df_box.melt().value)], 
                    palette=["green","red",'blue'], 
                    notch=True,
                    width=0.25,
                    flierprops={"marker": "*"},
                    medianprops={"color": "coral"},
                    ax=ax)
        ax.hlines(y=-5, xmin=-1,xmax=3,linestyles='dashed',color='red')
        ax.hlines(y=25, xmin=-1,xmax=3,linestyles='dashed',color='red')
        ax.set_xlabel(f"")
        ax.set_ylabel(f"Relative Bias [%]")
        ax.set_ylim(-100,100)
        # ax.set_xticklabels([f"{m[:4]}. {s}" for s in seasons for m in list_ms],rotation=-25)
        plt.title(title,pad=20,fontsize=20)
        if SAVE:
            plt.savefig(f"figures/{filename}.png")
            plt.close()
        else:
            plt.show()


# def plot_bin_hist(list_of_len,list_of_max,list_of_avg,
#                   bins_max,bins_avg,bins_duration,
#                   name_model='SPHERA',ev='Precipitation',
#                   SAVE=True,palette="rainbow",
#                   ):
#     """
    
#     Parameters
#     ----------
#     list_of_len : list, defaults to None
#         a list with the duration of the events

#     list_of_max : list, defaults to None
#         a list with the max value of each event

#     list_of_avg : list, defaults to None
#         a list with the average value of each event

#     name_model : str, defaults to None
#         Name of the model 

#     ev : str, defaults to None
#         environmental variable to plot. Either "precipitation","wind", or "combined

#     SAVE : bool, defaults to None
#         True if you want to save it to disk.


#     Returns
#     -------

#     A plto
    

#     Examples
#     --------


#     """
#     duration=get_unlist(list_of_len)
#     avg_inte=get_unlist(list_of_avg)
#     max_inte=get_unlist(list_of_max)

#     H, yedges, xedges = np.histogram2d(max_inte,
#                                     duration,
#                                     bins=[bins_max,bins_duration]);

#     H2, yedges2, xedges2 = np.histogram2d(avg_inte,
#                                         duration,
#                                         bins=[bins_avg,bins_duration]);
#     # print(xedges)
#     #np.log([0.2,5,10,20,50,100,170])

#     # Plot histogram using pcolormesh
#     fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,6), sharey=False)
#     if ev =="Precipitation":
#         pcm=ax1.pcolormesh(xedges, np.log(yedges), np.where(H==0,np.nan,H)/len(duration), cmap=palette)
#         ax1.set_ylim(np.log(np.min(max_inte)), np.log(np.max(max_inte)))
#         ax1.set_xlim(np.min(duration), np.min([np.max(duration),8]))
#         ax1.set_ylabel('Peak Intensity')
#         ax1.set_xlabel('Duration')
#         ax1.set_title('')
#         ax1.set_xticks(np.arange(1,11,2))
#         ax1.set_xticklabels(np.arange(1,11,2))
#         ax1.set_yticks(np.log([5,10,20,50,100,170]))
#         ax1.set_yticklabels([5,10,20,50,100,170])
#         # ax1.grid()
#         cbar = plt.colorbar(pcm)
#         # Set colorbar label
#         cbar.set_label('', rotation=270, labelpad=20)

#         pcm2=ax2.pcolormesh(xedges2, np.log(yedges2), (np.where(H2==0,np.nan,H2)/len(duration)), 
#                             cmap=palette,
#                             norm=mpl.colors.Normalize(vmin=0.01,vmax=0.2))
#         ax2.set_ylim(np.log(np.min(avg_inte)), np.log(np.max(avg_inte)))
#         ax2.set_xlim(np.min(duration), np.min([np.max(duration),8]))
#         ax2.set_ylabel('Mean Intensity')
#         ax2.set_xlabel('Duration')
#         ax2.set_title('')
#         # ax2.set_yticks(np.arange(0,100,2.5))
#         # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
#         ax2.set_xticks(np.arange(1,11,2))
#         ax2.set_xticklabels(labels=np.arange(1,11,2))
#         ax2.set_yticks(np.log([5,10,20,50,100,170]))
#         ax2.set_yticklabels([5,10,20,50,100,170])
#     elif ev == 'Wind':
#         pcm=ax1.pcolormesh(xedges,(yedges), np.where(H==0,np.nan,H)/len(duration), cmap=palette)
#         ax1.set_ylim((np.min(max_inte)), (np.max(max_inte)))
#         # ax1.set_ylim((np.min(max_inte)), 20)
#         ax1.set_xlim(np.min(duration), np.max(duration))
#         # ax1.set_xlim(np.min(duration), 20)
#         ax1.set_ylabel('Peak Intensity')
#         ax1.set_xlabel('Duration')
#         ax1.set_title('')
#         # ax1.set_xticks(np.arange(0,10,2))
#         # ax1.set_xticklabels(np.arange(0,10,2))
#         # ax1.set_yticks(([5,10,20,50,100,170]))
#         # ax1.set_yticklabels([5,10,20,50,100,170])
#         # ax1.grid()
#         cbar = plt.colorbar(pcm)
#         # Set colorbar label
#         cbar.set_label('', rotation=270, labelpad=20)

#         pcm2=ax2.pcolormesh(xedges2, (yedges2), (np.where(H2==0,np.nan,H2)/len(duration)), cmap=palette)
#         ax2.set_ylim((np.min(avg_inte)), (np.max(avg_inte)))
#         ax2.set_xlim(np.min(duration), np.max(duration))
#         ax2.set_ylabel('Mean Intensity')
#         ax2.set_xlabel('Duration')
#         ax2.set_title('')
#         # ax2.set_yticks(np.arange(0,100,2.5))
#         # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
#         # ax2.set_xticks(np.arange(0,10,2))
#         # ax2.set_xticklabels(labels=np.arange(0,10,2))
#         # ax2.set_yticks(([5,10,20,50,100,170]))
#         # ax2.set_yticklabels([5,10,20,50,100,170])
#     # ax2.grid()
#     cbar = plt.colorbar(pcm2)
#     # Set colorbar label
#     cbar.set_label('', rotation=270, labelpad=20)
#     plt.suptitle(f"{name_model} for {ev}")
#     if SAVE:
#         plt.savefig(f"/mnt/beegfs/lcesarini//{name_model}_{ev}.png")
#     else:
#         plt.show()

def plot_bin_hist(list_of_len,list_of_max,list_of_avg,
                  bins_max,bins_avg,bins_duration,
                  name_model='SPHERA',ev='Precipitation',
                  SAVE=True,palette="rainbow",
                  ):
    """
    
    Parameters
    ----------
    list_of_len : list, defaults to None
        a list with the duration of the events

    list_of_max : list, defaults to None
        a list with the max value of each event

    list_of_avg : list, defaults to None
        a list with the average value of each event

    name_model : str, defaults to None
        Name of the model 

    ev : str, defaults to None
        environmental variable to plot. Either "precipitation","wind", or "combined

    SAVE : bool, defaults to None
        True if you want to save it to disk.


    Returns
    -------

    A plot
    

    Examples
    --------


    """
    duration=get_unlist(list_of_len)
    avg_inte=get_unlist(list_of_avg)
    max_inte=get_unlist(list_of_max)

    H, yedges, xedges = np.histogram2d(max_inte,
                                    duration,
                                    bins=[bins_max,bins_duration]);

    H2, yedges2, xedges2 = np.histogram2d(avg_inte,
                                        duration,
                                        bins=[bins_avg,bins_duration]);
    # print(xedges)
    #np.log([0.2,5,10,20,50,100,170])

    # Plot histogram using pcolormesh
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,6), sharey=False)
    if ev =="Precipitation":
        pcm=ax1.pcolormesh(xedges, np.log(yedges), np.where(H==0,np.nan,H)/len(duration), 
                            cmap=palette,
                            norm=mpl.colors.Normalize(vmin=0.0,vmax=0.08)
                            )

        ax1.set_ylim(np.log(np.min(max_inte)), np.log(np.max(max_inte)))
        ax1.set_xlim(np.min(duration), np.min([np.max(duration),8]))
        ax1.set_ylabel('Peak Intensity')
        ax1.set_xlabel('Duration')
        ax1.set_title('')
        ax1.set_xticks(np.arange(1,11,2))
        ax1.set_xticklabels(np.arange(1,11,2))
        ax1.set_yticks(np.log([5,10,20,50,100,170]))
        ax1.set_yticklabels([5,10,20,50,100,170])
        # ax1.grid()
        cbar = plt.colorbar(pcm)
        # Set colorbar label
        cbar.set_label('', rotation=270, labelpad=20)

        pcm2=ax2.pcolormesh(xedges2, np.log(yedges2), (np.where(H2==0,np.nan,H2)/len(duration)), 
                            cmap=palette,
                            norm=mpl.colors.Normalize(vmin=0.0,vmax=0.04)
                            )
        ax2.set_ylim(np.log(np.min(avg_inte)), np.log(np.max(avg_inte)))
        ax2.set_xlim(np.min(duration), np.min([np.max(duration),8]))
        ax2.set_ylabel('Mean Intensity')
        ax2.set_xlabel('Duration')
        ax2.set_title('')
        # ax2.set_yticks(np.arange(0,100,2.5))
        # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
        ax2.set_xticks(np.arange(1,11,2))
        ax2.set_xticklabels(labels=np.arange(1,11,2))
        ax2.set_yticks(np.log([5,10,20,50,100,170]))
        ax2.set_yticklabels([5,10,20,50,100,170])
    elif ev == 'Wind':
        pcm=ax1.pcolormesh(xedges, (yedges), (np.where(H==0,np.nan,H)/len(duration)),
                            cmap=palette,
                            norm=mpl.colors.Normalize(vmin=0.0025,vmax=0.05)
                            )
        ax1.set_ylim((np.min(max_inte)), (np.max(max_inte)))
        # ax1.set_ylim((np.min(max_inte)), 20)
        ax1.set_xlim(np.min(duration),  np.min([np.max(duration),8]))        # ax1.set_xlim(np.min(duration), 20)
        ax1.set_ylabel('Peak Intensity')
        ax1.set_xlabel('Duration')
        ax1.set_title('')
        # ax1.set_xticks(np.arange(0,10,2))
        # ax1.set_xticklabels(np.arange(0,10,2))
        # ax1.set_yticks(([5,10,20,50,100,170]))
        # ax1.set_yticklabels([5,10,20,50,100,170])
        # ax1.grid()
        cbar = plt.colorbar(pcm)
        # Set colorbar label
        cbar.set_label('', rotation=270, labelpad=20)

        pcm2=ax2.pcolormesh(xedges2, (yedges2), (np.where(H2==0,np.nan,H2)/len(duration)),
                            cmap=palette,
                            norm=mpl.colors.Normalize(vmin=0.0025,vmax=0.05)
                            )
        ax2.set_ylim((np.min(avg_inte)), (np.max(avg_inte)))
        ax2.set_xlim(np.min(duration),  np.min([np.max(duration),8]))
        ax2.set_ylabel('Mean Intensity')
        ax2.set_xlabel('Duration')
        ax2.set_title('')
        # ax2.set_yticks(np.arange(0,100,2.5))
        # ax2.set_yticklabels(position=np.arange(0,100,10),labels=np.arange(0,100,10))
        # ax2.set_xticks(np.arange(0,10,2))
        # ax2.set_xticklabels(labels=np.arange(0,10,2))
        # ax2.set_yticks(([5,10,20,50,100,170]))
        # ax2.set_yticklabels([5,10,20,50,100,170])
    # ax2.grid()
    cbar = plt.colorbar(pcm2)
    # Set colorbar label
    cbar.set_label('', rotation=270, labelpad=20)
    plt.suptitle(f"{name_model} for {ev}")
    if SAVE:
        plt.savefig(f"/mnt/beegfs/lcesarini//{name_model}_{ev}.png")
    else:
        plt.show()

    return(H,H2)



if __name__=="__main__":
    path_model="/mnt/data/RESTRICTED/CARIPARO/datiDallan"

    meta_station = pd.read_csv("meta_station_updated_col.csv")

    gds=gpd.GeoDataFrame(meta_station,geometry=gpd.points_from_xy(meta_station[["lon"]],
                            meta_station["lat"], 
                            crs="EPSG:4326"))    
    
    ds=xr.open_dataset(path_model+"/CPM_ETH_MPI_historical_Italy_1996-2005_pr_hour.nc")



    plot_slice_model(ds,gds,proj,rot)