library(tmap)
library(glue)
library(ncdf4)
library(dplyr)
library(raster)
library(ggplot2)

setwd("/home/lcesarini/2022_resilience/csv")

select <- dplyr::select
filter <- dplyr::select

SEAS="DJF"
SPLIT="SEQUENTIAL"
ADJUST="VALIDATION"
m="q"
M="q"
SAVE_MAPS_METRIC=TRUE
WITH_SS <- FALSE
REF = "STATIONS"
BBOX <- st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))

source("/home/lcesarini/2022_resilience/resilience/scripts/plot_function.R")

list_mdl <- c("MOHC","ETH","CNRM","KNMI","ICTP","HCLIMcom","KIT","CMCC","ENSEMBLE","STATIONS","EQM","QDM")
# list_mdl <- c("Stations","Ensemble","Ensemble EQM","Ensemble QDM")
# list_mdl <- c("Stations","Ensemble","Ensemble EQM","Ensemble QDM")
list_plots <- list()


for (mdl in list_mdl){
    if (mdl=="STATIONS") {
        sf <- fread(glue("{mdl}_q_{SEAS}_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }else if(mdl=="ENSEMBLE"){
        sf <- fread(glue("{mdl}_q_{SEAS}_biased_STATIONS_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }else if (mdl == "EQM" | mdl == "QDM") {
        sf <- fread(glue("ENSEMBLE_q_{SEAS}_{mdl}_STATIONS_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }else{
        sf <- fread(glue("{mdl}_q_{SEAS}_biased_SEQUENTIAL.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }
  make_maps(sf,0.5,'q',mdl,mdl,BBOX) -> list_plots[[mdl]]
}
length(list_plots)
tmap_arrange(list_plots, ncol = 3, nrow = 4)

for (mdl in list_mdl){
    if (mdl=="STATIONS") {
        # sf <- fread(glue("{mdl}_q_{SEAS}_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
        sf <- fread(glue("Stations_q_DJF_SEQUENTIAL.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }else if(mdl=="ENSEMBLE"){
        sf <- fread(glue("{mdl}_q_{SEAS}_QDM_STATIONS_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }else if (mdl == "EQM" | mdl == "QDM") {
        sf <- fread(glue("ENSEMBLE_q_{SEAS}_{mdl}_STATIONS_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }else{
        sf <- fread(glue("{mdl}_q_{SEAS}_QDM_SEQUENTIAL.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
    }
  make_maps(sf,0.5,'q',mdl,mdl,BBOX) -> list_plots[[mdl]]
}
length(list_plots)
tmap_arrange(list_plots, ncol = 3, nrow = 4)
