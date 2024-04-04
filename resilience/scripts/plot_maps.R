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
m="q"
M="q"

source("/home/lcesarini/2022_resilience/resilience/scripts/plot_function.R")


make_maps(sf=fread(glue("SPHERA_q_DJF_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
        size_marker=0.5,M='q',
        title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1


make_maps(sf=fread(glue("ENSEMBLE_q_DJF_biased_SPHERA_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
        size_marker=0.5,M='q',
        title_panel=glue("RAW"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

make_maps(sf=fread(glue("ENSEMBLE_q_DJF_EQM_SPHERA_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
        size_marker=0.5,M='q',
        title_panel=glue("EQM"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map3

make_maps(sf=fread(glue("ENSEMBLE_q_DJF_QDM_SPHERA_1000_SEQUENTIAL_VALIDATION.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
        size_marker=0.5,M='q',
        title_panel=glue("QDM"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map4

tmap_save(tmap_arrange(list(map1,map2,map3,map4),nrow=2,ncol=2),
        glue("/home/lcesarini/2022_resilience/map_sphera_correction_{SEAS}_q999.png"), 
        width = 24, height = 16, units="cm", dpi = 450)

for (SEAS in c("SON","MAM","JJA","DJF")) {

        ##MAP ----
        min=fread(glue("/home/lcesarini/2022_resilience/csv/q99_station_all_{SEAS}.csv"))$q %>% summary() %>% .[c(2)] %>% as.numeric() * 0.9
        max=fread(glue("/home/lcesarini/2022_resilience/csv/q99_station_all_{SEAS}.csv"))$q %>% summary() %>% .[c(6)] %>% as.numeric() * 0.9

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/q99_station_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='q',brks=seq(min * 0.8,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/q99_sphera_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='q',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_q99.png"), 
                width = 24, height = 16, units="cm", dpi = 450)

        # BOXPLOT ----
        fread(glue("/home/lcesarini/2022_resilience/csv/q99_station_all_{SEAS}.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/q99_sphera_all_{SEAS}.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("q")) %>% 
                melt() %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Precipitation [mm/hr]',fill="",
                title=glue("Heavy Precip. 99th percentile in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title       = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_q99_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450,)


        min=fread(glue("/home/lcesarini/2022_resilience/csv/q999_station_all_{SEAS}.csv"))$q %>% summary() %>% .[c(2)] %>% as.numeric() * 0.9
        max=fread(glue("/home/lcesarini/2022_resilience/csv/q999_station_all_{SEAS}.csv"))$q %>% summary() %>% .[c(6)] %>% as.numeric() * 0.9

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/q999_station_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='q',brks=seq(min * 0.8,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/q999_sphera_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='q',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_q999.png"), 
                width = 24, height = 16, units="cm", dpi = 450)

                # BOXPLOT ----
        fread(glue("/home/lcesarini/2022_resilience/csv/q999_station_all_{SEAS}.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/q999_sphera_all_{SEAS}.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("q")) %>% 
                melt() %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Precipitation [mm/hr]',fill="",
                title=glue("Heavy Precip. 99.9th percentile in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title       = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_q999_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450,)

}
#FREQUENCY of WETHOURS
for (SEAS in c("SON","MAM","JJA","DJF")) {

        min=fread(glue("/home/lcesarini/2022_resilience/csv/freq_station_all_{SEAS}.csv"))$f %>% summary() %>% .[c(2)] %>% as.numeric() * 0.9
        max=fread(glue("/home/lcesarini/2022_resilience/csv/freq_station_all_{SEAS}.csv"))$f %>% summary() %>% .[c(6)] %>% as.numeric() * 0.9

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/freq_station_all_{SEAS}.csv")) %>% mutate(f=ifelse(f != 0,f,NA)) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='f',brks=seq(min * 0.8,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2) * 100," [%]"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/freq_sphera_all_{SEAS}.csv")) %>% mutate(f=ifelse(f != 0,f,NA)) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='f',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2) * 100," [%]"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_freq.png"), 
                width = 24, height = 16, units="cm", dpi = 450)

        fread(glue("/home/lcesarini/2022_resilience/csv/freq_station_all_{SEAS}.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/freq_sphera_all_{SEAS}.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("f")) %>% 
                melt() %>% 
                mutate(value=ifelse(value==0,NA,value)) %>% 
                ggplot(aes(x=variable,fill=variable,y=value * 100))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Frequency [%]',fill="",
                title=glue("Frequency wethours in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title       = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_freq_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450,)

        min=fread(glue("/home/lcesarini/2022_resilience/csv/mean_intensity_station_all_{SEAS}.csv"))$i %>% summary() %>% .[c(2)] %>% as.numeric() * 0.9
        max=fread(glue("/home/lcesarini/2022_resilience/csv/mean_intensity_station_all_{SEAS}.csv"))$i %>% summary() %>% .[c(6)] %>% as.numeric() * 0.9

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/mean_intensity_station_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='i',brks=seq(min * 0.8,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/mean_intensity_sphera_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='i',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_intensity.png"), 
                width = 24, height = 16, units="cm", dpi = 450)
        
        fread(glue("/home/lcesarini/2022_resilience/csv/mean_intensity_station_all_{SEAS}.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/mean_intensity_sphera_all_{SEAS}.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("i")) %>% 
                melt() %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Intensity [mm/hr]',fill="",
                title=glue("Intensity of wethours in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_inte_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450,)

}
## WETHOURS PREC ---
for (SEAS in c("SON","MAM","JJA","DJF")) {

        min=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_99_on_WH_station_all_{SEAS}.csv"))$i %>% min(na.rm=TRUE)
        max=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_99_on_WH_station_all_{SEAS}.csv"))$i %>% max(na.rm=TRUE)

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_99_on_WH_station_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326) ,
                                                size_marker=0.5,M='i',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_99_on_WH_sphera_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='i',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_wh_99.png"), 
                width = 24, height = 16, units="cm", dpi = 450)

        fread(glue("/home/lcesarini/2022_resilience/csv/heavy_99_on_WH_station_all_{SEAS}.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/heavy_99_on_WH_sphera_all_{SEAS}.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("i")) %>% 
                melt() %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Precipitation [mm/hr]',fill="",
                title=glue("Heavy Prec. 99th of WH in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_99_WH_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450,)


        min=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_999_on_WH_station_all_{SEAS}.csv"))$i %>% min(na.rm=TRUE)
        max=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_999_on_WH_station_all_{SEAS}.csv"))$i %>% max(na.rm=TRUE)

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_999_on_WH_station_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326) ,
                                                size_marker=0.5,M='i',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/heavy_999_on_WH_sphera_all_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='i',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_wh_999.png"), 
                width = 24, height = 16, units="cm", dpi = 450)

        fread(glue("/home/lcesarini/2022_resilience/csv/heavy_999_on_WH_station_all_{SEAS}.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/heavy_999_on_WH_sphera_all_{SEAS}.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("i")) %>% 
                melt() %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Precipitation [mm/hr]',fill="",
                title=glue("Heavy Prec. 99.9th of WH in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_999_WH_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450,)


}


# MEAN DAILY PRECIPITATION ----
for (SEAS in c("SON","MAM","JJA","DJF")) {

        ##MAP ----
        min=fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_station_all.csv"))$pr %>% summary() %>% .[c(2)] %>% as.numeric() * 0.9
        max=fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_station_all.csv"))$pr %>% summary() %>% .[c(6)] %>% as.numeric() * 0.9 

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_station_all.csv")) %>% mutate(pr=ifelse(pr != 0,pr,NA)) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='pr',brks=seq(min * 0.8,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map1

        make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_sphera_all.csv")) %>% mutate(pr=ifelse(pr != 0,pr,NA)) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
                                                size_marker=0.5,M='pr',brks=seq(min,max,(max-min)/9),lbls=paste(round(seq(min,max,(max-min)/9),2)," mm/hr"),
                                                title_panel=glue("SPHERA"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> map2

        tmap_save(tmap_arrange(list(map1,map2),nrow=1,ncol=2),
                glue("/home/lcesarini/2022_resilience/map_sphera_stations_{SEAS}_mean_prec.png"), 
                width = 24, height = 16, units="cm", dpi = 450)

        fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_station_all.csv")) %>% 
                left_join(.,fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_sphera_all.csv")),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains("pr")) %>% 
                melt() %>% 
                mutate(value=ifelse(value==0,NA,value)) %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                labs(x='',y='Precipitation [mm]',fill="",
                title=glue("Mean daily precipitation in {SEAS}"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plt1
        ggsave(glue("/home/lcesarini/2022_resilience/boxplot_mean_prec_{SEAS}.png"),device='png',plot = plt1,height=16,width=24,units = c("cm"),dpi = 450)

}
# fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_station_all.csv")) %>% 
#         mutate(pr=ifelse(pr != 0,pr,NA)) %>% 
#         left_join(.,(fread(glue("/home/lcesarini/2022_resilience/csv/{SEAS}_mean_prec_sphera_all.csv"))  %>% mutate(pr=ifelse(pr != 0,pr,NA))),by=c('lat','lon')) %>% 
#         dplyr::select(tidyselect::contains("pr")) %>% 
#         melt() %>% 
#         ggplot()+
#         geom_boxplot(aes(y=value,fill=variable),width=0.1)+
#         theme_bw()
# #### ----

# for (SEAS in c("SON","MAM","JJA","DJF")) {
#         for (SPLIT in c("SEQUENTIAL","RANDOM")) {

#                 list_sf <- list()
#                 make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/Stations_q_{SEAS}_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                                                         size_marker=0.5,M='q',
#                                                         title_panel=glue("STATIONS"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[1]]

#                 make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/Ensemble_q_{SEAS}_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                                                         size_marker=0.5,M='q',
#                                                         title_panel=glue("ENSEMBLE RAW"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[2]]

#                 make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/Ensemble_q_{SEAS}_EQM_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                                                         size_marker=0.5,M='q',
#                                                         title_panel=glue("ENSEMBLE EQM"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[3]]

#                 make_maps(sf=fread(glue("/home/lcesarini/2022_resilience/csv/Ensemble_q_{SEAS}_QDM_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                                                         size_marker=0.5,M='q',
#                                                         title_panel=glue("ENSEMBLE QDM"),bbox=st_bbox(fread(glue("Ensemble_q_DJF_RAW_RANDOM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[4]]


#                 tmap_save(tmap_arrange(list_sf,nrow=2,ncol=2),
#                         glue("map_ensemble_{SEAS}_{SPLIT}.png"), 
#                         width = 24, height = 16, units="cm", dpi = 450)
#         }
# }



# df_coords=read.csv("/home/lcesarini/2022_resilience/csv/bias_boot.csv")
# sf::st_as_sf(df_coords,coords=c("lon","lat"),crs=4326) -> sf_obj
# make_maps(sf=fread("/home/lcesarini/2022_resilience/csv/bias_boot.csv") %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                                         size_marker=0.5,M='SS',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                                         title_panel=glue("{mdl} {bc}"),bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)))
# #PLOT GRIPHO
# df_coords=read.csv("heavy_prec_gripho.csv")
# sf::st_as_sf(df_coords,coords=c("lon","lat"),crs=4326) -> sf_obj

# ds_gri=raster(glue("output/{SEAS}/GRIPHO_ORIGINAL_{m}.nc"))

# ds_sta=raster(glue("output/{SEAS}/STATIONS_{m}.nc"))
# ds_cpm=raster(glue("output/{SEAS}/ENSEMBLE_{m}.nc")) 

# ds_sph=raster(glue("output/{SEAS}/SPHERA_{m}.nc"))

# ds_vhr=raster(glue("output/{SEAS}/CMCC_VHR_{m}.nc"))

# shp_triveneto = sf::st_read("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1") 
# shp_triveneto <- shp_triveneto %>% dplyr::filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))

# mask=raster("data/mask_stations_nan_common.nc")

# sea_mask=raster("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/CNRM/sftlf_ALP-3_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-AROME41t1_fx_remap.nc")


# rea_sph=crop((ds_sph*sea_mask$sftlf),ds_sta)

# rea_vhr=crop((ds_vhr*sea_mask$sftlf),ds_sta)

# cpm_tri=crop((ds_cpm*sea_mask$sftlf),ds_sta)


# #BIAS CPM-STATION
# bias_cpm_sta=((cpm_tri - ds_sta) / ds_sta) * 100
# bias_sph_sta=((rea_sph - ds_sta) / ds_sta) * 100
# bias_cpm_sph

# coords <- xyFromCell(ds_sta,1:ncell(ds_sta))

# values <- extract(ds_gri,coords)

# library(RColorBrewer)
# blups <- brewer.pal(11, "PuOr")
# breaks=seq(-50,50,10)

# cbind(coords[!is.na(values),],values[!is.na(values)]) %>% 
# data.frame() -> df_obj


# sf::st_as_sf(df_obj,coords=c("x","y"),crs=4326) -> sf_obj

# tm_shape(sf_obj)+
#   tm_dots(col='quantile',
#           size=5.5,
#           palette=mycols,
#           style="fixed",
#           breaks=seq(2,20,2),
#           labels=paste0(seq(2,20,2),"mm/hr"),
#           showNA=0,colorNA=NULL
#           )+
#   tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#   tm_layout(panel.labels="GRIPHO",
#           panel.show =TRUE,
#           title="Heavy Prec.",
#           title.bg.color="azure")  -> tm4


# colnames(sf_obj)[1] <- "values"

# tm_1 <- tm_shape(sf_obj)+tm_dots(size=5,col='values',
#                          colorNA=NA,
#                          palette=blups,
#                          breaks=breaks,
#                          title.size = 2,
#                          legend.show = FALSE,
#                          legend.hist=1)+
# tm_shape(shp_triveneto)+tm_borders(col='green')+
# # tm_layout(frame=TRUE, legend.show = TRUE,title = "",title.fontfamily = "Calibri",legend.only = TRUE,
# #             legend.outside = TRUE, legend.outside.position = "right",legend.title.fontface = "bold",
# #             legend.text.fontfamily = "Calibri")+
# tm_add_legend(type = "fill", col=blups,
#                 labels = breaks,
#                 title="Bias [%]",is.portrait = FALSE) +
# tm_layout(frame=TRUE,frame.double.line = TRUE, 
#         #   legend.outside.size = 0.1, legend.show = FALSE,
#         #   inner.margins = c(0.025,0.025,0.025,0.025),
#         #   outer.margins = c(0.05,0.08,0.05,0.05),
#           legend.outside=0,
#           legend.position=c(0.0075,0),
#           legend.width=1,
#           legend.hist.width=.25
#           )
                  

# tmap_save(tm=tm_1, filename = "figures/bias_cpm_sph.png",
#           width = 20,height = 24, unit = "cm", dpi = 300,
#           outer.margins = c(0,0,0,0)
#           )

# par(mfrow=c(1,2))
# boxplot(bias_cpm_sta,ylim=c(-80,80))
# boxplot(bias_sph_sta,ylim=c(-80,80))

