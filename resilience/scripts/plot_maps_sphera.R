#!/usr/bin/env Rscript

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

source("/home/lcesarini/2022_resilience/resilience/scripts/plot_function.R")

list_mdl <- c("MOHC","ETH","CNRM","KNMI","ICTP","HCLIMcom","KIT","CMCC","Ensemble")
list_mdl <- c("Stations","Ensemble","Ensemble EQM","Ensemble QDM")

list_sf <- c()
list_sf2 <- c()

mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
                                '#E85413','#E82313')

size_marker <- 0.5
for(mdl in list_mdl){
        if (M=="mean_prec") {
        
                if ("Ensemble"==mdl) {
                        sf <- fread("Ensemble_mean_prec_biased.csv") %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else if (grepl("EQM", mdl)){
                        sf <- fread("Ensemble_mean_prec_EQM.csv") %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else if (grepl("QDM", mdl)){
                        sf <- fread("Ensemble_mean_prec_QDM.csv") %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else{
                        sf <- fread("Stations_mean_prec_JJA.csv") %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }
        }else if (M=="q") {
                if (REF=="SPHERA") {
                        if ("Ensemble"==mdl) {
                                sf <- fread(glue("ENSEMBLE_q_{SEAS}_biased_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }else if (grepl("EQM", mdl)){
                                sf <- fread(glue("ENSEMBLE_q_{SEAS}_EQM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }else if (grepl("QDM", mdl)){
                                sf <- fread(glue("ENSEMBLE_q_{SEAS}_QDM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }else{
                                sf <- fread(glue("{REF}_q_{SEAS}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }
                }
                if (REF == "STATIONS") {
                        
                        if ("Ensemble"==mdl) {
                                sf <- fread(glue("ENSEMBLE_q_{SEAS}_biased_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }else if (grepl("EQM", mdl)){
                                sf <- fread(glue("ENSEMBLE_q_{SEAS}_EQM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }else if (grepl("QDM", mdl)){
                                sf <- fread(glue("ENSEMBLE_q_{SEAS}_QDM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }else{
                                sf <- fread(glue("STATIONS_q_{SEAS}_1000_{SPLIT}_{ADJUST}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                        }
                }
                
        }else if (M=="i") {
        
                if ("Ensemble"==mdl) {
                        sf <- fread(glue("Ensemble_i_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else if (grepl("EQM", mdl)){
                        sf <- fread(glue("Ensemble_i_{SEAS}_EQM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else if (grepl("QDM", mdl)){
                        sf <- fread(glue("Ensemble_i_{SEAS}_QDM.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else{
                        sf <- fread(glue("Stations_i_{SEAS}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }
                
        }

        list_sf2[[mdl]] <- sf        

        if(M=='mean_prec'){
                brks <- seq(0.1,0.275,0.025)
                lbls <- paste0(seq(0.1,0.275,0.05),"mm/hr")
        }else if (M=="q"){
                brks <- seq(2,20,2)
                lbls <- paste0(seq(2,20,2),"mm/hr")
        }else if (M=="i") {
                brks <- seq(0.3,3.3,0.3)
                lbls <- paste0(seq(0.3,3.0,0.3),"mm/hr")
        }
        tm_shape(sf)+
        tm_dots(col=ifelse(M=='mean_prec','mean',M),
                size=size_marker,
                palette=mycols,
                style="fixed",
                breaks=brks,
                labels=lbls,
                showNA=0,colorNA=NULL,legend.show=FALSE
                )+
        tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
        tm_layout(panel.labels=mdl,
                panel.show =TRUE,
                # title="Mean Prec.",
                title.bg.color="azure")+
        tm_add_legend('fill', 
	        col = mycols,
	        border.col = "black",
                is.portrait=TRUE,
	        # size = bubble_sizes,
	        labels = lbls,
	        title="[mm/hr]")  -> list_sf[[mdl]]

        } 

if (SAVE_MAPS_METRIC){

        tmap_save(tmap_arrange(list_sf,nrow=2,ncol=2),
                glue("../figures/my_map_q_{REF}_{SEAS}_{SPLIT}_{ADJUST}.png"), 
                width = 25, height = 21, units="cm", dpi = 300)
        # tmap_save(tmap_arrange(list_sf,nrow=3,ncol=3), "my_map.png", width = 21, height = 16, units="cm", dpi = 300)
}


if(M=='mean_prec'){
        list_sf2$Stations$bias_bia <- ((list_sf2[["Ensemble"]]$mean     - list_sf2$Stations$mean) / list_sf2$Stations$mean)
        list_sf2$Stations$bias_eqm <- ((list_sf2[["Ensemble EQM"]]$mean - list_sf2$Stations$mean) / list_sf2$Stations$mean)
        list_sf2$Stations$bias_qdm <- ((list_sf2[["Ensemble QDM"]]$mean - list_sf2$Stations$mean) / list_sf2$Stations$mean)
}else if (M=="q"){
        list_sf2$Stations$bias_bia <- ((list_sf2[["Ensemble"]]$q     - list_sf2$Stations$q) / list_sf2$Stations$q)
        list_sf2$Stations$bias_eqm <- ((list_sf2[["Ensemble EQM"]]$q - list_sf2$Stations$q) / list_sf2$Stations$q)
        list_sf2$Stations$bias_qdm <- ((list_sf2[["Ensemble QDM"]]$q - list_sf2$Stations$q) / list_sf2$Stations$q)
}else if (M=="i") {
        list_sf2$Stations$bias_bia <- ((list_sf2[["Ensemble"]]$i     - list_sf2$Stations$i) / list_sf2$Stations$i)
        list_sf2$Stations$bias_eqm <- ((list_sf2[["Ensemble EQM"]]$i - list_sf2$Stations$i) / list_sf2$Stations$i)
        list_sf2$Stations$bias_qdm <- ((list_sf2[["Ensemble QDM"]]$i - list_sf2$Stations$i) / list_sf2$Stations$i)
}

if(WITH_SS){
        boot_ori <- read.csv(glue("bias_boot_ori_{SEAS}_{REF}.csv"))
        boot_eqm <- read.csv(glue("bias_boot_eqm_{SEAS}_{REF}.csv"))
        boot_qdm <- read.csv(glue("bias_boot_qdm_{SEAS}_{REF}.csv"))

        boot_ori$SS <- ifelse(boot_ori$SS == 1, "RAW", NA)
        boot_eqm$SS <- ifelse(boot_eqm$SS == 1, "EQM", NA)
        boot_qdm$SS <- ifelse(boot_qdm$SS == 1, "QDM", NA)

        boot_ori %>%
        st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_boot_ori
        boot_eqm %>%
        st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_boot_eqm
        boot_qdm %>%
        st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_boot_qdm

        st_join(list_sf2$Stations,sf_boot_ori) %>% dplyr::filter(!is.na(SS)) %>% dplyr::filter(bias_bia >= 0) %>% pull(bias_bia) %>% mean() * 100 -> mean_bia_pos
        st_join(list_sf2$Stations,sf_boot_ori) %>% dplyr::filter(!is.na(SS)) %>% dplyr::filter(bias_bia <= 0) %>% pull(bias_bia) %>% mean() * 100 -> mean_bia_neg

        st_join(list_sf2$Stations,sf_boot_eqm) %>% dplyr::filter(!is.na(SS)) %>% dplyr::filter(bias_eqm >= 0) %>% pull(bias_eqm) %>% mean() * 100 -> mean_eqm_pos
        st_join(list_sf2$Stations,sf_boot_eqm) %>% dplyr::filter(!is.na(SS)) %>% dplyr::filter(bias_eqm <= 0) %>% pull(bias_eqm) %>% mean() * 100 -> mean_eqm_neg

        st_join(list_sf2$Stations,sf_boot_qdm) %>% dplyr::filter(!is.na(SS)) %>% dplyr::filter(bias_qdm >= 0) %>% pull(bias_qdm) %>% mean() * 100 -> mean_qdm_pos
        st_join(list_sf2$Stations,sf_boot_qdm) %>% dplyr::filter(!is.na(SS)) %>% dplyr::filter(bias_qdm <= 0) %>% pull(bias_qdm) %>% mean() * 100 -> mean_qdm_neg

        st_join(list_sf2$Stations,sf_boot_ori) %>% pull(bias_bia) %>% median(na.rm=TRUE) * 100 -> median_bia
        st_join(list_sf2$Stations,sf_boot_eqm) %>% pull(bias_eqm) %>% median(na.rm=TRUE) * 100 -> median_eqm
        st_join(list_sf2$Stations,sf_boot_qdm) %>% pull(bias_qdm) %>% median(na.rm=TRUE) * 100 -> median_qdm

        print("MEAN POSITIVE BIAS OF SSS")
        print(paste(    round(mean_bia_pos,2),
                        round(mean_eqm_pos,2),       
                        round(mean_qdm_pos,2)))
        print("MEAN NEGATIVE BIAS OF SSS")
        print(paste(    round(mean_bia_neg,2),
                        round(mean_eqm_neg,2),       
                        round(mean_qdm_neg,2)))
        print("MEDIAN BIASES")
        print(paste(round(median_bia,2),
                round(median_eqm,2),
                round(median_qdm,2)))


}

ll_bias <- list()
for(bs in c("bias_bia","bias_eqm","bias_qdm")){
        if (grepl("eqm",bs)) {
        title_plot <- "Ensemble EQM"
        list_sf2$Stations %>% dplyr::select(starts_with("bias")) %>% #dplyr::filter(!is.na(bias_eqm)) %>% View
        tm_shape()+
        tm_dots(col=bs,
                size=size_marker,
                palette=tmaptools::get_brewer_pal("PuOr",plot=FALSE),
                style="fixed",
                breaks=seq(-0.5,0.5,0.1),
                labels=paste0(seq(-0.5,0.5,0.1)*100,""),
                title="Bias [%]",legend.is.portrait=FALSE,
                showNA=0,colorNA=NULL,legend.show=FALSE
                        )+
                tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
                tm_layout(panel.labels=title_plot,
                        panel.show =TRUE,
                        legend.title.size = 1,
                        legend.text.size = 0.8,
                        title.bg.color="azure")+
                tm_add_legend('fill', 
                        col = RColorBrewer::brewer.pal(11, "PuOr"),
                        border.col = "black",
                        is.portrait=TRUE,
                        # size = bubble_sizes,
                        labels = paste0(seq(-0.5,0.5,0.1)*100,""),
                        title="Bias \n[%]") -> ll_bias[[bs]]           
                
                if (WITH_SS) {
                        add_SS <- tm_shape(sf_boot_eqm %>% dplyr::filter(!is.na(SS)))+
                                        tm_dots(col="SS",palette='black',size=0.15 * size_marker,legend.show=FALSE)
                
                        ll_bias[[bs]]  <- ll_bias[[bs]] + add_SS      
                }
        }else if(grepl("qdm",bs)){
        title_plot <- "Ensemble QDM"
        list_sf2$Stations %>% dplyr::select(starts_with("bias")) %>% #dplyr::filter(!is.na(bias_eqm)) %>% View
        tm_shape()+
        tm_dots(col=bs,
                size=size_marker,
                palette=tmaptools::get_brewer_pal("PuOr",plot=FALSE),
                style="fixed",
                breaks=seq(-0.5,0.5,0.1),
                labels=paste0(seq(-0.5,0.5,0.1)*100,""),
                title="Bias [%]",legend.is.portrait=FALSE,
                showNA=0,colorNA=NULL,legend.show=FALSE
                        )+
                tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
                tm_layout(panel.labels=title_plot,
                        panel.show =TRUE,
                        legend.title.size = 1,
                        legend.text.size = 0.8,
                        title.bg.color="azure")+
                tm_add_legend('fill', 
                        col = RColorBrewer::brewer.pal(11, "PuOr"),
                        border.col = "black",
                        is.portrait=TRUE,
                        # size = bubble_sizes,
                        labels = paste0(seq(-0.5,0.5,0.1)*100,""),
                        title="Bias \n[%]") -> ll_bias[[bs]]

                if (WITH_SS) {
                        add_SS <- tm_shape(sf_boot_qdm %>% dplyr::filter(!is.na(SS)))+
                                tm_dots(col="SS",palette='black',size=0.15 * size_marker,legend.show=FALSE)
                
                        ll_bias[[bs]]  <- ll_bias[[bs]] + add_SS      
                }
        }else{
        title_plot <- "Ensemble"
        list_sf2$Stations %>% dplyr::select(starts_with("bias")) %>% #dplyr::filter(!is.na(bias_eqm)) %>% View
        tm_shape()+
        tm_dots(col=bs,
                size=size_marker,
                palette=tmaptools::get_brewer_pal("PuOr",plot=FALSE),
                style="fixed",
                breaks=seq(-0.5,0.5,0.1),
                labels=paste0(seq(-0.5,0.5,0.1)*100,""),
                title="Bias [%]",legend.is.portrait=FALSE,
                showNA=0,colorNA=NULL,legend.show=FALSE
                        )+
                tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
                # tm_shape(sf_boot_ori %>% dplyr::filter(!is.na(SS)))+
                # tm_dots(col="SS",palette='black',size=0.25 * size_marker,legend.show=FALSE)+
                tm_layout(panel.labels=title_plot,
                        panel.show =TRUE,
                        legend.title.size = 1,
                        legend.text.size = 0.8,
                        title.bg.color="azure")+
                tm_add_legend('fill', 
                        col = RColorBrewer::brewer.pal(11, "PuOr"),
                        border.col = "black",
                        is.portrait=TRUE,
                        # size = bubble_sizes,
                        labels = paste0(seq(-0.5,0.5,0.1)*100,""),
                        title="Bias \n[%]") -> ll_bias[[bs]]

                if (WITH_SS) {
                        add_SS <- tm_shape(sf_boot_ori %>% dplyr::filter(!is.na(SS)))+
                                tm_dots(col="SS",palette='black',size=0.15 * size_marker,legend.show=FALSE)
                
                        ll_bias[[bs]]  <- ll_bias[[bs]] + add_SS      
                }

        }


}

if (WITH_SS){
        tmap_save(tmap_arrange(ll_bias,1,3), glue("../figures/my_{M}_bias_{REF}_{SEAS}_{SPLIT}_{ADJUST}_SS.png"), width = 32, height = 12, units="cm", dpi = 300)
}else{
        tmap_save(tmap_arrange(ll_bias,1,3), glue("../figures/my_{M}_bias_{REF}_{SEAS}_{SPLIT}_{ADJUST}.png"), width = 32, height = 12, units="cm", dpi = 300)
} 

ll_box <- list()
M <- 'q'
for(mdl in list_mdl){
        if (M=="mean_prec") {

                if ("Ensemble"==mdl) {
                        ll_box[[mdl]] <- fread("Ensemble_mean_prec_biased.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else if (grepl("EQM", mdl)){
                        ll_box[[mdl]] <- fread("Ensemble_mean_prec_EQM.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else if (grepl("QDM", mdl)){
                        ll_box[[mdl]] <- fread("Ensemble_mean_prec_QDM.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else{
                        ll_box[[mdl]] <- fread("Stations_mean_prec_JJA.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }
                               
        }else if (M=="q") {
                if (REF=="SPHERA") {
                        if ("Ensemble"==mdl) {
                                ll_box[[mdl]] <- fread(glue("ENSEMBLE_q_{SEAS}_biased_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }else if (grepl("EQM", mdl)){
                                ll_box[[mdl]] <- fread(glue("ENSEMBLE_q_{SEAS}_EQM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }else if (grepl("QDM", mdl)){
                                ll_box[[mdl]] <- fread(glue("ENSEMBLE_q_{SEAS}_QDM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }else{
                                ll_box[[mdl]] <- fread(glue("{REF}_q_{SEAS}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }
                }
                if (REF == "STATIONS") {            
                        if ("Ensemble"==mdl) {
                                ll_box[[mdl]] <- fread(glue("ENSEMBLE_q_{SEAS}_biased_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }else if (grepl("EQM", mdl)){
                                ll_box[[mdl]] <- fread(glue("ENSEMBLE_q_{SEAS}_EQM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }else if (grepl("QDM", mdl)){
                                ll_box[[mdl]] <- fread(glue("ENSEMBLE_q_{SEAS}_QDM_{REF}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }else{
                                ll_box[[mdl]] <- fread(glue("{REF}_q_{SEAS}_1000_{SPLIT}_{ADJUST}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                        }
                }

        }else if (M=="i") {
                if ("Ensemble"==mdl) {
                        ll_box[[mdl]] <- fread("Ensemble_i_JJA.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else if (grepl("EQM", mdl)){
                        ll_box[[mdl]] <- fread("Ensemble_i_JJA_EQM.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else if (grepl("QDM", mdl)){
                        ll_box[[mdl]] <- fread("Ensemble_i_JJA_QDM.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else{
                        ll_box[[mdl]] <- fread("Stations_i_JJA.csv") %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }

        }
}

str_title <- ifelse(M == 'q', "heavy precipitation","")
str_title=REF
list_sf2$Stations %>%
dplyr::select(starts_with("bias")) %>% 
rename(
        "RAW"="bias_bia",
        "EQM"="bias_eqm",
        "QDM"="bias_qdm"
)  %>% 
st_set_geometry(NULL) %>% 
reshape2::melt() %>% 
ggplot(aes(x=variable,fill=variable,y=value*100))+
geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
geom_jitter(width=0.2)+
labs(x='',y='Bias [%]',fill="",
     title=glue("Bias of {str_title} in {SEAS}"))+
     ylim(-100,100)+
theme_bw()+
theme(title       = element_text(size=35),
      axis.title  = element_text(size=25),
      axis.text   = element_text(size=22),
      legend.text = element_text(size=23),
      ) -> bb_1

ggsave(glue("../figures/myboxplotbias_{REF}_{M}_{SEAS}_{SPLIT}_{ADJUST}.png"),plot=bb_1)

     

# x1 <- fread(glue("ENSEMBLE_q_{SEAS}_EQM_SPHERA_1000_{SPLIT}_{ADJUST}.csv")) %>% dplyr::filter(!is.na(q))
# x2 <- fread(glue("ENSEMBLE_q_{SEAS}_QDM_SPHERA_1000_{SPLIT}_{ADJUST}.csv")) %>% dplyr::filter(!is.na(q))
# all(x1$q == x2$q) 
