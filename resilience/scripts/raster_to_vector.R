¯#!/usr/bin/env Rscript
suppressPackageStartupMessages(suppressWarnings(library(sf)))
suppressPackageStartupMessages(suppressWarnings(library(glue)))
suppressPackageStartupMessages(suppressWarnings(library(tmap)))
suppressPackageStartupMessages(suppressWarnings(library(dplyr)))
suppressPackageStartupMessages(suppressWarnings(library(raster)))
suppressPackageStartupMessages(suppressWarnings(library(ggplot2)))
suppressPackageStartupMessages(suppressWarnings(library(data.table)))

defaultW <- getOption("warn") 
options(warn = -1) 

setwd("/home/lcesarini/2022_resilience/csv")

args = commandArgs(trailingOnly=TRUE)

shp_triveneto = st_read("../data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")  %>% 
                        dplyr::filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))

`%!in%` <- Negate(`%in%`)
M=args[1] #q,i
# read argument from command line
SEAS=args[2]
SPLIT = args[3]#"RANDOM"

print(paste(args[1],args[2]))

if (M=='f') {
        mycols <- c('#B1DFFA','#36BCFF', '#508D5E','#55CB70','#E5E813','#E8AB13','#E85413','#E82313')
}else if(M=='i'){
        mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
                    '#E85413','#E82313')
}else{
        mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
                    '#E85413','#E82313')

}

source("../resilience/scripts/plot_function.R")

list_mdl <- c("MOHC","ETH","CNRM","KNMI","ICTP","HCLIMcom","KIT","CMCC","Ensemble")
list_mdl <- c("Stations","Ensemble","Ensemble EQM","Ensemble QDM")

list_sf <- c()
list_sf2 <- c()


# for(mdl in c("ICTP","ETH")){list_sf[[mdl]] <- fread(glue("{mdl}_{M}_biased.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)} 
# counter <- 0
# for(mdl in c("ICTP","ETH")){
#         for (bc in c("RAW", "EQM", "QDM")) {
#                 counter <- counter+1
#                 list_sf[[counter]] <- make_maps(sf=fread(glue("{mdl}_{M}_{SEAS}_{bc}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                                         title_panel=glue("{mdl} {bc}"),bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)))
#         }
# } 

# tmap_save(tmap_arrange(list_sf,nrow=2,ncol=3), glue("ICTP_ETH_{M}_{SEAS}.png"), width = 32, height = 12, units="cm", dpi = 300)

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
        
                if ("Ensemble"==mdl) {
                        sf <- fread(glue("Ensemble_q_{SEAS}_RAW_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else if (grepl("EQM", mdl)){
                        sf <- fread(glue("Ensemble_q_{SEAS}_EQM_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else if (grepl("QDM", mdl)){
                        sf <- fread(glue("Ensemble_q_{SEAS}_QDM_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
                }else{
                        sf <- fread(glue("Stations_q_{SEAS}_{SPLIT}.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326)
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

tmap_save(tmap_arrange(list_sf,nrow=2,ncol=2),
          glue("../figures/my_map_q_{SEAS}_{SPLIT}.png"), 
          width = 25, height = 21, units="cm", dpi = 300)
# tmap_save(tmap_arrange(list_sf,nrow=3,ncol=3), "my_map.png", width = 21, height = 16, units="cm", dpi = 300)


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


boot_ori <- read.csv(glue("bias_boot_ori_{SEAS}.csv"))
boot_eqm <- read.csv(glue("bias_boot_eqm_{SEAS}.csv"))
boot_qdm <- read.csv(glue("bias_boot_qdm_{SEAS}.csv"))

boot_ori$SS <- ifelse(boot_ori$SS == 1, "RAW", NA)
boot_eqm$SS <- ifelse(boot_eqm$SS == 1, "EQM", NA)
boot_qdm$SS <- ifelse(boot_qdm$SS == 1, "QDM", NA)

boot_ori %>%
  st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_boot_ori
boot_eqm %>%
  st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_boot_eqm
boot_qdm %>%
  st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_boot_qdm


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
                # tm_shape(sf_boot_eqm %>% dplyr::filter(!is.na(SS)))+
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
                tm_shape(sf_boot_qdm %>% dplyr::filter(!is.na(SS)))+
                tm_dots(col="SS",palette='black',size=0.25 * size_marker,legend.show=FALSE)+
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
                tm_shape(sf_boot_ori %>% dplyr::filter(!is.na(SS)))+
                tm_dots(col="SS",palette='black',size=0.25 * size_marker,legend.show=FALSE)+
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
        }


}
tmap_save(tmap_arrange(ll_bias,1,3), glue("../figures/my_{M}_bias_{SEAS}_{SPLIT}_SS.png"), width = 32, height = 12, units="cm", dpi = 300)

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
                if ("Ensemble"==mdl) {
                        ll_box[[mdl]] <- fread(glue("Ensemble_q_{SEAS}_RAW_{SPLIT}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else if (grepl("EQM", mdl)){
                        ll_box[[mdl]] <- fread(glue("Ensemble_q_{SEAS}_EQM_{SPLIT}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else if (grepl("QDM", mdl)){
                        ll_box[[mdl]] <- fread(glue("Ensemble_q_{SEAS}_QDM_{SPLIT}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
                }else{
                        ll_box[[mdl]] <- fread(glue("Stations_q_{SEAS}_{SPLIT}.csv")) %>%  rename(!!glue("{mdl}"):=ifelse(M=='mean_prec','mean',M))
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

# ll_box[[1]]

# plyr::join_all(ll_box,by=c("lon","lat")) %>% 
# reshape2::melt(id.vars=c("lon","lat")) %>% 
# ggplot(aes(x=variable,fill=variable,y=value))+
# geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
# geom_jitter(width=0.2)+
# labs(x='',y='Mean Precipitation [mm/hr]',fill="",
#      title="Mean Precipitation in JJA")+
# theme_bw()+
# theme(title       = element_text(size=35),
#       axis.title  = element_text(size=25),
#       axis.text   = element_text(size=22),
#       legend.text = element_text(size=23),
#       )


# list_sf2$Stations %>%
# dplyr::select(starts_with("bias")) %>% 
# rename(
#         "RAW"="bias_bia",
#         "EQM"="bias_eqm",
#         "QDM"="bias_qdm"
# ) %>% 
# st_set_geometry(NULL) %>% 
# reshape2::melt() %>% 
# ggplot(aes(x=variable,fill=variable,y=value*100))+
# geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
# geom_jitter(width=0.2)+
# labs(x='',y='Bias [%]',fill="",
#      title=glue( "Bias of Heavy Precipitation in {SEAS}"))+
# theme_bw()+
# theme(title       = element_text(size=35),
#       axis.title  = element_text(size=25),
#       axis.text   = element_text(size=22),
#       legend.text = element_text(size=23),
#       ) -> bb_1

# ggsave(glue("../figures/myboxplotbia{M}_{SEAS}_{SPLIT}.png"),plot=bb_1)

str_title <- ifelse(M == 'q', "heavy precipitation","")
list_sf2$Stations %>%
dplyr::select(starts_with("bias")) %>% 
rename(
        "RAW"="bias_bia",
        "EQM"="bias_eqm",
        "QDM"="bias_qdm"
) %>% 
st_set_geometry(NULL) %>% 
reshape2::melt() %>% 
ggplot(aes(x=variable,fill=variable,y=value*100))+
geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
geom_jitter(width=0.2)+
labs(x='',y='Bias [%]',fill="",
     title=glue("Bias of {str_title} in JJA"))+
     ylim(-100,100)+
theme_bw()+
theme(title       = element_text(size=35),
      axis.title  = element_text(size=25),
      axis.text   = element_text(size=22),
      legend.text = element_text(size=23),
      ) -> bb_1

ggsave(glue("../figures/myboxplotbias_{M}_{SEAS}_{SPLIT}.png"),plot=bb_1)
     


# ens_sta <- fread(glue("Ensemble_{M}_{SEAS}.csv"))
# moh_sta <- fread(glue("MOHC_q_biased.csv"))s
# rea_sta <- fread(glue("rea_sta_{SEAS}_{M}.csv"))
# vhr_sta <- fread(glue("vhr_sta_{SEAS}_{M}.csv"))
# sta <- fread(glue("Stations_{M}_{SEAS}.csv"))
# df_coords=read.csv("gripho_q_JJA.csv") %>% dplyr::select(-1)


# tm <- list()
# tmap_mode("plot")

# ens_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_ens

# moh_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_moh

# tm_shape(sf_moh)+
#         tm_dots(col='q',
#                 size=size_marker,
#                 palette=mycols,
#                 style="fixed",
#                 breaks=seq(2,20,2),
#                 labels=paste0(seq(2,20,2),"mm/hr"),
#                 showNA=0,colorNA=NULL
#                 )+
#         tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#         tm_layout(panel.labels="SPHERA",
#                 panel.show =TRUE,
#                 title="Heavy Prec.",
#                 title.bg.color="azure") 

# rea_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_rea

# vhr_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_vhr

# sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_sta

# df_coords %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_gri

# ####PLOT THE SINGLE METRICS ----
# # ENSEMBLE ----
# size_marker = 0.25
# if (M=="f") {
#    tm_shape(sf_ens)+
#         tm_dots(col='f',
#                 size=size_marker,
#                 palette=mycols,
#                 style="fixed",
#                 breaks=seq(0.04,0.28,0.03),
#                 labels=paste0(seq(2.4,18.5,2),"%"),
#                 showNA=0,colorNA=NULL
#                 )+
#         tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#         tm_layout(panel.labels="Ensemble",
#                 panel.show =TRUE,
#                 title="Freq.",
#                 title.bg.color="azure")  -> tm1
# }else if (M=='i') {
#         tm_shape(sf_ens)+
#           tm_dots(col='i',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(0.3,3.3,0.3),
#                  labels=paste0(seq(0.3,3.3,0.3),"mm/hr%"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="Ensemble",
#                  panel.show =TRUE,
#                  title="Int.",
#                  title.bg.color="azure")  -> tm1
   
# }else {
#         tm_shape(sf_ens)+
#           tm_dots(col='q',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(2,20,2),
#                  labels=paste0(seq(2,20,2),"mm/hr"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="Ensemble",
#                  panel.show =TRUE,
#                  title="Heavy Prec.",
#                  title.bg.color="azure")  -> tm1
# }
# # VHR CMCC ----
# if (M=="f") {
#    tm_shape(sf_vhr)+
#         tm_dots(col='f',
#                 size=size_marker,
#                 palette=mycols,
#                 style="fixed",
#                 breaks=seq(0.04,0.28,0.03),
#                 labels=paste0(seq(2.4,18.5,2),"%"),
#                 showNA=0,colorNA=NULL
#                 )+
#         tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#         tm_layout(panel.labels="VHR-CMCC",
#                 panel.show =TRUE,
#                 title="Freq.",
#                 title.bg.color="azure")  -> tm2
# }else if (M=='i') {
#         tm_shape(sf_vhr)+
#           tm_dots(col='i',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(0.3,3.3,0.3),
#                  labels=paste0(seq(0.3,3.3,0.3),"mm/hr%"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="VHR-CMCC",
#                  panel.show =TRUE,
#                  title="Int.",
#                  title.bg.color="azure")  -> tm2
   
# }else {
#         tm_shape(sf_vhr)+
#           tm_dots(col='q',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(2,20,2),
#                  labels=paste0(seq(2,20,2),"mm/hr"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="VHR-CMCC",
#                  panel.show =TRUE,
#                  title="Heavy Prec.",
#                  title.bg.color="azure")  -> tm2
# }

# # REANALYSIS ----
# if (M=="f") {
#    tm_shape(sf_rea)+
#         tm_dots(col='f',
#                 size=size_marker,
#                 palette=mycols,
#                 style="fixed",
#                 breaks=seq(0.04,0.28,0.03),
#                 labels=paste0(seq(2.4,18.5,2),"%"),
#                 showNA=0,colorNA=NULL
#                 )+
#         tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#         tm_layout(panel.labels="SPHERA",
#                 panel.show =TRUE,
#                 title="Freq.",
#                 title.bg.color="azure")  -> tm3
# }else if (M=='i') {
#         tm_shape(sf_rea)+
#           tm_dots(col='i',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(0.3,3.3,0.3),
#                  labels=paste0(seq(0.3,3.3,0.3),"mm/hr%"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="SPHERA",
#                  panel.show =TRUE,
#                  title="Int.",
#                  title.bg.color="azure")  -> tm3
   
# }else {
#         tm_shape(sf_rea)+
#           tm_dots(col='q',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(2,20,2),
#                  labels=paste0(seq(2,20,2),"mm/hr"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="SPHERA",
#                  panel.show =TRUE,
#                  title="Heavy Prec.",
#                  title.bg.color="azure")  -> tm3
# }


# # STATIONS ----
# if (M=="f") {
#    tm_shape(sf_sta)+
#         tm_dots(col='f',
#                 size=size_marker,
#                 palette=mycols,
#                 style="fixed",
#                 breaks=seq(0.04,0.28,0.03),
#                 labels=paste0(seq(2.4,18.5,2),"%"),
#                 showNA=0,colorNA=NULL
#                 )+
#         tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#         tm_layout(panel.labels="Stations",
#                 panel.show =TRUE,
#                 title="Freq.",
#                 title.bg.color="azure")  -> tm4
# }else if (M=='i') {
#         tm_shape(sf_sta)+
#           tm_dots(col='i',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(0.3,3.3,0.3),
#                  labels=paste0(seq(0.3,3.3,0.3),"mm/hr%"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="Stations",
#                  panel.show =TRUE,
#                  title="Int.",
#                  title.bg.color="azure")  -> tm4
   
# }else {
#         tm_shape(sf_sta)+
#           tm_dots(col='q',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(2,20,2),
#                  labels=paste0(seq(2,20,2),"mm/hr"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="Stations",
#                  panel.show =TRUE,
#                  title="Heavy Prec.",
#                  title.bg.color="azure")  -> tm4
# }

# # GRIPHO ----
# if (M=="f") {
#    tm_shape(sf_gri)+
#         tm_dots(col='f',
#                 size=size_marker,
#                 palette=mycols,
#                 style="fixed",
#                 breaks=seq(0.04,0.28,0.03),
#                 labels=paste0(seq(2.4,18.5,2),"%"),
#                 showNA=0,colorNA=NULL
#                 )+
#         tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#         tm_layout(panel.labels="GRIPHO",
#                 panel.show =TRUE,
#                 title="Freq.",
#                 title.bg.color="azure")  -> tm5
# }else if (M=='i') {
#         tm_shape(sf_gri)+
#           tm_dots(col='i',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(0.3,3.3,0.3),
#                  labels=paste0(seq(0.3,3.3,0.3),"mm/hr%"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="GRIPHO",
#                  panel.show =TRUE,
#                  title="Int.",
#                  title.bg.color="azure")  -> tm5
   
# }else {
#         tm_shape(sf_gri)+
#           tm_dots(col='q',
#                  size=size_marker,
#                  palette=mycols,
#                  style="fixed",
#                  breaks=seq(2,20,2),
#                  labels=paste0(seq(2,20,2),"mm/hr"),
#                  showNA=0,colorNA=NULL
#                  )+
#           tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
#           tm_layout(panel.labels="GRIPHO",
#                  panel.show =TRUE,
#                  title="Heavy Prec.",
#                  title.bg.color="azure")  -> tm5
# }

# tmap_save(tmap_arrange(list(tm1,tm2,tm3,tm4,tm5),nrow=1,ncol=5), "my_map.png", width = 24, height = 15, units="cm", dpi = 300)


# ####PLOT THE SPATIAL VARIABILITY OF THE SINGLE METRICS ----

# sf_ens %>% 
#         mutate(sv_rea=f/sf_rea[,M],
#                sv_sta=f/sf_sta[,M]) -> sf_sv 
# tm_shape(sf_sv)+
#         tm_dots(col="sv_rea",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# tm_shape(sf_sv)+
#         tm_dots(col="sv_sta",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# sd(sf_sv$sv_rea,na.rm=TRUE) / mean(sf_sv$sv_rea, na.rm=TRUE)
# sd(sf_sv$sv_sta,na.rm=TRUE) / mean(sf_sv$sv_sta, na.rm=TRUE)



# ens <- read.csv("/home/lcesarini/2022_resilience/ensemble.csv")
# rea <- read.csv("/home/lcesarini/2022_resilience/reanalysis.csv")
# library(plotrix)
# ref<-rnorm(30,sd=2) 
# #addalittlenoise 
# model1<-ref+rnorm(30)/2 
# #addmorenoise 
# model2<-ref+rnorm(30) 
# #displaythediagramwiththebettermodel 
# oldpar<-taylor.diagram(ref,model1) 
# #nowaddtheworsemodel 
# taylor.diagram(ens,rea,add=TRUE,col="blue") 
# #getapproximatelegendposition 
# # lpos<-1.5*sd(ref) 
# # #addalegend 
# # legend(lpos,lpos,legend=c("Better","Worse"),pch=19,col=c("red","blue")) 
# # #nowrestoreparvalues 
# # par(oldpar) 
# # #showthe"allcorrelation"display 
# # taylor.diagram(ref,model1,pos.cor=FALSE) 
# # taylor.diagram(ref,model2,add=TRUE,col="blue")
# shp_triveneto = st_read("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")  %>% 
#                         filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))

# M="q"

# ens_sta <- fread(glue("ens_sta_{M}.csv"))
# rea_sta <- fread(glue("rea_sta_{M}.csv"))
# sta <- fread(glue("sta_{M}.csv"))

# tm <- list()
# tmap_mode("plot")

# ens_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_ens

# rea_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_rea

# sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_sta

# ####PLOT THE SINGLE METRICS ----
# tm_shape(sf_ens)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Ensemble",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm1



# tm_shape(sf_rea)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="SPHERA",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm2


# tm_shape(sf_sta)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Stations",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure"
#           )  -> tm3

# tmap_save(tmap_arrange(list(tm1,tm2,tm3),3,1), "my_map.png", width = 24, height = 15, units="cm", dpi = 300)


# ####PLOT THE SPATIAL VARIABILITY OF THE SINGLE METRICS ----

# sf_ens %>% 
#         mutate(sv_rea=f/sf_rea[,M],
#                sv_sta=f/sf_sta[,M]) -> sf_sv 
# tm_shape(sf_sv)+
#         tm_dots(col="sv_rea",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# tm_shape(sf_sv)+
#         tm_dots(col="sv_sta",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# sd(sf_sv$sv_rea,na.rm=TRUE) / mean(sf_sv$sv_rea, na.rm=TRUE)
# sd(sf_sv$sv_sta,na.rm=TRUE) / mean(sf_sv$sv_sta, na.rm=TRUE)



# ens <- read.csv("/home/lcesarini/2022_resilience/ensemble.csv")
# rea <- read.csv("/home/lcesarini/2022_resilience/reanalysis.csv")
# library(plotrix)
# ref<-rnorm(30,sd=2) 
# #addalittlenoise 
# model1<-ref+rnorm(30)/2 
# #addmorenoise 
# model2<-ref+rnorm(30) 
# #displaythediagramwiththebettermodel 
# oldpar<-taylor.diagram(ref,model1) 
# #nowaddtheworsemodel 
# taylor.diagram(ens,rea,add=TRUE,col="blue") 
# #getapproximatelegendposition 
# # lpos<-1.5*sd(ref) 
# # #addalegend 
# # legend(lpos,lpos,legend=c("Better","Worse"),pch=19,col=c("red","blue")) 
# # #nowrestoreparvalues 
# # par(oldpar) 
# # #showthe"allcorrelation"display 
# # taylor.diagram(ref,model1,pos.cor=FALSE) 
# # taylor.diagram(ref,model2,add=TRUE,col="blue")
# shp_triveneto = st_read("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")  %>% 
#                         filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))

# M="q"

# ens_sta <- fread(glue("ens_sta_{M}.csv"))
# rea_sta <- fread(glue("rea_sta_{M}.csv"))
# sta <- fread(glue("sta_{M}.csv"))

# tm <- list()
# tmap_mode("plot")

# ens_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_ens

# rea_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_rea

# sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_sta

# ####PLOT THE SINGLE METRICS ----
# tm_shape(sf_ens)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Ensemble",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm1



# tm_shape(sf_rea)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="SPHERA",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm2


# tm_shape(sf_sta)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Stations",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure"
#           )  -> tm3

# tmap_save(tmap_arrange(list(tm1,tm2,tm3),3,1), "my_map.png", width = 24, height = 15, units="cm", dpi = 300)


# ####PLOT THE SPATIAL VARIABILITY OF THE SINGLE METRICS ----

# sf_ens %>% 
#         mutate(sv_rea=f/sf_rea[,M],
#                sv_sta=f/sf_sta[,M]) -> sf_sv 
# tm_shape(sf_sv)+
#         tm_dots(col="sv_rea",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# tm_shape(sf_sv)+
#         tm_dots(col="sv_sta",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# sd(sf_sv$sv_rea,na.rm=TRUE) / mean(sf_sv$sv_rea, na.rm=TRUE)
# sd(sf_sv$sv_sta,na.rm=TRUE) / mean(sf_sv$sv_sta, na.rm=TRUE)



# ens <- read.csv("/home/lcesarini/2022_resilience/ensemble.csv")
# rea <- read.csv("/home/lcesarini/2022_resilience/reanalysis.csv")
# library(plotrix)
# ref<-rnorm(30,sd=2) 
# #addalittlenoise 
# model1<-ref+rnorm(30)/2 
# #addmorenoise 
# model2<-ref+rnorm(30) 
# #displaythediagramwiththebettermodel 
# oldpar<-taylor.diagram(ref,model1) 
# #nowaddtheworsemodel 
# taylor.diagram(ens,rea,add=TRUE,col="blue") 
# #getapproximatelegendposition 
# # lpos<-1.5*sd(ref) 
# # #addalegend 
# # legend(lpos,lpos,legend=c("Better","Worse"),pch=19,col=c("red","blue")) 
# # #nowrestoreparvalues 
# # par(oldpar) 
# # #showthe"allcorrelation"display 
# # taylor.diagram(ref,model1,pos.cor=FALSE) 
# # taylor.diagram(ref,model2,add=TRUE,col="blue")
# shp_triveneto = st_read("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")  %>% 
#                         filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))

# M="q"

# ens_sta <- fread(glue("ens_sta_{M}.csv"))
# rea_sta <- fread(glue("rea_sta_{M}.csv"))
# sta <- fread(glue("sta_{M}.csv"))

# tm <- list()
# tmap_mode("plot")

# ens_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_ens

# rea_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_rea

# sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_sta

# ####PLOT THE SINGLE METRICS ----
# tm_shape(sf_ens)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Ensemble",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm1



# tm_shape(sf_rea)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="SPHERA",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm2


# tm_shape(sf_sta)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Stations",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure"
#           )  -> tm3

# tmap_save(tmap_arrange(list(tm1,tm2,tm3),3,1), "my_map.png", width = 24, height = 15, units="cm", dpi = 300)


# ####PLOT THE SPATIAL VARIABILITY OF THE SINGLE METRICS ----

# sf_ens %>% 
#         mutate(sv_rea=f/sf_rea[,M],
#                sv_sta=f/sf_sta[,M]) -> sf_sv 
# tm_shape(sf_sv)+
#         tm_dots(col="sv_rea",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# tm_shape(sf_sv)+
#         tm_dots(col="sv_sta",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# sd(sf_sv$sv_rea,na.rm=TRUE) / mean(sf_sv$sv_rea, na.rm=TRUE)
# sd(sf_sv$sv_sta,na.rm=TRUE) / mean(sf_sv$sv_sta, na.rm=TRUE)




# ens <- read.csv("/home/lcesarini/2022_resilience/ensemble.csv")
# rea <- read.csv("/home/lcesarini/2022_resilience/reanalysis.csv")
# library(plotrix)
# ref<-rnorm(30,sd=2) 
# #addalittlenoise 
# model1<-ref+rnorm(30)/2 
# #addmorenoise 
# model2<-ref+rnorm(30) 
# #displaythediagramwiththebettermodel 
# oldpar<-taylor.diagram(ref,model1) 
# #nowaddtheworsemodel 
# taylor.diagram(ens,rea,add=TRUE,col="blue") 
# #getapproximatelegendposition 
# # lpos<-1.5*sd(ref) 
# # #addalegend 
# # legend(lpos,lpos,legend=c("Better","Worse"),pch=19,col=c("red","blue")) 
# # #nowrestoreparvalues 
# # par(oldpar) 
# # #showthe"allcorrelation"display 
# # taylor.diagram(ref,model1,pos.cor=FALSE) 
# # taylor.diagram(ref,model2,add=TRUE,col="blue")
# shp_triveneto = st_read("data/gadm36_ITA.gpkg", layer="gadm36_ITA_1")  %>% 
#                         filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))

# M="q"

# ens_sta <- fread(glue("ens_sta_{M}.csv"))
# rea_sta <- fread(glue("rea_sta_{M}.csv"))
# sta <- fread(glue("sta_{M}.csv"))

# tm <- list()
# tmap_mode("plot")

# ens_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_ens

# rea_sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_rea

# sta %>% 
# st_as_sf(coords=c("lon","lat"),crs=4326) -> sf_sta

# ####PLOT THE SINGLE METRICS ----
# tm_shape(sf_ens)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Ensemble",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm1



# tm_shape(sf_rea)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="SPHERA",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure")  -> tm2


# tm_shape(sf_sta)+
# tm_dots(col=ifelse(M=="f","f",ifelse(M=="i","i","q")),
#         size=1,
#         palette="plasma",
#         style="fixed",
#         breaks=seq(2.4,18.5,2)/100,
#         labels=paste0(seq(2.4,18.5,2),"%"),
#         showNA=0,colorNA=NULL
#         )+
# tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
# tm_layout(panel.labels="Stations",
#           panel.show =TRUE,
#           title=ifelse(M=="f","Freq.",ifelse(M=="i",Int.,"Heavy Prec.")),
#           title.bg.color="azure"
#           )  -> tm3

# tmap_save(tmap_arrange(list(tm1,tm2,tm3),3,1), "my_map.png", width = 24, height = 15, units="cm", dpi = 300)


# ####PLOT THE SPATIAL VARIABILITY OF THE SINGLE METRICS ----

# sf_ens %>% 
#         mutate(sv_rea=f/sf_rea[,M],
#                sv_sta=f/sf_sta[,M]) -> sf_sv 
# tm_shape(sf_sv)+
#         tm_dots(col="sv_rea",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# tm_shape(sf_sv)+
#         tm_dots(col="sv_sta",
#                 size=1,
#                 palette="RdBu",
#                 showNA=0,
#                 colorNA=NULL)+
#         tm_shape(shp_triveneto)+
#         tm_borders(col="red",lwd=2)

# sd(sf_sv$sv_rea,na.rm=TRUE) / mean(sf_sv$sv_rea, na.rm=TRUE)
# sd(sf_sv$sv_sta,na.rm=TRUE) / mean(sf_sv$sv_sta, na.rm=TRUE)



# ens <- read.csv("/home/lcesarini/2022_resilience/ensemble.csv")
# rea <- read.csv("/home/lcesarini/2022_resilience/reanalysis.csv")
# library(plotrix)
# ref<-rnorm(30,sd=2) 
# #addalittlenoise 
# model1<-ref+rnorm(30)/2 
# #addmorenoise 
# model2<-ref+rnorm(30) 
# #displaythediagramwiththebettermodel 
# oldpar<-taylor.diagram(ref,model1) 
# #nowaddtheworsemodel 
# taylor.diagram(ens,rea,add=TRUE,col="blue") 
# #getapproximatelegendposition 
# # lpos<-1.5*sd(ref) 
# # #addalegend 
# # legend(lpos,lpos,legend=c("Better","Worse"),pch=19,col=c("red","blue")) 
# # #nowrestoreparvalues 
# # par(oldpar) 
# # #showthe"allcorrelation"display 
# # taylor.diagram(ref,model1,pos.cor=FALSE) 
# # taylor.diagram(ref,model2,add=TRUE,col="blue")

# data(World)
# data(metro)

# # legend bubble size (10, 20, 30, 40 million) are
# # - are normlized by upper limit (40e6),
# # - square rooted (see argument perceptual of tm_symbols), and 
# # - scaled by 2:
# bubble_sizes <- ((c(10, 20, 30, 40) * 1e6) / 40e6) ^ 0.5 * 2 

# tm_shape(World) + 
# 	tm_polygons() + 
# tm_shape(metro) +
# 	tm_symbols(col='pop2020', 
# 		breaks = c(0, 15, 25, 35, 40) * 1e6,
# 		n=4,
# 		palette = 'YlOrRd',
# 		size='pop2020',
# 		sizes.legend = c(10, 20, 30, 40) * 1e6,
# 		size.lim = c(0, 40e6),
# 		scale = 2,
# 		legend.size.show = FALSE,    # comment this line to see the original size legend
# 		legend.col.show = FALSE,     # comment this line to see the original color legend
# 		legend.size.is.portrait = TRUE) + 
# tm_add_legend('fill', 
# 	col = RColorBrewer::brewer.pal(4, "YlOrRd"),
# 	border.col = "grey40",
# 	size = bubble_sizes,
# 	labels = c('0-15 mln','15-25 mln','25-35 mln','35-40 mln'),
# 	title="Population Estimate") -> tm1

# tmap_save(tm1, "my_map2.png", width = 21, height = 16, units="cm", dpi = 300)


options(warn = defaultW)
