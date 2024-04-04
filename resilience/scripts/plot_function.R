library(tmap)
library(dplyr)
library(sf)
library(glue)
library(data.table)
library(raster, help, pos = 2, lib.loc = NULL)        
library(ggplot2, help, pos = 2, lib.loc = NULL)

setwd("/home/lcesarini/2022_resilience/csv")

defaultW <- getOption("warn") 

select <- dplyr::select
filter <- dplyr::select

options(warn = -1) 
st_read("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1",quiet=TRUE)  %>% colnames()
shp_triveneto = st_read("/home/lcesarini/2022_resilience/data/gadm36_ITA.gpkg", layer="gadm36_ITA_1",quiet=TRUE)  %>% 
                        dplyr::filter(NAME_1 %in% c("Veneto","Trentino-Alto Adige"))




make_maps <- function(sf,size_marker,M,mdl,title_panel,bbox,brks,lbls) {
        
        mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
                                '#E85413','#E82313')
        # if (M=='f' &) {
        #     mycols <- c('#B1DFFA','#36BCFF', '#508D5E','#55CB70','#E5E813','#E8AB13','#E85413','#E82313')
        # }else if(M=='i'){
        #     mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
        #                 '#E85413','#E82313')
        # }else if(M=='pr'){
        #     mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
        #                 '#E85413','#E82313')
        # }else{
        #     mycols <- c('#ECF7FE','#B1DFFA','#36BCFF', '#508D5E','#55CB70','#88F7A1','#E5E813','#E8AB13',
        #                 '#E85413','#E82313')
        # }
        
        if (missing(brks) | missing(lbls)) {
                if(M=='mean_prec'){
                        brks <- seq(0.1,0.275,0.025)
                        lbls <- paste0(seq(0.1,0.275,0.05),"")
                }else if (M=="q"){
                        brks <- seq(2,20,2)
                        lbls <- paste0(seq(2,20,2),"")
                }else if (M=="i") {
                        brks <- seq(0.3,3.3,0.3)
                        lbls <- paste0(seq(0.3,3.0,0.3),"")
                }
        }


        tm_shape(sf,bbox=bbox)+
        tm_dots(col=ifelse(M=='mean_prec','mean',M),
                size=size_marker,
                palette=mycols,
                style="fixed",
                breaks=brks,
                labels=lbls,
                showNA=0,colorNA=NULL,legend.show=FALSE
                )+
        tm_shape(shp_triveneto)+tm_borders(col="red",lwd=2)+
        tm_add_legend('fill', 
	        col = mycols,
	        border.col = "black",
                is.portrait=TRUE,
	        # size = bubble_sizes,
	        labels = lbls,
	        title="[mm/hr]")+      
        tm_layout(panel.labels=title_panel,
                panel.show =TRUE,
                # title="Mean Prec.",
                title.bg.color="azure") -> plt

        return(plt)
}
# brks <- seq(0,0.15,length.out=10)
# lbls <- round(brks,2) %>% as.character()
# mdl <- "SPHERA"
# bc=""
# M='i'

boxplot_to_file <- function(path_sta,path_sph,m='q'){

        fread(path_sta) %>% 
                left_join(.,fread(path_sph),by=c("lon","lat"))  %>% 
                dplyr::select(tidyselect::contains(m)) %>% 
                melt() %>% 
                ggplot(aes(x=variable,fill=variable,y=value))+
                geom_boxplot(notch=TRUE,outlier.shape = NA,width=0.5)+
                # geom_jitter(width=0.2)+
                scale_fill_manual(values=c("red","blue"),labels=c("STATIONS","SPHERA"))+
                scale_x_discrete(labels=c("",""))+
                theme_bw()+
                theme(title       = element_text(size=26),
                axis.title  = element_text(size=25),
                axis.text   = element_text(size=22),
                legend.text = element_text(size=23),
                ) -> plot_plot

        return(plot_plot)
}


# ##### MEAN INTENSITY OF WET HOURS
# make_maps(sf=fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='i',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("Ensemble"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[1]]

# make_maps(sf=fread(glue("VHR_i_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='i',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("VHR-CMCC"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[2]]

# make_maps(sf=fread(glue("SPHERA_i_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='i',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("SPHERA"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[3]]

# make_maps(sf=fread(glue("Stations_i_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='i',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("Stations"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[4]]

# make_maps(sf=fread(glue("gripho_i_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='i',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("GRIPHO"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[5]]


# ##### HEAVY PRECIPITATION

# # make_maps(sf=fread(glue("Ensemble_q_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
# #                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
# #                         title_panel=glue("Ensemble"),
# #                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[6]]

# # make_maps(sf=fread(glue("VHR_q_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
# #                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
# #                         title_panel=glue("VHR-CMCC"),
# #                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[7]]

# # make_maps(sf=fread(glue("SPHERA_q_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
# #                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
# #                         title_panel=glue("SPHERA"),
# #                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[8]]

# # make_maps(sf=fread(glue("Stations_q_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
# #                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
# #                         title_panel=glue("Stations"),
# #                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[9]]

# # make_maps(sf=fread(glue("gripho_q_JJA.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
# #                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
# #                         title_panel=glue("GRIPHO"),
# #                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[10]]

# make_maps(sf=fread(glue("Ensemble_q_JJA_valid.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("Ensemble"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[6]]

# make_maps(sf=fread(glue("VHR_q_JJA_valid.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("VHR-CMCC"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[7]]

# make_maps(sf=fread(glue("SPHERA_q_JJA_valid.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("SPHERA"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[8]]

# make_maps(sf=fread(glue("Stations_q_JJA_valid.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("Stations"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[9]]

# make_maps(sf=fread(glue("gripho_q_JJA_valid.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326),
#                         size_marker=0.5,M='q',mycols=mycols,brks=brks,lbls=lbls,mdl=mdl,
#                         title_panel=glue("GRIPHO"),
#                         bbox=st_bbox(fread(glue("Ensemble_i_JJA_2.csv")) %>% st_as_sf(coords=c("lon","lat"),crs=4326))) -> list_sf[[10]]

# tmap_save(tmap_arrange(list_sf,nrow=2,ncol=5),
#           glue("my_map_i_JJA2.png"), 
#           width = 53, height = 17, units="cm", dpi = 450)







options(warn = defaultW)

