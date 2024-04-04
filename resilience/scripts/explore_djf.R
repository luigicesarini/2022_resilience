.libPaths(c("/mnt/ssd/lcesarini/pkgR", .libPaths()))
install.packages("sf")
remove.packages(c("sf","tmaptools","star"))
install.packages("fastmap",lib="/mnt/ssd/lcesarini/pkgR")
install.packages("htmltools",lib="/mnt/ssd/lcesarini/pkgR")
install.packages("tmap",lib="/mnt/ssd/lcesarini/pkgR")
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


#DO IT FOR ONE MODEL
sta <- read.csv(glue("/home/lcesarini/2022_resilience/csv/STATIONS_q_DJF_1000_SEQUENTIAL_VALIDATION.csv"))
list_boxplot   <- list()
list_histogram <- list()
list_tmap      <- list()
list_dfs       <- list()


for (MODEL in c("CMCC","CNRM","MOHC","KNMI","KIT","ETH","HCLIMcom","ICTP")) {
    
    bia <- read.csv(glue("/home/lcesarini/2022_resilience/csv/{MODEL}_q_{SEAS}_biased_{SPLIT}.csv"))
    eqm <- read.csv(glue("/home/lcesarini/2022_resilience/csv/{MODEL}_q_{SEAS}_EQM_{SPLIT}.csv"))
    qdm <- read.csv(glue("/home/lcesarini/2022_resilience/csv/{MODEL}_q_{SEAS}_QDM_{SPLIT}.csv"))


    oro <- raster(list.files(glue("/mnt/data/RESTRICTED/CARIPARO/DATA_FPS/ECMWF-ERAINT/{MODEL}/"),"orog",full.names=TRUE))

    elev <- extract(oro,bia[,c("lon","lat")])

    bia$ele <- elev
    eqm$ele <- elev
    qdm$ele <- elev


    bia %>% 
        mutate(
            bias_ori=(q-sta$q) / sta$q * 100,
            bias_eqm=(eqm$q-sta$q) / sta$q * 100,
            bias_qdm=(qdm$q-sta$q) / sta$q * 100
        ) %>% 
        select(starts_with("bias"),ele,lon,lat) -> df_biases

    df_biases %>% 
        st_as_sf(coords=c("lon","lat")) %>% 
        dplyr::filter(!is.na(bias_ori)) %>% 
        tm_shape()+tm_symbols(col='bias_ori',size='ele',colorNA=NULL) 


    df_biases %>%
        dplyr::filter(!is.na(bias_ori)) %>% 
        dplyr::select(starts_with("bias")) %>% 
        reshape2::melt() %>% 
        ggplot(aes(fill=variable,y=value))+
        geom_boxplot()+
        geom_hline(yintercept=0,color='red',lwd=2)+
        labs(title=glue("{MODEL} {SEAS}"))+
        coord_cartesian(ylim=c(-50,50))+
        theme_minimal() -> list_boxplot[[MODEL]]

    df_biases %>%
        dplyr::filter(!is.na(bias_ori)) %>% 
        mutate(increase=case_when(bias_ori < bias_eqm ~ TRUE,
                                TRUE~FALSE)) -> list_dfs[[MODEL]]
    list_dfs[[MODEL]] %>% 
        dplyr::select(ele,increase) %>% 
        ggplot(aes(x=(ele),after_stat(density),fill=increase))+
        geom_histogram(bins=50)+
        labs(title=glue("{MODEL} {SEAS}"))+
        theme_minimal() -> list_histogram[[MODEL]]

    df_biases %>%
        dplyr::filter(!is.na(bias_ori)) %>% 
        mutate(increase=case_when(bias_ori < bias_eqm ~ TRUE,
                                TRUE~FALSE)) %>% 
        st_as_sf(coords=c("lon","lat")) %>% 
        tm_shape()+tm_symbols(col="increase")+tm_layout(title=glue("{MODEL}")) -> list_tmap[[MODEL]]


}


plyr::joinall()
do.call("rbind",list_dfs) %>% dim()



counter <- 0
ll <- lapply(list_dfs, function(g){
    counter <<- counter+1

    g %>% 
    dplyr::select(lon,lat,increase) %>% 
    rename(glue("{names(list_dfs)[counter]}"="increase"))   

})


x <- plyr::join_all(ll,by=c("lon","lat"))
colnames(x)[3:10] <- paste0("increase_",names(list_dfs))

list_dfs[[1]][,c("lon","lat","ele")]


x %>% 
    reshape2::melt(id.vars=c("lon","lat")) %>% 
    mutate(value=as.integer(value)) %>% 
    group_by(lon,lat) %>% 
    summarise(n=sum(value)) %>%
    left_join(.,list_dfs[[1]][,c("lon","lat","ele")],by=c("lon","lat"))  %>% 
    st_as_sf(coords=c("lon","lat")) %>% 
    tm_shape()+
        tm_symbols(col="n",size="ele",scale=2.75,palette='RdBu')



tmap_arrange(list_tmap,nrow=2,ncol=4)

library(patchwork)


(list_boxplot[[1]] | list_boxplot[[2]] | list_boxplot[[3]] |  list_boxplot[[4]]) /
(list_boxplot[[5]] | list_boxplot[[6]] | list_boxplot[[7]] |  list_boxplot[[8]])

(list_histogram[[1]] | list_histogram[[2]] | list_histogram[[3]] |  list_histogram[[4]]) /
(list_histogram[[5]] | list_histogram[[6]] | list_histogram[[7]] |  list_histogram[[8]])


