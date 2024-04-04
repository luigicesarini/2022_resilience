library(tmap)
library(dplyr)
library(sf)
library(data.table)
library(raster, help, pos = 2, lib.loc = NULL)        
library(ggplot2, help, pos = 2, lib.loc = NULL)

setwd("/home/lcesarini/2022_resilience/")


df <- fread("/home/lcesarini/2022_resilience/df_check_dist.csv")



colnames(df) <- c("name","dist_lon","dist_lat","err_medio","err_massimo","err_999","err_max_ecdf")



df %>% 
ggplot(aes(x=name))+
geom_point(aes(y=dist_lon))+
theme_bw()


apply(df[,2:7],2,function(g){
    
    g %>% 
        abs() %>% 
        mean() -> media

    print(media)
})




