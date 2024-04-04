library(tmap)
library(dplyr)
library(sf)
library(glue)
library(data.table)
library(raster, help, pos = 2, lib.loc = NULL)        
library(ggplot2, help, pos = 2, lib.loc = NULL)

setwd("/home/lcesarini/2022_resilience/")

x <- read.csv("csv/bias_boot.csv")
x %>%
    st_as_sf(coords=c("lon","lat"),crs=4326) -> sf

tmap_mode("plot")   

tm_shape(sf)+tm_dots(col="SS")

name <- "VE_0091"
STATIONS <- c("VE_0088","VE_0100","TN_0147","AA_4740","VE_0239","VE_0011")

for(name in STATIONS){
    df_dt = data.table::fread(glue("/home/lcesarini/2022_resilience/data/check_nathalia/dates_{name}_station.csv"),sep=';')
    df_pr = data.table::fread(glue("/home/lcesarini/2022_resilience/data/check_nathalia/pr_{name}_station.csv"),sep=';')
    
    cbind(df_dt,df_pr) %>% 
        write.table(glue("/home/lcesarini/2022_resilience/data/check_nathalia/{name}_station.csv"),quote=FALSE, row.names=FALSE,sep=',')
}
df_dt = data.table::fread(glue("/home/lcesarini/2022_resilience/data/dates_{name}.csv"),sep=';')
df_pr = data.table::fread(glue("/home/lcesarini/2022_resilience/data/pr_{name}.csv"),sep=';')
df_dt = data.table::fread(glue("/home/lcesarini/2022_resilience/data/dates_VE_0011_station.csv"),sep=';')
df_pr = data.table::fread(glue("/home/lcesarini/2022_resilience/data/pr_VE_0011_station.csv"),sep=';')

cbind(df_dt,df_pr) %>% 
    write.table(glue("/home/lcesarini/dates_VE_0011_station.csv"),quote=FALSE, row.names=FALSE,sep=',')

pr_st <- read.csv(glue("/home/lcesarini/2022_resilience/stations/text/prec_{name}.csv"))
dt_st <- read.csv(glue("/home/lcesarini/2022_resilience/dates/{name}.csv"),header=FALSE)


st_remap <- cbind(df_dt,df_pr) 
st_origi <- cbind(dt_st$V1[stringr::str_detect(dt_st$V1,"2000|2001|2002|2003|2004|2005|2006|2007|2008|2009")],
                  pr_st[stringr::str_detect(dt_st$V1,"2000|2001|2002|2003|2004|2005|2006|2007|2008|2009"),]
) 

cbind(st_remap,as.numeric(st_origi[,2])) %>% 
rename("pr_model"="pr",
        "pr_obs"="V2") %>% 
reshape2::melt(id.vars="date") -> df_tall 

ggplot(df_tall %>% filter(value > 5),aes(x=value,color=variable))+
    stat_ecdf(geom='point')+
    theme_bw()


ggplot(df_tall,aes(x=as.Date(date),y=value,color=variable))+
    geom_point()+
    geom_line()+
    theme_bw()

ggplot(df_tall %>% filter(value > 5),aes(x=value,
                   fill=variable
                   ))+
    geom_density(aes(color=variable),fill=NA, size=3)+
    geom_histogram(aes(y=..density..),
                   binwidth=1,position='identity',
                   alpha=0.25,color="grey")+
    theme_bw()




as.POSIXct(dt_st[1,1],format="%d-%b-%Y %H:%M:%S")
as.POSIXct("01-Jan-1993 03:00:00", format="%d-%b-%Y %H:%M:%S")


df = data.table::fread("meta_station_updated_col.csv")


df %>% 
filter(name=="VE_0091")

df %>%
    st_as_sf(coords=c("lon","lat"),crs=4326) -> sf

print(head(sf))


plot(sf$geometry)
tmap_mode("view")
tm_shape(sf)+tm_dots(col="elv", 
                     midpoint = NA,
                     size = rev("elv"))


df_heatmap <- data.table::fread("output/array_heatmap.txt")


ggplot()
lattice::levelplot(df_heatmap)
lattice::levelplot(df_heatmap)

heatmap(as.matrix(df_heatmap),
        Rowv = NA, Colv = NA,
        labRow=c("SON","DJF","MAM","JJA"),
        labCol=c("Freq.","Int.","Heavy Prec.")
        )

tm_shape(raster(as.matrix(df_heatmap)))+
    tm_raster()+
    tm_legend(legend.position='outside')

df_heatmap_2 <- df_heatmap
df_heatmap_2$seas <- c("SON","DJF","MAM","JJA")
colnames(df_heatmap_2) <- c("Freq.","Int.","Heavy Prec.")

# cmap = (mpl.colors.ListedColormap(['#7E1104',
#                                    '#E33434', 
#                                    '#F58080', 
#                                    '#F8BCBC', 
#                                    '#FBE2E2', 
#                                    'white',
#                                    '#D4F7FA',
#                                    '#90DEF8',
#                                    '#7BB2ED',
#                                    '#262BBD',
#                                    '#040880'
#                                     ]))
    
df_heatmap_2 %>% 
    reshape2::melt(id.vars='seas') %>% 
    mutate(color=case_when(value < -80 ~ "#7E1104",
                           value < -60 & value >= -80 ~ "cat_1",#E33434", 
                           value < -40 & value >= -60 ~ "cat_2",#F58080", 
                           value < -25 & value >= -40 ~ "cat_3",#F8BCBC", 
                           value < -5 & value >= -25  ~ "cat_4",#FBE2E2", 
                           value < 25 & value >= -5   ~ "cat_5",#white", 
                           value < 40 & value >= 25   ~ "cat_6",#D4F7FA", 
                           value < 60 & value >= 40   ~ "cat_7",#90DEF8", 
                           value < 80 & value >= 60   ~ "cat_8",#7BB2ED", 
                           value < 100 & value >= 80  ~ "cat_9",#262BBD", 
                           value >= 100               ~ "cat_10",#040880"
                           )
    ) -> df_reshaped
    # ggplot(aes(y=seas,x=variable,fill=color))+
    # geom_tile(aes(color="black"))+
    # guides(fill=NA)

counter <- 1
for(lr in 0:2){

    for(tl in 3:0){
        #glue::glue("idx:{counter},xmin:{col+0.5},xmax:{col+1.5},ymin:{row+1.5},ymax:{row+0.5}")  %>%  print()
        df_reshaped[counter,c("xmin","xmax","ymin","ymax")] <- c(lr+0.5,lr+1.5,tl+0.5,tl+1.5)
        counter <- counter +1
    }
}

df_reshaped %>% str
unique(df_reshaped$xmin)
ggplot(df_reshaped, aes(variable, seas)) + 
        geom_rect(aes(fill = as.integer(value),
                      xmin=xmin,xmax=xmax,
                      ymin=ymin,ymax=ymax),color='white')+
        scale_fill_continuous(
            breaks=c(-80,-60,-40,-25,-5,
                     25,40,60,80,100),
            labels=c(-80,-60,-40,-25,-5,
                     25,40,60,80,100),
                        #  values=c("#E33434","#F58080","#F8BCBC",
                        #         "#FBE2E2","#white","#D4F7FA",
                        #         "#90DEF8","#7BB2ED","#262BBD","#040880"),
                                guide = guide_legend(title="[%]"))+
        scale_x_discrete("")+
        scale_y_discrete("")+
        theme_minimal()+
        theme(axis.text=element_text(size=30),
              legend.text=element_text(size=25),
              legend.title=element_text(size=35))


p1






df <- fread("/mnt/data/lcesarini/tmp/data.txt")

df %>% 
    mutate(prec=as.numeric(V3)) %>% 
    pull(prec) %>% 
    range(na.rm=TRUE)
