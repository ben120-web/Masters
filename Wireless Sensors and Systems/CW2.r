belfastdata <- read.csv("8096data.csv", header = TRUE, sep = ",")
boxdata <- read.csv("box.csv", header = TRUE, sep = ",")


modelPM2.5 <- lm(PM2.5~humidity + NO2 + O3 + temperature, data = belfastdata)
modelNO2 <- lm(NO2~PM2.5 + humidity + O3 + temperature, data = belfastdata)
modelO3 <- lm(O3~PM2.5 + NO2 + humidity + temperature, data = belfastdata)

# Oveall view method
# --------------------------------------------------------------------------------
library(car)
states <- as.data.frame(belfastdata[,c('PM2.5','NO2','O3','temperature','humidity')])
cor(states)#查看变量相关系数
pdf("D:/Rstudio4.1.2/workspace/overall.pdf",width=4,height=4)
scatterplotMatrix(states,spread = FALSE)  
dev.off()

# # adjr2 method
# # --------------------------------------------------------------------------------
library(leaps)

leaps <- regsubsets(PM2.5~humidity + NO2 + O3 + temperature, data = belfastdata,nbest = 4)
pdf("D:/Rstudio4.1.2/workspace/adjr2 PM2.5.pdf",width=8,height=8)
plot(leaps,scale = 'adjr2') 
dev.off()

leaps <- regsubsets(NO2~humidity + PM2.5 + O3 + temperature, data = belfastdata,nbest = 4)
pdf("D:/Rstudio4.1.2/workspace/adjr2 NO2.pdf",width=8,height=8)
plot(leaps,scale = 'adjr2') 
dev.off()

leaps <- regsubsets(O3~humidity + PM2.5 + NO2 + temperature, data = belfastdata,nbest = 4)
pdf("D:/Rstudio4.1.2/workspace/adjr2 O3.pdf",width=8,height=8)
plot(leaps,scale = 'adjr2') 
dev.off()

# # qqplot method
# # --------------------------------------------------------------------------------
pdf("D:/Rstudio4.1.2/workspace/qqplotPM2.5.pdf",width=8,height=8)
qqPlot(modelPM2.5,labels = row.names(states),id.method = 'identify',simulate = T)
dev.off()

pdf("D:/Rstudio4.1.2/workspace/qqplotNO2.pdf",width=8,height=8)
qqPlot(modelNO2,labels = row.names(states),id.method = 'identify',simulate = T)
dev.off()

pdf("D:/Rstudio4.1.2/workspace/qqplotO3.pdf",width=8,height=8)
qqPlot(modelO3,labels = row.names(states),id.method = 'identify',simulate = T)
dev.off()

# box plots
# --------------------------------------------------------------------------------
library(tidyverse)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(ggpubr)

box <- melt(belfastdata)
ggplot(box,aes(x=variable,y=value,color=variable))+
geom_boxplot(aes(fill=factor(variable)))
theme(axis.text.x=element_text(angle=50,hjust=0.5, vjust=0.5)) +
theme(legend.position="none")
ggsave(filename="D:/Rstudio4.1.2/workspace/overallbox.pdf",width=12,height=9)


# preboxPM2.5 <- melt(boxdata)
# boxPM2.5 <- preboxPM2.5[,'PM2.5']
# ggplot(boxPM2.5,aes(x=variable,y=value))+
# geom_boxplot()
# ggsave(filename="D:/Rstudio4.1.2/workspace/boxPM2.5.pdf",width=12,height=9)