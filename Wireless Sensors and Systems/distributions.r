setwd("C:/C_files/patsy chan/QUB_doc/ELE8096/Assignment 2")

# Load all data
belfastdata<-read.csv("8096data.csv", header = TRUE, sep = ",")
pollutants<-subset(belfastdata, select=-c(temperature,humidity))

# Histogram for all pollutants
par(mfrow=c(2,3))
p1<-hist(pollutants$PM2.5, breaks = 10)
p2<-hist(pollutants$NO2, breaks = 10)
p3<-hist(pollutants$O3, breaks = 10)
plot(p1, col=rgb(0,0,1,1/4),xlim=c(0,140))
plot(p2, col=rgb(1,0,0,1/4), add=T) 
plot(p3, col=rgb(0,1,0,1/4), add=T) 

# Density for all pollutants
par(mfrow=c(2,3))
d1<-density(pollutants$PM2.5, na.rm = TRUE)
d2<-density(pollutants$NO2, na.rm = TRUE)
d3<-density(pollutants$O3, na.rm = TRUE)
plot(d1, col=rgb(0,0,1,1/4))
plot(d2, col=rgb(1,0,0,1/4)) 
plot(d3, col=rgb(0,1,0,1/4)) 
par(mfrow=c(1,1))
plot(d1, col=rgb(0,0,1,1/4),xlim=c(0,140))
lines(d2, col=rgb(1,0,0,1/4), add=T) 
lines(d3, col=rgb(0,1,0,1/4), add=T) 

# Empirical cumulative distribution function (ECDF) of pollutants
par(mfrow=c(2,3))
e1<-ecdf(pollutants$PM2.5)
e2<-ecdf(pollutants$NO2)
e3<-ecdf(pollutants$O3)
plot(e1, col=rgb(0,0,1,1/4))
plot(e2, col=rgb(1,0,0,1/4)) 
plot(e3, col=rgb(0,1,0,1/4)) 
par(mfrow=c(1,1))
plot(e1, col=rgb(0,0,1,1/4),xlim=c(0,140))
plot(e2, col=rgb(1,0,0,1/4), add=T) 
plot(e3, col=rgb(0,1,0,1/4), add=T) 

# Empirical cumulative distribution function (ECDF) of weather condition
par(mfrow=c(1,2))
e4<-ecdf(belfastdata$temperature)
e5<-ecdf(belfastdata$humidity)
plot(e4, col=rgb(0,0,1,1/4))
plot(e5, col=rgb(1,0,0,1/4)) 
par(mfrow=c(1,1))
plot(e4, col=rgb(0,0,1,1/4),xlim=c(0,80))
plot(e5, col=rgb(1,0,0,1/4), add=T) 