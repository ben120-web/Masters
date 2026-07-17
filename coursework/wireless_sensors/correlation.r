mydata = read.csv("C:/Users/Noll Chen/OneDrive - Queen's University Belfast/Desktop/airdata2.csv", header = TRUE, stringsAsFactors = F)
str(mydata)
dfclean<-mydata[2:6]
mydata.cor = cor(dfclean)
install.packages("corrplot")
library(corrplot)
corrplot(mydata.cor, method="circle")
corrplot(mydata.cor, method="number")