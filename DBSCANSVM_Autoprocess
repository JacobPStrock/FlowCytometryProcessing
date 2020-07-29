#These are the core functions for clustering/classification of FC data
#This is a subset of a complete algorithm for processing many samples
#The full algorithm incorporates important subsampling/upsampling methods that will make classification robust/consistent
#across different samples. This file is intended to roughly visualize how the clustering will work with a given dataset
#Important places you may tinker with: minimum cluster count (line 38) and preprocessing (line 118)

#Last edited by JPS 07/15/2020


library(ggplot2)
library(gridExtra)
library(dplyr)
library(dbscan)
library(fpc)
library(stringr)
library(e1071)

################################################################################################################

#Rotation matrix Function
Rmat<- function(theta){
  Rad<- theta
  M<- matrix(c(cos(Rad),-sin(Rad),sin(Rad),cos(Rad)),byrow=TRUE,ncol = 2)
  return(M)
}

#Data Process and DBSCAN Function
DBSCAN.SVM.process <- function(D,NC){
  #select epsilon parameter
  Dist<- sort(kNNdist(D,k=4))
  Distmat<- cbind(seq(1:length(Dist)),Dist)
  Theta<- atan2((max(Distmat[,2])-min(Distmat[,2])),(max(Distmat[,1])-min(Distmat[,1]))) #Find optimal angle to rotate
  RM<- Rmat(-Theta)
  RDat<- Distmat%*%t(RM)
  Epsilon<- Dist[which(RDat[,2]==min(RDat[,2]))]
  
  #select minPts parameter
  D.S=D[sample(seq(1,nrow(D),1),5000),]
  Tester=function(MP){
    m1=dbscan::dbscan(D.S,eps=Epsilon,minPts=MP)
    return(length(unique(m1$cluster)))
  }
  Grid=seq(10,150,2)
  Out=lapply(Grid,function(x) Tester(x))
  MP=Grid[min(which(Out==NC))]

  #run dbscan using selected epsilon
  dboutput <- fpc::dbscan(D,eps=Epsilon, MinPts = MP)
  D$Cluster<- dboutput$cluster
  
  #run svm on uncertain data
  Classified <- D[D$Cluster!=0,]
  Unclassified<-D[D$Cluster==0,]
  
  svm1 <- svm(data=Classified,as.factor(Cluster)~RED.B.HLog+FSC.HLog+SSC.HLog+GRN.B.HLog+YEL.B.HLog, kernel='polynomial',class.weights='inverse')
  
  #return dbscan object and svm object
  return(list(dbop=dboutput, svmop=svm1 ))
}

#Plotting Function
DBSCAN.plot <- function(data){
  
  #Summary Attributes per Cluster
  datsum       = data%>%group_by(Cluster)%>%
    add_tally()%>%
    summarise_all(.,funs(mean))
  
  #makefolder for output
  #ii= substr(File,1,nchar(File)-4)
  #mkdir(paste0("Figures",ii))
  #setwd(paste0('C:/Users/Jacob Strock/Documents/Menden-Deuer Lab/Misc/LightCommunityTreatment/TestProcessing_07082020/DataFiles/',paste0("Figures",ii)))
  
  #Plot
  par(mfrow=c(1,1))
  
  #pdf(paste('FSC.RED_',ii,'.pdf',sep=''),height=4, width=6)
  plot1<-ggplot(data = data, aes(x=FSC.HLog,y=RED.B.HLog,color=as.factor(Cluster)))+geom_point(alpha=0.4)+
    scale_color_manual(values=rainbow(length(unique(data$Cluster))))+
    theme_minimal()+
    geom_label(data=datsum, aes(x=FSC.HLog,y=RED.B.HLog,label=paste("Clust.",datsum$Cluster,sep=" ")))+
    labs(color='Cluster')
  #print(plot1)
  #dev.off()
  
  #pdf(paste0('FSC.YEL_',ii,'.pdf'),height=4, width=6)
  plot2<- ggplot(data = data, aes(x=FSC.HLog,y=YEL.B.HLog,color=as.factor(Cluster)))+geom_point(alpha=0.4)+
    scale_color_manual(values=rainbow(length(unique(data$Cluster))))+
    theme_minimal()+
    geom_label(data=datsum, aes(x=FSC.HLog,y=YEL.B.HLog,label=paste("Clust.",datsum$Cluster,sep=" ")))+
    labs(color='Cluster')
  #print(plot2)
  #dev.off()
  
  #pdf(paste0('FSC.SSC_',ii,'.pdf'),height=4, width=6)
  plot3<- ggplot(data = data, aes(x=FSC.HLog,y=SSC.HLog,color=as.factor(Cluster)))+geom_point(alpha=0.4)+
    scale_color_manual(values=rainbow(length(unique(data$Cluster))))+
    theme_minimal()+
    geom_label(data=datsum, aes(x=FSC.HLog,y=SSC.HLog,label=paste("Clust.",datsum$Cluster,sep=" ")))+
    labs(color='Cluster')
  #print(plot3)
  #dev.off()
  
  #pdf(paste0('FSC.GRN_',ii,'.pdf'),height=4, width=6)
  plot4<- ggplot(data = data, aes(x=FSC.HLog,y=GRN.B.HLog,color=as.factor(Cluster)))+geom_point(alpha=0.4)+
    scale_color_manual(values=rainbow(length(unique(data$Cluster))))+
    theme_minimal()+
    geom_label(data=datsum, aes(x=FSC.HLog,y=GRN.B.HLog,label=paste("Clust.",datsum$Cluster,sep=" ")))+
    labs(color='Cluster')
  #print(plot4)
  #dev.off()
  
  #return classified data
  print(grid.arrange(plot1,plot2,plot3,plot4,nrow=2))
  return(datsum)
  
}

#####################################################################################################################
###############
#Load
setwd("C:/Users/Jacob Strock/Documents/Menden-Deuer Lab/Misc/LightCommunityTreatment/TestProcessing_07082020/DataFiles")
df=read.csv('2020-07-04_at_12-18-32pm.C06.CSV')
df=read.csv('2020-07-04_at_12-18-32pm.C11.CSV')

#Proprocessing (sensitivity of clustering if groups are highly unbalanced, so try to cut out smallest/dimmest material
#that is definitely not of interest, this will make clustering more robust)
D= df %>%
  filter(RED.B.HLog>1.5, FSC.HLog>0.25,SSC.HLog>0.25,GRN.B.HLog>0.25)%>%   #Change these thresholds per your data
  dplyr::select(c(RED.B.HLog,FSC.HLog,SSC.HLog,GRN.B.HLog,YEL.B.HLog))
df.subsamp= D[sample(1:nrow(D),min(nrow(D),5000)),]
#Cluster with DBSCAN model
Out = DBSCAN.SVM.process(df.subsamp,4)
df.subsamp$Cluster=predict(Out$dbop,df.subsamp)
DBSCAN.plot(df.subsamp)

#Classify uncertain with SVM model
Classified <- df.subsamp[df.subsamp$Cluster!=0,]
Unclassified<-df.subsamp[df.subsamp$Cluster==0,]
Pred <- predict(Out$svmop,newdata=Unclassified[,-6])
Unclassified$Cluster <- Pred
All  <- rbind(Classified, Unclassified)

#Plot and summarize each cluster in each file
Data.summary <- DBSCAN.plot(All)
