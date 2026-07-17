% ELE8066 assignment 2: Load  Forecasting
% Sample script -- for exploring linear forecasting models 
% Se n McLoone, Feb 2023�
%close all
clear all
load Group12  %load the dataset.
% Note: Dataset contains variables for 3 different times of the day. If you
% are a group of 3 then you are expceted to develop models for all 3 times,
% and if you are a group of 2 then you may choose any two our the 3 time
% slots.
%Select time of day to model and define the input and output variables
Xsel=X25(:,2:36);
Ysel=X25(:,1);
Date=Date25;
Vnames=labels(2:36);
Vnames(26)={'Sun durat*pot. sol. irrad.'};   %shorten longer variable names if 
needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Prepare the data for variable selection and modelling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
%Normalise input data to be in the range [-1,1]  or standardised (i.e. mean=0, 
std=1)
%[X, norm_params] = mapminmax(X0',-1,1); X=X';  %normalise all variables in the 
range [-1 1]
[Xnorm, norm_params] = mapstd(Xsel'); Xnorm=Xnorm'; %normalise all variables to 
have mean 0 and std of 1
Y=Ysel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  split into training and test datasets
%% 2014-2106 = training 
%% 2017-2018 = test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s2014=find(year(Date)==2014,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');
XTrain=Xnorm(s2014:f2016,:);
XTest=Xnorm(s2017:f2018,:);
DateTrain=Date(s2014:f2016);
DateTest=Date(s2017:f2018);
YTrain=Y(s2014:f2016);
YTest=Y(s2017:f2018);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Seleect variables and define dataset for linear model building
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Select variables as input for the linar prediction model
% y=a1*x1+a2*x2+ ...an*xn + a0     (ao =constant offset term)
SelVar=[2 18];  %Specify the index numbers of the variables you wish to include in 
the model (between 1 and 35)
SelVarNames=Vnames(SelVar)
XoptTrain=[XTrain(:,SelVar) ones(size(YTrain))];  %vector of ones included for the 
offset term
XoptTest=[XTest(:,SelVar) ones(size(YTest))];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build Linear model and evaluate its performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th=pinv(XoptTrain)*YTrain;  %computes the model parameters using the 'direct 
method'
YTrainPred=XoptTrain*th; %model predictions for the training data
YTestPred=XoptTest*th; %model predictions for the test data
RMSETrainLinear = sqrt(mean((YTrain-YTrainPred).^2));  %the RMSE on the training 
data  
RMSETestLinear = sqrt(mean((YTest-YTestPred).^2));  %the RMSE on the test data
fprintf('\nLinear Model: RMSE (Training data)      = %2.2f MW\n',RMSETrainLinear); 
%print RMSE test error to screen
fprintf('Linear Model: RMSE (Test data)          = %2.2f MW\n',RMSETestLinear); 
%print RMSE test error to screen
R2= corr(YTest,YTestPred,'row','pairwise')^2;  %the R-squared value for the test 
data (for a perfect fit R2=1)
%display results
figure(1); clf
plot(YTest,YTestPred,'.') %scatter plot of actual versus predicted
title(sprintf('Linear Model: R2=%2.1f',R2));
xlabel('Actual Power (MW)')
ylabel('Predicted Power (MW)')