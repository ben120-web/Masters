% ELE8066 assignment 2: Load  Forecasting
% Sample script -- selecting the optimum variable using using forward selection for
linear regression
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
vnames=labels(2:36);
vnames(26)={'Sun durat*pot. sol. irrad.'};   %shorten longer variable names if 
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
%%  split into training, validation, and test datasets
%% 2014-2105 = training 
%% 2016 = validation 
%% 2017-2018 = test
%%
%% You may choose different splits of 2014-2016 data when forming the traniing and 
validation datasets if you wish.
%% But you must use 2017/2018 as the test dataset in all experiments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s2014=find(year(Date)==2014,1,'first');
f2015=find(year(Date)==2015,1,'last');
s2016=find(year(Date)==2016,1,'first');
f2016=find(year(Date)==2016,1,'last');
s2017=find(year(Date)==2017,1,'first');
f2018=find(year(Date)==2018,1,'last');
XTrain=Xnorm(s2014:f2015,:);
XVal=Xnorm(s2016:f2016,:);
XTest=Xnorm(s2017:f2018,:);
DateTrain=Date(s2014:f2015);
DateVal=Date(s2016:f2016);
DateTest=Date(s2017:f2018);
YTrain=Y(s2014:f2015);
YVal=Y(s2016:f2016);
YTest=Y(s2017:f2018);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Autoselect best linear model variables using forward selection method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nvar=size(XTrain,2);
AvailVar=1:1:Nvar; % Variables to select from 
SelVar=[]; %Offset variable selected initially
for k=1:1:Nvar
    RMSE_Train=zeros(length(AvailVar),1)*NaN;
    RMSE_Val=zeros(length(AvailVar),1)*NaN;
    for i=1:1:length(AvailVar)
        
        Xall=[XTrain(:,[SelVar AvailVar(i)]) ones(size(YTrain))];
        [rnanX,cnan,vnan]=find(isnan(Xall)==1); %remove NaN data for pinv calc.
        [rnanY,cnan,vnan]=find(isnan(YTrain)==1); %remove NaN data for pinv calc.
        rnan=union(rnanX,rnanY);
        idxAll=1:length(YTrain);
        rvalid=setdiff(idxAll,rnan);
        th=pinv(Xall(rvalid,:))*YTrain(rvalid);
        
        YestTrain=Xall*th;
        RMSE_Train(i)= sqrt(mean((YTrain-YestTrain).^2,'omitnan'));
        
        Xall_Val=[XVal(:,[SelVar AvailVar(i)]) ones(size(YVal))];
        YestVal=Xall_Val*th;
        
        RMSE_Val(i)= sqrt(mean((YVal-YestVal).^2,'omitnan'));
    end
    [minVal minIdx]=min(RMSE_Val);
    SelVar=[SelVar AvailVar(minIdx)];  %add the variable with the lowest RMSE to 
the selected variables.
    AvailVar(minIdx)=[]; % Remove it from the available variables.
    RMSEBest_Val(k,1)=minVal;
    RMSEBest_Train(k,1)=RMSE_Train(minIdx);
end
[RMSEOpt Kopt]=min(RMSEBest_Val);
% Plot showing training and test RMSE with the selected variables
figure(2); clf
xcat = categorical(vnames(SelVar));
xcat=reordercats(xcat,vnames(SelVar));
subplot(2,1,1)
plot(xcat,[RMSEBest_Train],'k','LineWidth',2); grid on; hold on
plot(xcat,[RMSEBest_Train],'r*','LineWidth',2);
ylabel('RMSE-{tr}')
subplot(2,1,2)
plot(xcat,[RMSEBest_Val],'b','LineWidth',2); grid on; hold on
plot(xcat,[RMSEBest_Val],'r*','LineWidth',2);
plot(xcat(Kopt),RMSEOpt,'ko','LineWidth',2)
ylabel('RMSE-{Val}')
%display to screen the optimum variables in order of selection
fprintf('\nOptimum Forward selected variables(n=%d/%d): \n\n',Kopt,Nvar);
for k=1:1:Kopt
    fprintf('%2d: %-30s    RMSE: %2.2f \n',k, 
string(vnames(SelVar(k))),RMSEBest_Val(k));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define dataset for model building based on optimum variable selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VarOpt=SelVar(1:Kopt);
XoptTrain=[XTrain(:,SelVar(1:Kopt)) ones(size(YTrain))];
XoptVal=[XVal(:,SelVar(1:Kopt)) ones(size(YVal))];
XoptTest=[XTest(:,SelVar(1:Kopt)) ones(size(YTest))];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build Linear model (with Optimum variables) and evaluate its performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th=pinv(XoptTrain)*YTrain;  %computes the model parameters using the 'direct 
method'
YTrainPred=XoptTrain*th; %model predictions for the training data
YTestPred=XoptTest*th; %model predictions for the test data
YValPred=XoptVal*th;  %model predictions for the validation dataset
RMSETrainLinear = sqrt(mean((YTrain-YTrainPred).^2));  %the RMSE on the training 
data  
RMSEValLinear = sqrt(mean((YVal-YValPred).^2));  %the RMSE on the test data
RMSETestLinear = sqrt(mean((YTest-YTestPred).^2));  %the RMSE on the test data
fprintf('\nLinear Model: RMSE (Training data)      = %2.2f MW\n',RMSETrainLinear); 
%print RMSE test error to screen
fprintf('Linear Model: RMSE (Validation data)    = %2.2f MW\n',RMSEValLinear); 
%print RMSE test error to screen
fprintf('Linear Model: RMSE (Test data)          = %2.2f MW\n',RMSETestLinear); 
%print RMSE test error to screen
R2= corr(YTest,YTestPred,'row','pairwise')^2;  %the R-squared value for the test 
data (for a perfect fit R2=1)
%display results
figure(3); clf
plot(YTest,YTestPred,'.') %scatter plot of actual versus predicted
title(sprintf('Linear Model: R2=%2.1f',R2));
xlabel('Actual Power (MW)')
ylabel('Predicted Power (MW)')