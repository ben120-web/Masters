% ELE8066 assignment 2: Load  Forecasting
% Sample script -- for exploring MLP based forecasting models
% Se n McLoone, Feb 2023�
%close all
clear all
load Group12  %load the dataset.
% Note: Dataset contains variables for 3 different times of the day. If you
% are a group of 3 then you are expceted to develop models for all 3 times,
% and if you are a group of 2 then you may choose any two our the 3 time
% slots.
%%Select time of day to model and define the input and output variables
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
Y=Ysel; %Can also normalise the output, but not required if we use a linear ourput 
neuron.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  split into training, validation, and test datasets
% 2014-2105 = training
% 2016 = validation
% 2017-2018 = test
%
% You may choose different splits of 2014-2016 data when forming the traniing and 
validation datasets if you wish.
% But you must use 2017/2018 as the test dataset in all experiments
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Seleect variables and define dataset for linear model building
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Select variables as input for the linar prediction model
% y=a1*x1+a2*x2+ ...an*xn + a0     (ao =constant offset term)
SelVar=[1 18 23  30 32 35];   %Specify the index numbers of the variables you wish 
to include in the model
SelVarNames=vnames(SelVar) %Display the names of the selected variables
XLinTrain=[XTrain(:,SelVar) ones(size(YTrain))];  %vector of ones included for the 
offset term
XLinVal=[XVal(:,SelVar) ones(size(YVal))];     %vector of ones included for the 
offset term
XLinTest=[XTest(:,SelVar) ones(size(YTest))];     %vector of ones included for the 
offset term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build Linear model and evaluate its performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th=pinv(XLinTrain)*YTrain;  %computes the model parameters using the 'direct 
method'
YLinTrainPred=XLinTrain*th; %model predictions for the training data
YLinValPred=XLinVal*th; %model predictions for the test data
YLinTestPred=XLinTest*th; %model predictions for the test data
RMSETrainLinear = sqrt(mean((YTrain-YLinTrainPred).^2));  %the RMSE on the training
data
RMSETestLinear = sqrt(mean((YVal-YLinValPred).^2));  %the RMSE on the validation 
data
RMSETestLinear = sqrt(mean((YTest-YLinTestPred).^2));  %the RMSE on the test data
fprintf('\nLinear Model: RMSE (Training data)      = %2.2f MW\n',RMSETrainLinear); 
%print RMSE test error to screen
fprintf('Linear Model: RMSE (Val data)           = %2.2f MW\n',RMSETestLinear); 
%print RMSE test error to screen
fprintf('Linear Model: RMSE (Test data)          = %2.2f MW\n',RMSETestLinear); 
%print RMSE test error to screen
R2= corr(YTest,YLinTestPred,'row','pairwise')^2;  %the R-squared value for the test
data (for a perfect fit R2=1)
%display results
figure(1); clf
plot(YTest,YLinTestPred,'.') %scatter plot of actual versus predicted load on the 
test data
title(sprintf('Linear Model: R2=%2.2f',R2));
xlabel('Actual Power (MW)')
ylabel('Predicted Power (MW)')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Building an MLP model using Matlab's machine learning toolbox.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Note: You can use the tooblox gui 'nnstart' to explore the main steps of using
% the toolbox but using the command line functions gives greater control of the
% design and training process.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Define the Input and output training data for the MLP prediction model
XTrainNN = XLinTrain(:,1:end-1); %remove constant offset term as this is not needed
in a NN model
XValNN  = XLinVal(:,1:end-1);  %remove constant offset term as this is not needed 
in a NN model
XTestNN  = XLinTest(:,1:end-1);  %remove constant offset term as this is not needed
in a NN model
YTrainNN = YTrain;
YValNN  = YVal;
YTestNN  = YTest;
% Prepare dataset in format required for MLP toolbox
XdataNN=[XTrainNN  ; XValNN];  YdataNN=[YTrainNN ; YValNN];
idxTr=1:length(YTrain);
idxVal=(length(YTrain)+1):length(YdataNN);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ddefine the MLP model, training algorihtm and training parameters to use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create an empty NN model object - specifying the number of hidden layer neurons 
and taining algorithm to use
Nh=[10]; %if using one hidden layer
% Nh=[5 2]; %if using two hidden layers
%NNmod= fitnet(Nh,'traingd'); NNmod.trainParam.lr=0.00001; 
NNmod.trainParam.max_fail=100; %Gradient descent
%NNmod= fitnet(Nh,'traingdx');  NNmod.trainParam.max_fail=100; %Gradient descent 
with bold driving
NNmod= fitnet(Nh,'trainlm'); NNmod.trainParam.max_fail=10;  %Levenberg-Marquadt 
training algorithm
%Specify the type of activation functions to use in each layer if you want to 
override the defaults
%NNmod.layers{1}.transferFcn='logsig';  %defaults to 'tansig' (i.e. tanh) if not 
specified
% Defining the the training and validation dataset splits using by the NN training 
algorithm
NNmod.divideFcn='divideind';  %specify how 'train' function splits the data for 
training, validation and test datasets.
NNmod.divideParam.trainInd = idxTr;  %indexes of the training data.
NNmod.divideParam.valInd  = idxVal; %indexes of the validation data.
NNmod.divideParam.testInd = [];  %Not using test data during, so this is set to 
empty. (Can include test data to monitor performance during training, but this adds
to the computional complexity of training)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train the MLP model and evaluate its peformance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NNmod.trainParam.showWindow = true;  %set to true/false to see/not see a toolbox 
gui when the algorithm runs; 
[NNmodTrained trinfo] = train(NNmod,XdataNN',YdataNN'); %Train model based on 
datset {X,Y} and configuration parameters in NNmod
                                                        %Note: that the dataset is 
fed to the algorithm in tranposed form (i.e X' and Y').
%Compute Trained MLP model predictions and RMSE for Training, Validation and Test 
datasets
YNNestTrain=NNmodTrained(XTrainNN')';
YNNestVal=NNmodTrained(XValNN')';
YNNestTest=NNmodTrained(XTestNN')';
RMSETrainNN = sqrt(mean((YTrainNN-YNNestTrain).^2,'omitnan'));  %'omitnan' not 
needed here -- useful when dealing with datasets with missing values which are 
recorded as NaN
RMSEValNN= sqrt(mean((YValNN-YNNestVal).^2,'omitnan'));
RMSETestNN= sqrt(mean((YTestNN-YNNestTest).^2,'omitnan'));
%%
fprintf('\nNeuralNet Model: RMSE (Training data)   = %2.2f MW \n',RMSETrainNN); 
%print RMSE test error to screen
fprintf('NeuralNet Model: RMSE (Validation data) = %2.2f MW \n',RMSEValNN); %print 
RMSE test error to screen
fprintf('NeuralNet Model: RMSE (Test data)       = %2.2f MW \n',RMSETestNN); %print
RMSE test error to screen
R2= corr(YTest,YNNestTest,'row','pairwise')^2;  %the R-squared value for the test 
data (for a perfect fit R2=1)
%% display results
figure(2); clf
plot(YTest,YNNestTest,'.') %scatter plot of actual versus predicted
title(sprintf('NN Model: R2=%2.2f',R2));
xlabel('Actual Power (MW)')
ylabel('Predicted Power (MW)')
figure(3); clf
plot(DateTest,[YTestNN YNNestTest],'.')  % plot of predicted and actial load for 
test dataset
title(sprintf('NN Model: RMSE Test =%2.1f MW',RMSETestNN));
xlabel('Time (days)')
ylabel('Power (MW)')
legend('Actual', 'Predicted')