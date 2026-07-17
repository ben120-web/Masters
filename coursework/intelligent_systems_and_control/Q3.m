%% This script will perform Q3 of the assignment.

%% DATA PRE-PROCESSING
% Load the data into the workspace.
load('/Users/benrussell/Documents/Intelligent Systems and Control/Group10.mat');

% Remove one time slot (Only 2 used in this assignment)
clear Date40;
clear X40;

% Now we need to split the data into training and validation sets.
trainingData11 = X11(contains(string(Date11), '2014') | ...
    contains(string(Date11), '2015'), :); % Train model on 2-years data.
trainingData23 = X23(contains(string(Date23), '2014') | ...
    contains(string(Date23), '2015'), :);

validationData11 = X11(contains(string(Date11), '2016'), :); % Validate performance on 1 year data
validationData23 = X23(contains(string(Date23), '2016'), :);

% For AR model we are only interested in load data.
ModelTrainingData11 = trainingData11(:, 1);
ModelValidationData11 = validationData11(:, 1);

ModelTrainingData23 = trainingData23(:, 1);
ModelValidationData23 = validationData23(:, 1);

%% Build an LINEAR Auto-regressive models of the form y(k) = a1y(k-1) + a2
% This is a first order AR model and we will use the AR function
sampleTime = 1;

IDdata = iddata(ModelTrainingData11, [], sampleTime, 'TimeUnit', 'Days');
trend = getTrend(IDdata, 0);
IDdata = detrend(IDdata, 0);

opts = arOptions('Approach', 'ls', 'window', 'ppw');

% Set model order (Here the order is 1, since we are only interested in
% the previous day)
modelOrder = 1; % Increase this to improve accuarcy.

% Build model
arModel = ar(IDdata, modelOrder, opts);

% Predict data to validate model performance
yp = predict(arModel, ModelTrainingData11, 1);
figure(1);
plot(yp, 'r');
hold on
plot(ModelTrainingData11, 'b'); %% MODEL WORKS OK
title('Predicted model data on training set');
legend('predicted data', 'trainingData');

% Once we have validated the model, we use forecast.
[yPrediction] = forecast(arModel, IDdata, numel(ModelValidationData11));

% retrend
IDdata = retrend(IDdata, trend);
Y = retrend(yPrediction, trend);

predictedData = Y.OutputData;

% Compute performance stats
mse = mean((ModelValidationData11 - predictedData).^2);
rmse = sqrt(mse);

figure(2);
plot(predictedData, 'r');
hold on
plot(ModelValidationData11, 'b');
title('order = ' + string(modelOrder) + ' AR model on electricty load'...
    + '|| Model RMSE : ' + string(rmse));
legend('predicted Data', 'Actual Data');
xlabel('Days');
ylabel('Electricity load (kW)');

%% BUILD AN ARMA Model

% Same data as before
IDdata = iddata(ModelTrainingData11, [], sampleTime, 'TimeUnit', 'Days');
trend = getTrend(IDdata, 0);
IDdata = detrend(IDdata, 0); % Linear so detrend

% Specify coefficients
na = modelOrder;
nc = modelOrder;

armaModel = armax(IDdata, [na nc]);

% Predict data to validate model performance
yp = predict(armaModel, ModelTrainingData11, 1);
figure(1);
plot(yp, 'r');
hold on
plot(ModelTrainingData11, 'b'); %% MODEL WORKS OK
title('Predicted model data on training set');
legend('predicted data', 'trainingData');

% Once we have validated the model, we use forecast.
[yPrediction] = forecast(armaModel, IDdata, numel(ModelValidationData11));

% retrend
IDdata = retrend(IDdata, trend);
Y = retrend(yPrediction, trend);

predictedData = Y.OutputData;

% Compute performance stats
mse = mean((ModelValidationData11 - predictedData).^2);
rmse = sqrt(mse);

figure(2);
plot(predictedData, 'r');
hold on
plot(ModelValidationData11, 'b');
title('order = ' + string(modelOrder) + ' ARMA model on electricty load'...
    + '|| Model RMSE : ' + string(rmse));
legend('predicted Data', 'Actual Data');
xlabel('Days');
ylabel('Electricity load (kW)');


%% Build a LINEAR Auto-regressive exogenous (ARX) model of the form y(k) = a1y(k-1) +a2T(k) + a3T(k-1)

% Define exogenous input data
exogenousData = trainingData11(:, 2); % Lets use best performing temperature parameter.

trainingData = [ModelTrainingData11 exogenousData];
validationData = [ModelValidationData11 validationData11(:, 2)];

IDdata = iddata(ModelTrainingData11, exogenousData, sampleTime, 'TimeUnit', 'Days');

% Specify model order. Note we are looking at a first order model from Q.
na = 20; % Order of output data (y) - 1 as we have max y(k-1)
nb = 20; % Order of exogenous data T - 1 as we have T(k-1)
nk = 20; % System Delay - 1 as we are using previous day data.

% Estimate the ARX model
linearArxModel = arx(IDdata, [na nb nk]);

% Predict data to validate model performance
predictionDataArx = predict(linearArxModel, trainingData, 1);
figure(3);
plot(predictionDataArx, 'r');
hold on
plot(ModelTrainingData11, 'b'); %% MODEL WORKS OK

% Once we have validated the model, we use forecast.
futureInputs = validationData11(:, 2);
[yPrediction] = forecast(linearArxModel, IDdata, numel(ModelValidationData11), ...
    futureInputs);

yPrediction = yPrediction.OutputData;

% Compute performance stats
mse = mean((ModelValidationData11 - yPrediction).^2);
rmse = sqrt(mse);

figure(4);
plot(yPrediction, 'r');
hold on
plot(ModelValidationData11, 'b');
title('order = ' + string(modelOrder) + ' Linear ARX model on electricty load'...
    + '|| Model RMSE : ' + string(rmse));
legend('predicted Data', 'Actual Data');
xlabel('Days');
ylabel('Electricity load (kW)');

%% BUILD ARMIX MODEL
% Same data as before
% Define exogenous input data
% Define exogenous input data
exogenousData = trainingData11(:, 2); % Lets use best performing temperature parameter.

trainingData = [ModelTrainingData11 exogenousData];
validationData = [ModelValidationData11 validationData11(:, 2)];

IDdata = iddata(ModelTrainingData11, exogenousData, sampleTime, 'TimeUnit', 'Days');

% Specify model order. Note we are looking at a first order model from Q.
na = modelOrder; % Order of output data (y) - 1 as we have max y(k-1)
nb = modelOrder; % Order of exogenous data T - 1 as we have T(k-1)
nc = modelOrder;
nk = modelOrder; % System Delay - 1 as we are using previous day data.

% Estimate the ARX model
linearArmaxModel = armax(IDdata, [na nb nc nk]);

% Predict data to validate model performance
predictionDataArx = predict(linearArxModel, trainingData, 1);
figure(3);
plot(predictionDataArx, 'r');
hold on
plot(ModelTrainingData11, 'b'); %% MODEL WORKS OK

% Once we have validated the model, we use forecast.
futureInputs = validationData11(:, 2);
[yPrediction] = forecast(linearArxModel, IDdata, numel(ModelValidationData11), ...
    futureInputs);

yPrediction = yPrediction.OutputData;

% Compute performance stats
mse = mean((ModelValidationData11 - yPrediction).^2);
rmse = sqrt(mse);

figure(4);
plot(yPrediction, 'r');
hold on
plot(ModelValidationData11, 'b');
title('order = ' + string(modelOrder) + ' Linear ARMAX model on electricty load'...
    + '|| Model RMSE : ' + string(rmse));
legend('predicted Data', 'Actual Data');
xlabel('Days');
ylabel('Electricity load (kW)');

%% EXTEND THE SCRIPT TO ADD A FORWARD SELECTION METHOD

% Call function to select best performing parameters in ARX model.
[top5ARXParameters] = forwardSelect(linearArxModel);