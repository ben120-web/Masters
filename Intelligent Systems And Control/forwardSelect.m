function rankedParameters = forwardSelect(model)
%% Function to get best ARX parameters

% This fucntion will iterate through each paramter and calculate
% The RMSE of forecasted vs actual data and rank the predictors.

% Load the data
load('/Users/benrussell/Documents/Intelligent Systems and Control/Group10.mat');

% List the predictors
predictorLables = labels(2 : end);

exogenousData = X11(1 : 730, 2 : end);
loadData = X11(1 : 730, 1); % Training Data

validationData = X11(731 : end, 1);

% Start loop
for iPredictor = 1 : numel(predictorLables)

    % Extract predictor
    predictor = string(predictorLables(iPredictor));

    % Extract predictor data
    predictorData = exogenousData(:, iPredictor);

    IDdata = iddata(loadData, predictorData, 1, 'TimeUnit', 'Days');

    % Predict future data
    [yPrediction] = forecast(model, IDdata, numel(validationData));

    yPrediction = yPrediction.OutputData;

    % Compute performance stats
    mse = mean((validationData - yPrediction) .^ 2);
    rmse = sqrt(mse);

    % Save results
    rmseOut{iPredictor, 1} = predictor;
    rmseOut{iPredictor, 2} = rmse;
end

% Rank Parameters
rankedParameters = sortrows(rmseOut, 2);

% Top 5
rankedParameters = rankedParameters(1 : 5);
end