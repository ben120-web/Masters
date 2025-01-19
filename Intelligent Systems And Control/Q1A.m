%% This script will investigate individual parameters and show the top 10 most useful

%% Q1A

% Load the data into the workspace.
load('/Users/benrussell/Documents/Intelligent Systems and Control/Group10.mat');

% Remove one time slot (Only 2 used in this assignment)
clear Date40;
clear X40;

% Lets create a table of all predictors
dataTable11 = table();
dataTable23 = table();

% Append
for iField = 1 : numel(labels)

    dataTable11.(string(labels(iField))) = X11(:, iField);
    dataTable23.(string(labels(iField))) = X23(:, iField);
end

% Now we will loop through each field, and calculate the correlation
% between each predictor, and the respective load.

timeSlots = ["dataTable11", "dataTable23"];

% Set results structure
resultsTable1 = table();
resultsTable2 = table();

for iTimeOfDay = 1 : numel(timeSlots)

    if iTimeOfDay == 1
        time = timeSlots(iTimeOfDay);
        dataTable = dataTable11;
    else
        time = timeSlots(iTimeOfDay);
        dataTable = dataTable23;
    end

    % Grab the load data
    loadData = table2array(dataTable(:, 1));

    % Now we loop through each field
    for iPredictor = 1 : width(dataTable) - 1

        % Grab the predictor data
        predictorData = table2array(dataTable(:, iPredictor + 1)); % +1 As we ignore load

        % Get the predictor name
        predictorName = string(labels(iPredictor + 1));

        % Clean field name so it can be added to struct
        if contains(predictorName, '-')
            predictorName = erase(predictorName, '-');
        end

        if contains(predictorName, ' ')
            predictorName = erase(predictorName, ' ');
        end

        if contains(predictorName, '*')
            predictorName = erase(predictorName, '*');
        end

        if contains(predictorName, '/')
            predictorName = erase(predictorName, '/');
        end

        if contains(predictorName, '^')
            predictorName = erase(predictorName, '^');
        end

        if contains(predictorName, '(')
            predictorName = erase(predictorName, '(');
        end

        if contains(predictorName, ')')
            predictorName = erase(predictorName, ')');

        end


        if iTimeOfDay == 1

            % Now we have both datasets, we can compute the correlation and
            % save this in a structure.
            resultsTable1.predictorName(iPredictor) = predictorName;
            resultsTable1.Correlation(iPredictor) = corr(loadData, predictorData);
            resultsTable1.loadData(iPredictor) = {loadData};
            resultsTable1.predictorData(iPredictor) = {predictorData};

        else

            % Now we have both datasets, we can compute the correlation and
            % save this in a structure.
            resultsTable2.predictorName(iPredictor) = predictorName;
            resultsTable2.Correlation(iPredictor) = corr(loadData, predictorData);
            resultsTable2.loadData(iPredictor) = {loadData};
            resultsTable2.predictorData(iPredictor) = {predictorData};
        end



    end
end

% Lets plot all correlation values to see the best
figure(1);
scatter(1 : width(dataTable) - 1, abs(resultsTable1.Correlation));

ylabel("Correlation");
xlabel('Label Number');

figure(2);
scatter(1 : width(dataTable) - 1, abs(resultsTable2.Correlation));
ylabel("Correlation");
xlabel('Label Number');

% Sort rows based on highest correlation
sortedCorrValues1 = sortrows(resultsTable1, 'Correlation');
sortedCorrValues2 = sortrows(resultsTable2, 'Correlation');

% Select best 10 predictors in each time slot
bestPredictors(:, 1) = sortedCorrValues1(1 : 10, 1);
bestPredictors(:, 2) = sortedCorrValues2(1 : 10, 1);

%% Q1B
% This section will investigate different combinations of predictor variables
% and determine which of the predictor combinations performs best for a
% linear model.

% The combinations we will try are as follows:
%
%   1. Only Temperature variables
%   2. Only Calander related variables
%   3. A combination of weather and calander variables

% Model will be trained on 2014-2016 data and validated on 2017-2018 data.
% We will create a table with parameter details, alongside the RMS between
% training and validation datasets.

%% ---------------------------- BEGIN CODE --------------------------------

% List the best 10 predicitors for each time slot
bestPredictors = table2array(bestPredictors);

% Set the maximum number of predictos in each combination
MAX_NUM_OF_PREDICTORS = 5;

% Our best predictors will likely be variables that are contained in both
% time slots // Ie these parameters were consistently good.
bestPredictorsTemp = ["Temperatureover96hrs", "Temperatureover72hrs", ...
    "Temperatureover48hrs", "Temperatureover36hrs", "Temperatureover24hrs"];

% Clearly temperature is much more related to load than the other
% paramteters, so should be weighted heavily in our model.


%%  ------ Prepare the data for variable selection and modelling --------

% Loop through and normalise the predictor data
for iPredictorField = 1 : height(resultsTable1)

    % Pull data to be normalised
    dataToNormalise = cell2mat(resultsTable1.predictorData(iPredictorField));

    % Normalise the data
    [normalisedData, normParams] = mapstd(dataToNormalise');

    % Update table data
    resultsTable1.predictorData{iPredictorField} = normalisedData';

end

%%  split into training and test datasets
%% 2014-2106 = training
%% 2017-2018 = test

% Separate data from 2014 - 2016
trainingDataFlag = contains(string(Date11), '2014') | contains(string(Date11), '2015') | ...
    contains(string(Date11), '2016'); %% Same for both time slots

validationDataFlag = ~trainingDataFlag;

% Pull 5 paramteres to investigate
params = bestPredictorsTemp(1 : MAX_NUM_OF_PREDICTORS)';

% Get parameter data
paramData = resultsTable1.predictorData(contains(resultsTable1.predictorName, params));
paramLoad = resultsTable1.loadData(contains(resultsTable1.predictorName, params));

% Loop and filter out validation data
for i = 1 : numel(params)

    rawData = paramData{i};
    rawLoad = paramLoad{i};

    % Trim
    trainingData{i} = rawData(trainingDataFlag);
    trainingLoad{i} = rawLoad(trainingDataFlag);

    validationData{i} = rawData(validationDataFlag);
    validationLoad{i} = rawLoad(validationDataFlag);

end

% Convert data to matrix
trainingData = cell2mat(trainingData);
trainingLoad = cell2mat(trainingLoad);

validationData = cell2mat(validationData);
validationLoad = cell2mat(validationLoad);

% Now we start to build the model
xOptTrain = [trainingData ones(size(trainingLoad))];
xOptValidation = [validationData ones(size(validationLoad))];

% Compute th model parameters
th = pinv(xOptTrain) * trainingLoad;

% Compute predictions
YTrainPred = xOptTrain * th; %model predictions for the training data
YTestPred = xOptValidation * th; %model predictions for the test data

RMSETrainLinear = sqrt(mean((trainingLoad - YTrainPred).^2));
RMSETestLinear = sqrt(mean((validationLoad - YTestPred).^2));

R2= corr(validationLoad, YTestPred,'row','pairwise') ^ 2;

figure(1); clf
plot(validationLoad, YTestPred,'.') %scatter plot of actual versus predicted
title(sprintf('Linear Model: R2=%2.1f',R2));
xlabel('Actual Power (MW)')
ylabel('Predicted Power (MW)');

