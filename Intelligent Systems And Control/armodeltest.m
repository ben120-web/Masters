%% 3(A) This script will build and evaluate Linear 1-day ahead models and explore
% different AR/ARX models.

% LOAD IN DATASET
load('/Users/benrussell/Documents/Intelligent Systems and Control/Group10.mat');

% Remove one time slot as we only have 2 members
clear Date40;
clear X40;

% Now we need to split the data into training and validation sets.
trainingData11 = X11(contains(string(Date11), '2014') | contains(string(Date11), '2015'), :);
trainingData23 = X23(contains(string(Date23), '2014') | contains(string(Date23), '2015'), :);

validationData11 = X11(contains(string(Date11), '2015') | contains(string(Date11), '2016'), :);
validationData23 = X23(contains(string(Date23), '2015') | contains(string(Date23), '2016'), :);

trainingExogenous11 = trainingData11(:, 2 : end);
trainingTarget11 = trainingData11(:, 1);

trainingExogenous23 = trainingData23(:, 2 : end);
trainingTarget23 = trainingData23(:, 1);

validationTarget11 = validationData11(:, 1);
validationExogenous11 = validationData11(:, 2 : end);

validationTarget23 = validationData23(:, 1);
validationExogenous23 = validationData23(:, 2 : end);

%% Normalize The Data
normalised11Exogenous = mapminmax(trainingExogenous11');
normalised11Exogenous = normalised11Exogenous';

normalised11Target = mapminmax(trainingTarget11);
normalised11Target = normalised11Target';

normalisedValidationTarget11 = mapminmax(validationTarget11);
normalisedValidationTarget11 = normalisedValidationTarget11';

normalisedValidationExogenous11 = mapminmax(validationTarget11);
normalisedValidationExogenous11 = normalisedValidationExogenous11';

%% SET UP ADALINE MODEL

nLags = 5; % Number of auto-regressive lags

% Set training data (This is previous target data)
X_Train = zeros(length(normalised11Target) - nLags, nLags + size(trainingData11, 2));

% Start loop through each lag
for iLag = 1 : nLags

    X_Train(:, iLag) = normalised11Target(nLags - iLag + 1 : end - iLag);

end

for i = 1 : size(normalised11Exogenous, 2)

    X_Train(:, nLags + i) = normalised11Exogenous(nLags + 1 : end, i);
end

normalised11Target = normalised11Target(nLags + 1 : end);

% Initialise the ADALINE model
nInputs = size(X_Train, 2);

nOutputs = 1;

W = randn(nInputs, nOutputs); % Initialise weights randomly

b = randn(1, nOutputs); % Innitialise bias randomly

learningRate = 0.01;

nEpochs = 1000; % Num of epochs

%% TRAIN THE ADALINE MODEL
for iEpoch = 1 : nEpochs

    % Forward pass
    yPredTrain = X_Train * W + b;

    % Compute the loss
    lossTrain = mean((yPredTrain - normalised11Target) .^2);

    % Backward pass
    dW = (X_Train' * (yPredTrain - normalised11Target)) / size(X_Train, 1);

    dB = mean(yPredTrain - normalised11Target);

    % Update weights and bias
    W = W - learningRate * dW;
    b = b - learningRate * dB;

end

% Create the input output data for the validation set.
xTest = zeros(length(normalisedValidationTarget11) - nLags, nLags + ...
    size(normalisedValidationExogenous11, 2));

for iLags = 1 : nLags

    X_Test(:, iLags) = normalisedValidationTarget11(nLags - iLags + 1 : end - iLags);

end

for i = 1 : size(normalisedValidationExogenous11, 2)

    X_Test(:, nLags + iLags) = normalisedValidationExogenous11(nLags + 1 : end, i);
end

normalisedValidationTarget11 = normalisedValidationTarget11(nLags + 1 : end);

% Predict the next days value using the trained model with different AR
% models.
yPred = zeros(size(normalisedValidationTarget11));

for i = 1 : length(yPred)

    % AR model y(k) = a1y(k-1) + a2
    a1 = 0.5;
    a2 = 0;

    yPred = a1 * normalisedValidationTarget11(i) + a2;

    for j = 1 : nLags

        yPred(i) = yPred(i) * W(j) * normalisedValidationExogenous11(i, j);
    end
end




