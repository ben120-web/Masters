%% Develop a non-linaer ARX 1-day ahead prediction model

% Load the dataset
load('/Users/benrussell/Documents/Intelligent Systems and Control/Group10.mat');

clearvars Date40 X40;

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

% Preprocess the data
input_data = ModelTrainingData11(:,1:end-1)';
output_data = ModelTrainingData11(:,end)';
[input_data,ps_input] = mapminmax(input_data);
[output_data,ps_output] = mapminmax(output_data);

% Create the MLP ARX model
hidden_neurons = 10;
net = newff(input_data,output_data,hidden_neurons);

% Train the MLP ARX model
epochs = 100;
learning_rate = 0.01;
net.trainFcn = 'trainlm';
net.trainParam.epochs = epochs;
net.trainParam.lr = learning_rate;
[net,tr] = train(net,input_data,output_data);

% Test the MLP ARX model
test_input_data = ModelValidationData11(:,1:end-1)';
test_output_data = ModelValidationData11(:,end)';
[test_input_data,~] = mapminmax('apply',test_input_data,ps_input);
[test_output_data,~] = mapminmax('apply',test_output_data,ps_output);
test_output_prediction = net(test_input_data);
test_error = test_output_prediction - test_output_data;
test_mse = mean(test_error.^2);

% Make predictions
input_prediction = dataset(end, 1 : end-1)';
[input_prediction,~] = mapminmax('apply',input_prediction,ps_input);
output_prediction = net(input_prediction);
output_prediction = mapminmax('reverse',output_prediction,ps_output);
