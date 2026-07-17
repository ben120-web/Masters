% LOAD FORCASTING MLP MODEL

clear all

% Load Data
load('/Users/benrussell/Documents/Intelligent Systems and Control/Group10.mat');

% Select a time of the day to model
% and define the input and output variables.
selectedX = X11(:, 2 : 36);
selectedY = X11(:, 1);

date = Date11;

vNames = labels(2 : 36);


%% Prepare the data for variable selection and modelling

% Normalise the input data to be in the range [-1, 1]
[X, norm_params] = mapminmax(X11,-1, 1);

X = X';

%% Split into training, validation and test datasets
% 2014 - 2015 - training
% 2016 - Validation
% 2017 - 2018 - Test





