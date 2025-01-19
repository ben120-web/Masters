%% This script will investigate different combinations of predictor variables
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

% Start by loading our top ten parameters from 1(a)
bestPredictorVariables = load(); %% ADD FILEPATH

% Define arbitrary combinations
