function [processedSignal] = adaptiveMainsSubtractionFilter(signal, fs, Opts) %#codegen
% adaptiveMainsSubtractionFilter - This function uses a multi-stage
% filter to remove 50 or 60 Hz power-line noise with linear phase and
% minimal computation. Based on the algorithm described in [85] with minor
% modifications.
%
% The four stages are:
% 1. An adaptive notch filter is used to estimate the fundamental frequency
%    of the noise. If this is known, this can also be input.
% 2. Based on the estimated frequency, harmonics are generated using
%    discrete-time oscillators.
% 3. The amplitude and phase of each harmonic are then estimated using a
%    modified recursive least squares algorithm.
% 4. The estimated interference is subtracted from the signal.
%
% This is effectively an adaptive power line filter with no need to
% compute a separate mains reference (like in adaptiveMainsNoiseFilter.m).
%
% Syntax: [processedSignal] = adaptiveMainsSubtractionFilter(signal, fs, Opts)
%
% Inputs:
%    signal - A vector containing the input ECG signal.
%    fs - ECG signal sample frequency. (Hz)
%
%    Optional:
%    Opts - A scalar structure containing the following fields:
%       * powerLineFrequency - The frequency of the power line noise in Hz
%          if known. (Default: Not Set - Algorithm locates the frequency automatically)
%       * nHarmonicsToRemove - The number of power line frequency harmonics
%          to remove. (Default: 3)
%       * notch - A sub-structure containing the following fields:
%          * initialBandwidth - The initial notch bandwidth of the
%             frequency estimator in Hz. (Default: 30)
%          * asymptoticBandwidth - The asymptotic notch bandwidth of
%             the frequency estimator in Hz. (Default: 0.01)
%          * settlingTime - The settling time from initialBandwidth to
%             asymptoticBandwidth in seconds. (Default: 3)
%       * freqEstimator - A sub-structure containing the following fields:
%          * initialSettlingTime - Initial settling time of the frequency
%             estimator in seconds. (Default: 0.01)
%          * asymptoticSettlingTime - The asymptotic settling time of the
%             frequency estimator in seconds. (Default: 4)
%          * settlingTime - The settling time from initialSettlingTime to
%             asymptoticSettlingTime in seconds. (Default: 1)
%       * amplitudePhaseEstimator - A sub-structure containing the following
%          fields:
%          * settlingTime - The settling time of the amplitude and phase
%             estimator in seconds. (Default: 0.5)
%
% Outputs:
%    processedSignal - A vector containing the input signal with power line
%       interference removed. This is a linear filter so there is no phase
%       distortion or delay.
%
% Example:
%    [processedSignal] = adaptiveMainsSubtractionFilter(signal, fs, Opts)
%
% Other m-files required: movingMedianHighPassFilter.m
% Subfunctions: getOpts.
% MAT-files required: none.
%
% ========================= COPYRIGHT NOTICE =========================
% The contents of this file/document are protected under copyright and
% other intellectual property laws and contain B-Secur Ltd. sensitive
% commercial and/or technical confidential information and must not be
% used other than for the purpose for which it was provided.
%
% If you have received this file/document in error, please send it back
% to us, and immediately and permanently delete it. Please do not use,
% copy, disclose or otherwise exploit the information contained in this
% file/document if you are not an authorized recipient.
%
% All content is copyright B-Secur Ltd. 2020-2022.
% ====================================================================

%------------- BEGIN CODE --------------

% Validate Inputs.
% Check the number of input arguments.
minArgs = 2;
maxArgs = 3;
narginchk(minArgs, maxArgs)


% Get Opts structure.
if nargin < maxArgs

    Opts = getOpts([], fs);

else

    Opts = getOpts(Opts, fs);

end

% Set constants.
DEFAULT_BANDPASS = [40, 70]; % [Hz]
FILTER_ORDER = 4;

% Define in-line functions for converting inputs. [Eq 15]
convertToForgettingFactor = @(x) exp(log(0.05) / (x * fs + 1));
convertToPoleRadii = @(x) (1 - tan(pi * x / fs)) / (1 + tan(pi * x / fs));

% Pre-allocate.
nSamples = numel(signal); % Ref: N
processedSignal = zeros(size(signal)); % Ref: S^(n)

method = Opts.baselineRemoval.method;

switch method

    case 'movmean'

        % Zero mean the signal. In reality this is not required if there is a 0.5
        % Hz hardware high pass filter.
        signal = signal(:);

        % Accumulate mean
        runningSum = 0;

        for iSample = 1 : nSamples

            runningSum = runningSum + signal(iSample);
            signal(iSample) = signal(iSample) - (runningSum / iSample);

        end

    case 'movmedian'

        % High pass filter to zero mean the signal.
        windowDuration = Opts.baselineRemoval.windowDuration;
        signal = movingMedianHighPassFilter(signal, fs, windowDuration);

    otherwise

        % Method is invalid, throw an error.
        ErrorStruct.message = sprintf('%s is an invalid method', method);
        ErrorStruct.identifier = [mfilename, ':invalidBaselineRemovalMethod'];
        error(ErrorStruct);

end

% Variables in this section are left in mathematical notation for ease of
% review vs reference.
% Convert inputs to pole radii.
alphaF = convertToPoleRadii(Opts.notch.initialBandwidth); % Ref: a_f
alphaInf = convertToPoleRadii(Opts.notch.asymptoticBandwidth); % Ref: a_inf

% Convert inputs to forgetting factors.
alphaSt = convertToForgettingFactor(Opts.notch.settlingTime); % Ref: a_st
lambdaF = convertToForgettingFactor(Opts.freqEstimator.initialSettlingTime); % Ref: ^_f
lambdaInf = convertToForgettingFactor(Opts.freqEstimator.asymptoticSettlingTime); % Ref: ^_inf
lambdaSt = convertToForgettingFactor(Opts.freqEstimator.settlingTime); % Ref: ^_st

% Set the smoothing parameter (which has a fixed cut-off of 90 Hz).
gammaSmooth = convertToPoleRadii(0.5 * min(90, fs / 2)); % Ref: y

% Set up the phase / amplitude forgetting factor (might be multiple).
lambdaA = convertToForgettingFactor(Opts.amplitudePhaseEstimator.settlingTime);
lambdaA = lambdaA * ones(1, Opts.nHarmonicsToRemove); % Ref: ^_a

% Initialise lattice variables.
kappaF = 0; % Ref: k_f
kappaK = zeros(1, Opts.nHarmonicsToRemove); % Ref: k_k
latticeC = 5; % Ref: C
latticeD = 10; % Ref: D
fn1 = 0;
fn2 = 0;

% Initialise the oscillator.
ukp = ones(1, Opts.nHarmonicsToRemove); % Ref: u_k
uk = ones(1, Opts.nHarmonicsToRemove); % Ref: u'_k

% Initialise RLS parameters.
rlsR1 = 10 * ones(1, Opts.nHarmonicsToRemove); % Ref: r1_k
rlsR4 = 10 * ones(1, Opts.nHarmonicsToRemove); % Ref: r4_k
rlsA = zeros(1, Opts.nHarmonicsToRemove); % Ref: b^_k
rlsB = zeros(1, Opts.nHarmonicsToRemove); % Ref: c^_k

% IIR bandpass.
% Phase distortion doesn't matter here as we are only using this signal for
% frequency estimation.
if isempty(Opts.powerLineFrequency)

    filterCutOffs = DEFAULT_BANDPASS;

else

    filterCutOffs = [Opts.powerLineFrequency - 2, Opts.powerLineFrequency + 2];

end

% Design the filter and convert to SOS format. Note that filter order is
% divided by 2 because butter designs bandpass filters of order 2n.
[z, p, k] = butter(FILTER_ORDER / 2, filterCutOffs / (fs / 2), 'bandpass');
sosParams = real(zp2sos(z, p, k));

% Filter and differentiate the signal. [Eq 4]
% You could move this into the for loop for sample by sample calculation
% (real time) but in MATLAB much more efficient to vector compute.
filteredSignal = sosfilt(sosParams, signal); % Ref: x_f
filteredSignal = [0; diff(filteredSignal)]; % Ref: x_d

% Loop over each sample.
for iSample = 1 : nSamples

    % Compute the output of the lattice filter. [Eq 6]
    fn = filteredSignal(iSample) + kappaF * (1 + alphaF) * fn1 - alphaF * fn2;

    % Compute the updated frequency estimation.
    % Note this is different from the reference paper. Original reference
    % for lattice algorithm shows extra terms which are required.
    latticeC = lambdaF * latticeC + (1 - lambdaF) * fn1 * (fn + fn2);
    latticeD = lambdaF * latticeD + (1 - lambdaF) * 2 * fn1 ^ 2;
    kappaT = latticeC / latticeD; % Ref: k_t

    % Limit kappaT.
    if kappaT > 1

        kappaT = 1;

    elseif kappaT < -1

        % This shouldn't be possible but is included here to ensure filter
        % stability.
        kappaT = -1;

    end

    % Update kappaF.
    kappaF = gammaSmooth * kappaF + (1 - gammaSmooth) * kappaT;

    % Update the previous lattice values.
    fn2 = fn1;
    fn1 = fn;

    % Update bandwidths and forgetting factors. [Eq 7]
    alphaF = alphaSt * alphaF + (1 - alphaSt) * alphaInf;
    lambdaF = lambdaSt * lambdaF + (1 - lambdaSt) * lambdaInf;

    % Remove harmonics.
    instantaneousError = signal(iSample); % Ref: e

    % Loop over each harmonic.
    for jHarmonic = 1 : Opts.nHarmonicsToRemove

        % Compute harmonic. [Eq 9]
        % Reference paper does this at the end of the loop but it's easier
        % to compute in line.
        if jHarmonic == 1

            kappaK(jHarmonic) = kappaF;

        elseif jHarmonic == 2

            kappaK(jHarmonic) = 2 * kappaF ^ 2 - 1;

        else

            kappaK(jHarmonic) = 2 * kappaF * kappaK(jHarmonic - 1) ...
                - kappaK(jHarmonic - 2);

        end

        % Discrete oscillator.
        s1 = kappaK(jHarmonic) * (uk(jHarmonic) + ukp(jHarmonic));
        s2 = ukp(jHarmonic);
        ukp(jHarmonic) = s1 - uk(jHarmonic);
        uk(jHarmonic) = s1 + s2;

        % Compute gain control.
        gainControl = 1.5 - (ukp(jHarmonic) ^ 2 - uk(jHarmonic) ^ 2 * ...
            (kappaK(jHarmonic) - 1) / (kappaK(jHarmonic) + 1)); % Ref: G

        % Limit gain control.
        if gainControl < 0

            gainControl = 1;

        end

        % Scale by gain control.
        ukp(jHarmonic) = gainControl * ukp(jHarmonic);
        uk(jHarmonic) = gainControl * uk(jHarmonic);

        % Amplitude and Phase estimation via recursive least squares. [Eq 13]
        hk = rlsA(jHarmonic) * ukp(jHarmonic) + rlsB(jHarmonic) * uk(jHarmonic);
        instantaneousError = instantaneousError - hk;
        rlsR1(jHarmonic) = lambdaA(jHarmonic) * rlsR1(jHarmonic) + ukp(jHarmonic) ^ 2;
        rlsR4(jHarmonic) = lambdaA(jHarmonic) * rlsR4(jHarmonic) + uk(jHarmonic) ^ 2;
        rlsA(jHarmonic) = rlsA(jHarmonic) + instantaneousError * ukp(jHarmonic) / rlsR1(jHarmonic);
        rlsB(jHarmonic) = rlsB(jHarmonic) + instantaneousError * uk(jHarmonic) / rlsR4(jHarmonic);

    end

    % The filter error is the remaining true signal. [Eq 14]
    processedSignal(iSample) = instantaneousError;

end

end

function Opts = getOpts(Opts, fs)

OptsDefault = struct();
OptsDefault.powerLineFrequency = [];
OptsDefault.nHarmonicsToRemove = 3;
OptsDefault.notch.initialBandwidth = 30; % [Hz]
OptsDefault.notch.asymptoticBandwidth = 0.01; % [Hz]
OptsDefault.notch.settlingTime = 3; % [Sec]
OptsDefault.freqEstimator.initialSettlingTime = 0.01; % [Hz]
OptsDefault.freqEstimator.asymptoticSettlingTime = 4; % [Sec]
OptsDefault.freqEstimator.settlingTime = 1; % [Sec]
OptsDefault.amplitudePhaseEstimator.settlingTime = 0.5; % [Sec]
OptsDefault.baselineRemoval.method = 'movmean';
OptsDefault.baselineRemoval.windowDuration = 0.3;

if isempty(Opts)

    Opts = OptsDefault;

elseif isstruct(Opts)

    % Check scalar.
    validateattributes(Opts, {'struct'}, {'scalar'}, mfilename, 'Opts', 3);

    % Validate powerLineFrequency.
    if isfield(Opts, 'powerLineFrequency') && ~isempty(Opts.powerLineFrequency)

        validateattributes(Opts.powerLineFrequency, {'numeric'}, {'scalar', ...
            'integer', 'positive', '<', fs}, mfilename, 'Opts.powerLineFrequency', 3);

    else

        Opts.powerLineFrequency = OptsDefault.powerLineFrequency;

    end

    % Validate nHarmonicsToRemove.
    if isfield(Opts, 'nHarmonicsToRemove')

        validateattributes(Opts.nHarmonicsToRemove, {'numeric'}, {'scalar', ...
            'integer', 'positive'}, mfilename, 'Opts.nHarmonicsToRemove', 3);

    else

        Opts.nHarmonicsToRemove = OptsDefault.nHarmonicsToRemove;

    end

    % Validate notch.initialBandwidth.
    if isfield(Opts, 'notch')

        if isfield(Opts.notch, 'initialBandwidth')

            validateattributes(Opts.notch.initialBandwidth, {'numeric'}, {'scalar', ...
                'finite', 'positive'}, mfilename, 'Opts.notch.initialBandwidth', 3);

        else

            Opts.notch.initialBandwidth = OptsDefault.notch.initialBandwidth;

        end

        % Validate notch.asymptoticBandwidth.
        if isfield(Opts.notch, 'asymptoticBandwidth')

            validateattributes(Opts.notch.asymptoticBandwidth, {'numeric'}, {'scalar', ...
                'finite', 'positive'}, mfilename, 'Opts.notch.asymptoticBandwidth', 3);

        else

            Opts.notch.asymptoticBandwidth = OptsDefault.notch.asymptoticBandwidth;

        end

        % Validate notch.settlingTime.
        if isfield(Opts.notch, 'settlingTime')

            validateattributes(Opts.notch.settlingTime, {'numeric'}, {'scalar', ...
                'finite', 'positive'}, mfilename, 'Opts.notch.settlingTime', 3);

        else

            Opts.notch.settlingTime = OptsDefault.notch.settlingTime;

        end

    else

        Opts.notch.initialBandwidth = OptsDefault.notch.initialBandwidth;
        Opts.notch.asymptoticBandwidth = OptsDefault.notch.asymptoticBandwidth;
        Opts.notch.settlingTime = OptsDefault.notch.settlingTime;

    end

    % Validate freqEstimator.initialSettlingTime.
    if isfield(Opts, 'freqEstimator')

        if isfield(Opts.freqEstimator, 'initialSettlingTime')

            validateattributes(Opts.freqEstimator.initialSettlingTime, {'numeric'}, {'scalar', ...
                'finite', 'positive'}, mfilename, 'Opts.freqEstimator.initialSettlingTime', 3);

        else

            Opts.freqEstimator.initialSettlingTime = OptsDefault.freqEstimator.initialSettlingTime;

        end

        % Validate freqEstimator.asymptoticSettlingTime.
        if isfield(Opts.freqEstimator, 'asymptoticSettlingTime')

            validateattributes(Opts.freqEstimator.asymptoticSettlingTime, {'numeric'}, {'scalar', ...
                'finite', 'positive'}, mfilename, 'Opts.freqEstimator.asymptoticSettlingTime', 3);

        else

            Opts.freqEstimator.asymptoticSettlingTime = OptsDefault.freqEstimator.asymptoticSettlingTime;

        end

        % Validate freqEstimator.settlingTime.
        if isfield(Opts.freqEstimator, 'settlingTime')

            validateattributes(Opts.freqEstimator.settlingTime, {'numeric'}, {'scalar', ...
                'finite', 'positive'}, mfilename, 'Opts.freqEstimator.settlingTime', 3);

        else

            Opts.freqEstimator.settlingTime = OptsDefault.freqEstimator.settlingTime;

        end

    else

        Opts.freqEstimator.initialSettlingTime = OptsDefault.freqEstimator.initialSettlingTime;
        Opts.freqEstimator.asymptoticSettlingTime = OptsDefault.freqEstimator.asymptoticSettlingTime;
        Opts.freqEstimator.settlingTime = OptsDefault.freqEstimator.settlingTime;

    end

    % Validate amplitudePhaseEstimator.settlingTime.
    if isfield(Opts, 'amplitudePhaseEstimator') && isfield(Opts.amplitudePhaseEstimator, 'settlingTime')

        validateattributes(Opts.amplitudePhaseEstimator.settlingTime, {'numeric'}, {'scalar', ...
            'finite', 'positive'}, mfilename, 'Opts.amplitudePhaseEstimator.settlingTime', 3);

    else

        Opts.amplitudePhaseEstimator.settlingTime = OptsDefault.amplitudePhaseEstimator.settlingTime;

    end

    % Validate baselineRemoval.method.
    if isfield(Opts, 'baselineRemoval') && isfield(Opts.baselineRemoval, 'method')

        validateattributes(Opts.baselineRemoval.method, {'char',...
            'string'}, {'scalartext'}, mfilename, 'Opts.baselineRemoval.method', 3);

    else

        Opts.baselineRemoval.method = OptsDefault.baselineRemoval.method;

    end

    % Validate baselineRemoval.windowDuration.
    if isfield(Opts, 'baselineRemoval') && isfield(Opts.baselineRemoval, 'windowDuration')

        validateattributes(Opts.baselineRemoval.windowDuration, {'numeric'}, {'scalar', ...
            'finite', 'positive'}, mfilename, 'Opts.baselineRemoval.windowDuration', 3);

    else

        Opts.baselineRemoval.windowDuration = OptsDefault.baselineRemoval.windowDuration;

    end

else

    ErrorStruct.message = 'Supplied Opts were invalid.';
    ErrorStruct.identifier = [mfilename, ':invalidOpts'];
    error(ErrorStruct);

end

end

%------------- END OF CODE -------------