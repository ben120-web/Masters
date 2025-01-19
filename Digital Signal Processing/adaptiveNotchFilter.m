function processedSignal = adaptiveNotchFilter(signal, fs)
% This function will perform adaptive notch filtering as detailed in
% [85].
%
% The four stages are:
% 1. An adaptive notch filter is used to estimate the fundamental frequency
%    of the noise. If this is known, this can also be input.
% 2. Based on the estimated frequency, harmonics are generated using
%    discrete-time oscillators.
% 3. The amplitude and phase of each harmonic are then estimated using a
%    modified recursive least squares algorithm.
% 4. The estimated interference is subtracted from the signal.

DEFAULT_BANDPASS = [40, 70]; % [Hz]
FILTER_ORDER = 4;
NUM_OF_HARMONICS_TO_REMOVE = 3;

% Define in-line functions for converting inputs. [Eq 15]
convertToForgettingFactor = @(x) exp(log(0.05) / (x * fs + 1));
convertToPoleRadii = @(x) (1 - tan(pi * x / fs)) / (1 + tan(pi * x / fs));

% Pre-allocate.
nSamples = numel(signal); % Ref: N
processedSignal = zeros(size(signal)); % Ref: S^(n)

% Zero mean the signal. In reality this is not required if there is a 0.5
% Hz hardware high pass filter.
signal = signal(:);

% Accumulate mean
runningSum = 0;

for iSample = 1 : nSamples

    runningSum = runningSum + signal(iSample);
    signal(iSample) = signal(iSample) - (runningSum / iSample);

end

% Convert inputs to pole radii.
alphaF = convertToPoleRadii(30); % Ref: a_f
alphaInf = convertToPoleRadii(0.01); % Ref: a_inf

% Convert inputs to forgetting factors.
alphaSt = convertToForgettingFactor(3); % Ref: a_st
lambdaF = convertToForgettingFactor(0.01); % Ref: ^_f
lambdaInf = convertToForgettingFactor(4); % Ref: ^_inf
lambdaSt = convertToForgettingFactor(1); % Ref: ^_st

% Set the smoothing parameter (which has a fixed cut-off of 90 Hz).
gammaSmooth = convertToPoleRadii(0.5 * min(90, fs / 2)); % Ref: y

% Set up the phase / amplitude forgetting factor (might be multiple).
lambdaA = convertToForgettingFactor(0.5);
lambdaA = lambdaA * ones(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: ^_a

% Initialise lattice variables.
kappaF = 0; % Ref: k_f
kappaK = zeros(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: k_k
latticeC = 5; % Ref: C
latticeD = 10; % Ref: D
fn1 = 0;
fn2 = 0;

% Initialise the oscillator.
ukp = ones(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: u_k
uk = ones(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: u'_k

% Initialise RLS parameters.
rlsR1 = 10 * ones(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: r1_k
rlsR4 = 10 * ones(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: r4_k
rlsA = zeros(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: b^_k
rlsB = zeros(1, NUM_OF_HARMONICS_TO_REMOVE); % Ref: c^_k

% IIR bandpass.
% Phase distortion doesn't matter here as we are only using this signal for
% frequency estimation.
filterCutOffs = DEFAULT_BANDPASS;

% Design the filter and convert to SOS format. Note that filter order is
% divided by 2 because butter designs bandpass filters of order 2n.
[z, p, k] = butter(FILTER_ORDER / 2, filterCutOffs / (fs / 2), 'bandpass');
sosParams = real(zp2sos(z, p, k));

% Filter and differentiate the signal. [Eq 4]
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
    for jHarmonic = 1 : NUM_OF_HARMONICS_TO_REMOVE
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
