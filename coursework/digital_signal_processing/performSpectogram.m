%% This script will perform time-frequency analysis on a noisy ECG
%% Signal, and identify the frequency range of the noisy components

%% Part 1 - Identify the Frequency components of the noisy signals.

% Load Test Signal
[testSignal, Fs] = audioread('/Users/benrussell/Documents/Digital Signal Processing Assignment/signals/signal23.wav');
timeSeries = 1/Fs : 1 / Fs : numel(testSignal)/Fs;

% Plot time domain signal
figure(1);

plot(timeSeries, testSignal);
xlabel("Time (Seconds)");
ylabel("Voltage (V)");
title('Time Domain Representation of ECG signal');


AZ = 0;                 % azimuth
EL = 90;                % elevation
TH = -100;              % amplitude threshold (dB)
Ns = length(testSignal);       
dur = Ns/Fs;

% Perform the short-time Fourier Transform.
[S, F, T, ~] = spectrogram(testSignal, blackman(512), 480, 512, Fs);

% Set the absolute values of the spectrogram.
A = (abs(S));

% Find the max value
Amax = max(max(A));

% Devide A by AMax
A = A/Amax;

% Set AL as the log of A
AL = 20*log10(A);

% Find values of AL that are less that our threshold
I = find(AL < TH);

% Set these values as being our threshold
AL(I) = TH;

% PLot
figure(2);
xlabel('time [s]'); ylabel('signal');
surf(T,0.001*F*1000,AL,'edgecolor','none'); 
grid;
axis tight;
xlabel('time [s]'); ylabel('frequency [Hz]');
box on;
set(gca,'BoxStyle','full');
set(gca,'XAxisLocation','origin');
view(AZ,EL);

%% Part 3 - Design a filter that removes noisy segments

% Call our notch filter to remove the Powerline Noise
processedSignal = adaptiveNotchFilter(testSignal, Fs);

% Plot the spectrogram to validate that mains noise has been removed. 
% Perform the short-time Fourier Transform.
[S, F, T, ~] = spectrogram(processedSignal, blackman(512), 480, 512, Fs);

% Set the absolute values of the spectrogram.
A = (abs(S));

% Find the max value
Amax = max(max(A));

% Devide A by AMax
A = A/Amax;

% Set AL as the log of A
AL = 20*log10(A);

% Find values of AL that are less that our threshold
I = find(AL < TH);

% Set these values as being our threshold
AL(I) = TH;

% PLot
figure(3);
xlabel('time [s]'); ylabel('signal');
surf(T,0.001*F*1000,AL,'edgecolor','none'); 
grid;
axis tight;
xlabel('time [s]'); ylabel('frequency [Hz]');
title("Validate mains noise removed")
box on;
set(gca,'BoxStyle','full');
set(gca,'XAxisLocation','origin');
view(AZ,EL); %% Mains nose removed.

% Now we need to remove the hf nosie above 170Hz from the signal
finalFilteredSignal = removeHFnoise(processedSignal, Fs);

% Plot final spectrogram to validate all noise is gone.
% Perform the short-time Fourier Transform.
[S, F, T, P] = spectrogram(finalFilteredSignal, blackman(512), 480, 512, Fs);

% Set the absolute values of the spectrogram.
A = (abs(S));

% Find the max value
Amax = max(max(A));

% Devide A by AMax
A = A/Amax;

% Set AL as the log of A
AL = 20*log10(A);

% Find values of AL that are less that our threshold
I = find(AL < TH);

% Set these values as being our threshold
AL(I) = TH;

% PLot
figure(4);
xlabel('time [s]'); ylabel('signal');
surf(T,0.001*F*1000,AL,'edgecolor','none'); 
grid;
axis tight;
xlabel('time [s]'); ylabel('frequency [Hz]');
title("Validate HF noise removed")
box on;
set(gca,'BoxStyle','full');
set(gca,'XAxisLocation','origin');
view(AZ,EL); %% Mains nose removed.

figure(5);
plot(finalFilteredSignal, 'm');

%% SUB FUNCTIONS
function filteredSignal = removeHFnoise(signal, fs)
% This function will remove the HF noise seen in the signal.

%% CONSTANTS
CUT_OFF = 120;

y = lowpass(signal, CUT_OFF, fs);

filteredSignal = filter(1, 1, y);

freqz(y)

end

function processedSignal = adaptiveNotchFilter(signal, fs)
% This function will perform adaptive notch filtering as detailed in
% [1].
%
% The four stages are:
% 1. An adaptive notch filter is used to estimate the fundamental frequency
%    of the noise. If this is known, this can also be input.
% 2. Based on the estimated frequency, harmonics are generated using
%    discrete-time oscillators.
% 3. The amplitude and phase of each harmonic are then estimated using a
%    modified recursive least squares algorithm.
% 4. The estimated interference is subtracted from the signal.

DEFAULT_BANDPASS = [45, 55]; % [Hz]
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

% Plot amplitude response
figure();
freqz(z, p);
title("Bandpass Response");

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

