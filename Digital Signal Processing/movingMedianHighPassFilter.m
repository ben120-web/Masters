function [signalFiltered] = movingMedianHighPassFilter(signal, sampleFrequency, windowDuration)
% movingMedianHighPassFilter - This function performs moving
% median smoothing on the raw ECG signal, then removes the smoothed
% signal from the raw signal. This method is intended to remove
% low frequency baseline drift from the raw signal.
%
% Syntax: [signalFiltered] = movingMedianHighPassFilter(signal,...
%             sampleFrequency, windowDuration)
%
% Inputs:
%    signal - Input ECG waveform.
%    sampleFrequency - The sampling frequency of the ECG. [Hz]
%    windowDuration - Smoothing window duration. [s]
%
% Outputs:
%    signalFiltered - A filtered ECG waveform.
%
% Example:
%    signalFiltered = movingMedianHighPassFilter(ecg, 500, 0.3)
%
% Other m-files required: none.
% Subfunctions: none.
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
% All content is copyright B-Secur Ltd. 2019-2022.
% ====================================================================

%------------- BEGIN CODE ---------------

% Check number of input arguments.
narginchk(3, 3);

% Validate inputs.
validateattributes(signal, {'numeric'}, {'vector'}, mfilename);
validateattributes(sampleFrequency, {'numeric'}, {'scalar', 'nonempty', 'positive'}, mfilename);
validateattributes(windowDuration, {'numeric'}, {'scalar', 'nonempty', 'positive'}, mfilename);

% Convert window duration from seconds to samples.
windowLength = nextOddInteger(round(windowDuration * sampleFrequency));

% Apply smoothing and remove.
isolatedBaseline = movmedian(signal, windowLength);
signalFiltered = signal - isolatedBaseline;

end

%------------- END OF CODE --------------