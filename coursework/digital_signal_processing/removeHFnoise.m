function filteredSignal = removeHFnoise(signal, fs)
% This function will remove the HF noise seen in the signal.

%% CONSTANTS
CUT_OFF = 190;

y = lowpass(signal, CUT_OFF, fs);

filteredSignal = filter(1, 1, y);

end 

