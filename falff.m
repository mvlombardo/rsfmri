function result = falff(data, samplingRate, lowFreqBand, allFreqBand)
%   fALFF - Fractional Amplitude of Low Frequency Fluctuations
%
%   INPUT
%
%   data = 1:nTimePoints vector
%   samplingRate = sampling rate in Hz (for imaging data 1/TR)
%   lowFreqBand = range of low frequencies of interest (e.g., [0.01, 0.03])
%   allFreqBand = full range of frequencies of interest (e.g., [0.01, 0.1])
%

%%
nTimePoints = size(data,2); % number of timepoints

% mean center data
mean_data = nanmean(data);
mc_data = data - mean_data;

% run fft
xdft = fft(mc_data);
xdft = xdft(1:nTimePoints/2+1);
powSpecDens(1,:) = (1/(samplingRate*nTimePoints)) * abs(xdft).^2;
powSpecDens(1,2:end-1) = 2*powSpecDens(1,2:end-1);
freqs = 0:samplingRate/nTimePoints:samplingRate/2;

pband1 = bandpower(powSpecDens, freqs, lowFreqBand, 'psd');
ptot1 = bandpower(powSpecDens, freqs, allFreqBand, 'psd');

% fALFF - fraction of power in low frequencies relative to total power
% across all frequencies of interest
result = pband1./ptot1;

end % function falff
