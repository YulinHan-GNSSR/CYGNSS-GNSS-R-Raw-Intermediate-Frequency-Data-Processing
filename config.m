%% project: GNSS-R_raw_data_processing
% Configuration script for CYGNSS raw IF data processing.
% Based on the original GNSS-R raw IF MATLAB processing scripts
% by Santiago, with modifications for this repository.

%% Input parameters

time_to_process = 5;   % Total processing time [s]
caseID = 13;           % Case ID defined in case_selector.m

% Processing parameters
Ti = 1e-3;                 % Coherent integration time [s]
Tnc = 0.1;                 % Non-coherent integration time [s]
dec_ratio = 4;             % Decimation ratio
Doppler_resolution = 500;  % Doppler bin resolution [Hz]
Doppler_spread = 2e3;      % Maximum Doppler deviation [Hz]
Doppler_offset = 0;        % Optional Doppler offset [Hz]

folder_path = ".\files\"; % Input data folder
plot_tracks = 1;           % Plot reflection ground tracks from metadata

%% Load selected case

case_selector

% Constants
fc  = 1.023e6;     % GPS L1 C/A code rate [Hz]
fL1 = 1575.42e6;   % GPS L1 carrier frequency [Hz]
fIF = fL1 - fOL;   % Intermediate frequency [Hz]

%% Delay bins

delay_resolution = dec_ratio / fs * fc;   % Delay resolution [chips]
delay_initial = 0;                        % Initial delay [chips]
delay0 = (0 : delay_resolution : Ti * fc - delay_resolution);   % delay axis starting in 0 chips
Lt = length(delay0);                      % Number of delay bins

%% Doppler bins

Doppler0 = -Doppler_spread : Doppler_resolution : Doppler_spread;   % Doppler axis centered in 0 Hz
Lf = length(Doppler0);                    % Number of Doppler bins

%% Integration parameters

K = floor(Tnc / Ti);                      % Coherent blocks per DDM
num_ddm = floor(time_to_process / Ti / K);% Number of DDMs

%% Read parameters

N = floor(Ti * fs);                       % Samples per coherent integration
bytes_to_read = ceil(K * Ti * fs / 4);    % Bytes per DDM batch (2-bit data)
deltaN_sampling = Ti * fs - N;            % Sample rounding error

bytes_offset = 0;
samples_offset = 0;
offset = 0;

% Extra bytes to avoid sample shortage due to Doppler-induced code drift
bytes_guard = ceil(abs(min(Doppler_central)) * Ti / 1540 * fs / fc * K / 4) + 1;

%% Initialization

Doppler = zeros(Lf, K);                   % Doppler grid for one DDM batch
ddm = zeros(Lt, Lf, num_ddm);             % Output DDM array
