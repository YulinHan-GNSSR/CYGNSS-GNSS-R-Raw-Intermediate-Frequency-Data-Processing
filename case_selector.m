%% project: CYGNSS-GNSS-R-Raw-Intermediate-Frequency-Data-Processing
%
% Based on the original GNSS-R raw IF MATLAB processing scripts
% by Santiago, with modifications for this repository.
%
% Script that defines the names of the binary and netCDF files to read the
% Level 1 Raw Intermediate Frequency Data Record containing the signal
% samples and the Level 1 Science Data Record used to load necessary
% metadata.
%
% To process new registers copy them to the files directory (folder_path)
% and add a new case in the switch below initializing the variables "file",
% "L1MetaFilename", "channel" and "ddmID".
% Additionaly, if present, it loads the .mat file containing the central
% delay estimation. It is used to retrack the processed DDMs.

%% File names initialization for each case

% Each case is defined by its dataset (bin and meta, same date!) and ddmID.
% For each case the preferred channel must be defined (0: zenith antenna, 
% 1: starboard nadir antenna, 2: port nadir antenna, if present).

switch caseID

    case 0 % 

        file = char("cyg04_raw_if_s20250112_092106_e20250112_092136");
        L1meta_filename = folder_path+"cyg04.ddmi.s20250112-000000-e20250112-235959.l1.power-brcs.a32.d33.nc";
        ddmID = 2; channel = 1;
        
end

bin_filename = folder_path + sprintf("%s_data.bin",file);
delay_filename = folder_path + sprintf("%s_ch%i_ddm%i_delay.mat",file,channel,ddmID);

%% Read necessary metadata for processing

% First we read the DRT0 packet in the header of the bin file. The
% important values are stored in the struct DRT0_vars, which are: starting
% time of the register in GPS weeks and seconds, data format as detailed in
% the report "CYGNSS Raw IF Data File Format", sample rate, and local
% oscillator frequencies for each channel
format = 'uint8';
fileID = fopen(bin_filename,'r');
if fileID == -1, error('Cannot open file: %s', bin_filename); end
DRT0_vars = DRT0packetRead(fileID, format); % function that reads the DRT0 packet in the bin file
fs = DRT0_vars.SampleRate;                  % sample rate
num_channels = DRT0_vars.DataFormat + 1;    % number of channels
fOL = DRT0_vars.CH0LOFreq;                  % LO frequency


% Then we read the netCDF file with information about the DDMs processed on
% board during the time interval of the bin file. We use this information
% to know which are the PRNs of the present reflections.
% If plot_tracks flag is 1, it plots the tracks of the 4 DDMs processed on
% board, with the antenna receiver gain value for that reflection
% represented by their color as reported in the metadata.
if isfile(L1meta_filename)
    meta = L1metaRead(L1meta_filename, DRT0_vars, time_to_process, ddmID, plot_tracks);
else
    error("netCDF file not found:\n\n'%s' missing",L1meta_filename)
end



%% Load central delay file if present

if isfile(delay_filename)
    load(delay_filename)
else
    delay_central0 = zeros(time_to_process,1);
    t0 = 0:time_to_process-1;
end

%% Interpolation of the central delay and Doppler to get a value every Ti
if time_to_process ~= 1
    t1 = meta.ddm_timestamp - meta.ddm_timestamp(1);
    t2 = (0:Ti:time_to_process-Ti);

    linear_param = polyfit(t1,meta.Doppler_central,1);
    Doppler_central = polyval(linear_param, t2);
    linear_param = polyfit(t0, delay_central0, 1);
    delay_central = polyval(linear_param,t2);
else
    Doppler_central = meta.Doppler_central;
end
