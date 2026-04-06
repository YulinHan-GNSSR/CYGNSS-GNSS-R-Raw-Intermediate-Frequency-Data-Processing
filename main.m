%% project: CYGNSS-GNSS-R-Raw-Intermediate-Frequency-Data-Processing
%
% This script:
%   1) loads processing parameters and metadata,
%   2) reads CYGNSS raw IF data batch by batch,
%   3) preprocesses the signal (downconversion + decimation),
%   4) computes DDMs/WFs using the CUDA MEX correlator,
%   5) visualizes the DDM/WF time series, reflection track, and SNR,

clear;
close all;
clc

config   % Load configuration parameters, file names, metadata, and axes

% Reset the persistent CUDA MEX context before a fresh run.
coh_corr_cuda_ddm_mex('reset');

%% DDM processing
tic
wb = waitbar(0, 'Processing data: 0% completed');

for batch = 1:num_ddm

    signal = zeros(ceil(N/dec_ratio), K);

    % Move the file pointer to the start position of the current batch.
    fseek(fileID, 35 + channel + bytes_offset*num_channels, -1);

    % Read one batch of raw IF data from the selected channel.
    [signal_IF, success] = bin2int(fileID, format, ...
        bytes_to_read + bytes_guard, samples_offset, num_channels);

    % Accumulated sample offset
    sum_offset = 0;

    if success == true
        % Pointer to the first sample of the current coherent segment
        % within signal_IF.
        pointer = 1;

        %% ---------- pre-processing ----------
        for kk = 1:K
            % Doppler search vector for the current coherent segment.
            Doppler(:, kk) = Doppler0 + ...
                Doppler_central((batch-1)*K + kk) + Doppler_offset;

            % Estimate the code-phase drift (in samples) during this
            % coherent integration interval due to the central Doppler.
            deltaN_Doppler = -Ti * ...
                (Doppler_central((batch-1)*K + kk) + Doppler_offset) ...
                / 1540 * fs / fc;

            % Extract the current coherent segment from the raw IF buffer.
            signal_aux = signal_IF((pointer:pointer+N-1) + floor(sum_offset));

            % Downconvert the IF signal to complex baseband and decimate it.
            % FIR-based decimation is used to suppress aliasing.
            signal(:,kk) = decimate( ...
                signal_aux .* exp(-1i*2*pi*fIF/fs*(0:N-1)), ...
                dec_ratio, 'fir');

            % Update the accumulated timing offset for the next segment.
            sum_offset = deltaN_sampling + deltaN_Doppler + sum_offset;

            % Advance to the next nominal coherent block.
            pointer = pointer + N;
        end

        %% ---------- correlation + ddm ----------
        % Central delay values used for the K coherent segments of
        % the current batch.
        delay_vec = delay_initial + ...
            delay_central((batch-1)*K + 1 : batch*K);

        % Get the active MATLAB GPU device object.
        g = gpuDevice;

        % Doppler chunk size used internally by the CUDA MEX function.
        % For the current example, the Doppler grid has 9 bins.
        doppler_chunk = 9;

        % Compute the DDM/WF for this batch using the CUDA MEX correlator.
        % Outputs:
        %   ddm_batch : DDM/WF power result for the current batch
        [ddm_batch, mexinfo] = coh_corr_cuda_ddm_mex( ...
            signal, fs/dec_ratio, meta.PRN, Doppler, ...
            delay0, Ti, delay_vec, doppler_chunk);

        %% ---------- assign output ----------
        if Lf > 1
            % Full delay-Doppler map
            ddm(:,:,batch) = ddm_batch;
        else
            % Single-Doppler waveform
            ddm(:,batch) = ddm_batch(:,1);
        end
    end

    %% Update offsets for the next batch
    offset = offset + pointer + sum_offset;
    bytes_offset = floor(offset/4);
    samples_offset = floor(mod(offset,4));

    %% Progress bar
    time_past = toc;
    time_per_batch = time_past / batch;
    time_left = (num_ddm - batch) * time_per_batch;
    time_left_h = floor(time_left / 3600);
    time_left_m = floor((time_left - time_left_h*3600) / 60);
    time_left_s = floor(time_left - time_left_h*3600 - time_left_m*60);

    msg = sprintf( ...
        'Processing data: %i%% completed - %i:%i:%i remaining', ...
        floor(batch/num_ddm*100), ...
        time_left_h, time_left_m, time_left_s);

    waitbar(batch/num_ddm, wb, msg)
end

close(wb)

%% Remove trailing zero DDMs/WFs if the record is shorter than requested
if Lf > 1
    while num_ddm > 1 && ~any(ddm(:,:,num_ddm), 'all')
        num_ddm = num_ddm - 1;
    end
else
    while num_ddm > 1 && ~any(ddm(:,num_ddm))
        num_ddm = num_ddm - 1;
    end
end

fclose(fileID);
clear signal signal_aux;

%% Plot DDM (or WF) time series

% Linearly interpolate the specular point (SP) ground track so that it
% matches the number of processed DDM batches.
a = linspace(0, 1, num_ddm);
dt1 = mode(diff(t1));
sp_lat_int = meta.sp_lat(1) + ...
    (meta.sp_lat(floor(num_ddm*K*Ti/dt1)) - meta.sp_lat(1)) * a;
sp_lon_int = meta.sp_lon(1) + ...
    (meta.sp_lon(floor(num_ddm*K*Ti/dt1)) - meta.sp_lon(1)) * a;

SNRdB = zeros(1, num_ddm);
SNRdBmax = 21;

fh = figure(102); clf;
fh.WindowState = 'maximized';

%%
for batch = 1:num_ddm
    % Force the plotting target to remain the same figure.
    % This helps avoid occasional figure switching bugs during updates.
    figure(fh);

    % Delay axis for the current batch
    delay = delay0 + delay_initial + delay_central((batch-1)*K + 1);

    if Lf > 1
        % Doppler axis for the current batch
        Doppler = Doppler0 + Doppler_central((batch-1)*K + 1) + Doppler_offset;

        % Use the time-averaged DDM to determine a representative peak
        % position for display and SNR estimation.
        ddm_mean = mean(ddm, 3);
        [delay_max, doppler_max] = find(ddm_mean == max(ddm_mean(:)));

        % Estimate the noise level from the delay bins sufficiently ahead
        % of the peak region.
        NL = mean(ddm(1:delay_max-30, :, batch), 'all');

        % Peak power of the current DDM
        ddm_max = max(ddm(:,:,batch), [], 'all');

        % SNR estimate in dB
        SNRdB(batch) = 10*log10(ddm_max / NL - 1);

        % ---- DDM plot ----
        subplot(2,2,1)
        colormap(subplot(2,2,1), parula)
        surf(Doppler*1e-3, delay, ddm(:,:,batch)); shading interp
        view([90 90])
        axis([Doppler(1)*1e-3, Doppler(end)*1e-3, ...
              delay(delay_max)-20, delay(delay_max)+20, ...
              min(ddm(:)), max(ddm(:,:,batch),[],'all')*1.1])
        ylabel('delay [chips]');
        xlabel('Doppler [kHz]');
        title(sprintf('DDM num %i - SNR = %.2g dB', batch, SNRdB(batch)))

        % ---- Waveform cut at peak Doppler ----
        subplot(2,2,3)
        hold off
        plot(delay, ddm(:,doppler_max,batch), 'LineWidth', 2);
        hold on
        plot(delay, NL*ones(size(delay)), '--r');
        plot(delay, ddm_max*ones(size(delay)), '--g');
        axis([delay(delay_max)-20, delay(delay_max)+20, ...
              min(ddm(:)), max(ddm(:,:,batch),[],'all')*1.1])
        xlabel('delay [chips]');
        grid on
        title(sprintf('WF num %i - SNR = %.2g dB', batch, SNRdB(batch)))

    else
        % Waveform-only case (single Doppler bin)
        ddm_mean = mean(ddm,2);
        delay_max = find(ddm_mean == max(ddm_mean(:)));

        % Noise level estimation from the pre-peak region
        NL = mean(ddm(1:delay_max-30, batch));

        % Peak value and index
        [ddm_max, max_idx] = max(ddm(:,batch));

        % SNR estimate in dB
        SNRdB(batch) = 10*log10(ddm_max / NL - 1);

        subplot(2,2,1)
        plot(delay, ddm(:,batch), 'LineWidth', 2);
        axis([delay(delay_max)-20, delay(delay_max)+20, ...
              min(ddm(:)), max(ddm(:,batch))*1.1])
        grid on
        hold on
        stem(delay(max_idx), ddm(max_idx,batch));
        plot(delay, NL*ones(size(delay)), '--r');
        plot(delay, ddm_max*ones(size(delay)), '--g');
        hold off
        xlabel('delay [chips]');
        title(sprintf('WF num %i - SNR = %.2g dB', batch, SNRdB(batch)))
    end

    % ---- Reflection ground track ----
    subplot(2,2,2);
    title('Reflection ground track')
    geoscatter(sp_lat_int(batch), sp_lon_int(batch), 36, SNRdB(batch), 'Filled');
    colormap(subplot(2,2,2), autumn)
    cb = colorbar; cb.Label.String = 'SNR [dB]';
    geolimits([min(sp_lat_int)-.1, max(sp_lat_int)+.1], ...
              [min(sp_lon_int)-.1, max(sp_lon_int)+.1])
    hold on
    geobasemap topographic

    % ---- SNR time series ----
    subplot(2,2,4);
    plot((0:batch-2)*K*Ti, SNRdB(1:batch-1), '*', ...
        'Color', [0, 0, 155]/256);
    grid on; hold on

    plot((batch-1)*K*Ti, SNRdB(batch), '*', ...
        'Color', [200, 50, 0]/256);
    grid on; hold on

    xlabel('t [s]');
    ylabel('SNR [dB]');
    title('Estimated signal to noise ratio')
    axis([0, (num_ddm-1)*K*Ti, -10, SNRdBmax])

    pause(.005)
end
