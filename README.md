# CYGNSS GNSS-R Raw Intermediate Frequency Data Processing

## MATLAB + CUDA MEX framework for efficient generation of delay-Doppler maps (DDMs) and waveforms from CYGNSS raw IF data.
This project extends the original MATLAB-based CYGNSS raw IF processing workflow with GPU acceleration for the computational bottleneck, making the reflected-signal processing chain more efficient, scalable, and suitable for larger data-processing experiments.

## Overview

This project processes CYGNSS raw IF binary data to generate a time series of delay-Doppler maps (DDMs) or waveforms. It also provides visualization of the reflection ground tracks and the estimated signal-to-noise ratio (SNR).

The input data consist of:

- a **CYGNSS Level 1 Raw Intermediate Frequency Data Record** (`*_data.bin`), which contains the sampled raw IF signal
- the corresponding **CYGNSS Level 1 Science Data Record** netCDF file (`*.nc`) from the same date and CYGNSS satellite

The netCDF file contains the DDMs processed on board during that day, together with relevant metadata such as PRN, central Doppler, timestamps, and geolocation-related variables. These metadata are read and used to guide the off-board DDM processing from the raw IF data.

The processing parameters are configurable, including:

- coherent integration time
- non-coherent integration time
- decimation ratio
- delay resolution
- Doppler resolution
- ...

## GPU Acceleration

In this version, the computationally intensive coherent correlation and DDM formation stage is accelerated using a CUDA MEX module.

MATLAB is used for:

- configuration
- metadata loading
- raw IF reading and decoding
- preprocessing, including downconversion and decimation
- visualization and timing

The GPU is used for:

- Doppler compensation
- frequency-domain correlation
- inverse FFT
- power accumulation for DDM generation

## Repository Structure

```text

.
├── main.m                       # Main processing script
├── config.m                     # Processing configuration
├── case_selector.m              # Dataset / case selection
├── DRT0packetRead.m             # Read DRT0 header from raw IF file
├── L1metaRead.m                 # Read required metadata from netCDF file
├── bin2int.m                    # Decode CYGNSS 2-bit interleaved IF samples
├── coarse_delay_tracking.m      # Optional residual delay correction
├── build_mex.m                  # Build CUDA MEX
├── coh_corr_cuda_ddm_mex.cu     # CUDA MEX entry point
├── coh_corr_cuda_kernels.cu     # CUDA kernels
├── coh_corr_cuda_kernels.h      # CUDA kernel declarations
├── cacode_gps_ca.cpp            # GPS L1 C/A code generator
├── cacode_gps_ca.h              # Header for C/A code generator
└── README.md
```

## Processing Flow

The overall processing flow is:

1. Select the dataset and processing case in config.m and case_selector.m
2. Read the raw IF file header using DRT0packetRead.m
3. Read the corresponding netCDF metadata using L1metaRead.m
4. Read and decode raw IF samples using bin2int.m
5. Downconvert and decimate each coherent integration block in MATLAB
6. Call coh_corr_cuda_ddm_mex to perform GPU-based correlation and DDM generation
7. Store, visualize, and analyze the resulting DDM / waveform sequence
8. Optionally run coarse_delay_tracking.m to estimate residual delay drift

## Notes and Limitations

- This project currently focuses on GPS L1 C/A reflected-signal processing from CYGNSS raw IF data.
- The input raw IF format is assumed to follow the CYGNSS 2-bit compressed interleaved sample format.
- Numerical differences may exist between CPU and GPU implementations because of floating-point precision and implementation details.
- CUDA MEX compilation may require environment-specific configuration depending on the MATLAB, CUDA Toolkit, compiler, and GPU driver versions.

## Relationship to the Original Code

This repository is based on the original CYGNSS raw IF MATLAB processing scripts developed by Santiago and published in the repository `GNSS-R_raw_data_processing`:

https://github.com/santiagooz/GNSS-R_raw_data_processing

This version has been modified and extended with GPU acceleration.

## The main changes in this repository include:

- replacing the original MATLAB coherent correlation stage with a CUDA MEX implementation
- adding GPU-based Doppler compensation
- adding GPU-based frequency-domain correlation
- adding GPU-based inverse FFT and DDM accumulation
- integrating GPU build support and acceleration-oriented workflow restructuring

In short, this repository mainly contributes an engineering-oriented GPU acceleration of the original CYGNSS raw IF processing chain.
