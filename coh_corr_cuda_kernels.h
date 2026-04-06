#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

void launchDopplerCompKernel(
    const cufftComplex* signal_d,
    const float* doppler_d,
    int N,
    int K,
    int Lf,
    int d_start,
    int nb,
    float fs_dec,
    cufftComplex* xp_d,
    cudaStream_t stream);

void launchFreqMultiplyKernel(
    cufftComplex* Xf_d,
    const cufftComplex* Cf_conj_d,
    int N,
    int K,
    int nb,
    cudaStream_t stream);

void launchPowerAccumulateKernel(
    const cufftComplex* y_ifft_d,
    int N,
    int K,
    int Lt,
    int d_start,
    int nb,
    float* ddm_accum_d,
    cudaStream_t stream);

void launchConjugateKernel(
    cufftComplex* data_d,
    int count,
    cudaStream_t stream);
