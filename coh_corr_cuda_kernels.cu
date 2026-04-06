#include "coh_corr_cuda_kernels.h"

#include <math_constants.h>

namespace {

__global__ void dopplerCompKernel(
    const cufftComplex* signal_d,
    const float* doppler_d, // [Lf x K] column-major as MATLAB
    int N,
    int K,
    int Lf,
    int d_start,
    int nb,
    float fs_dec,
    cufftComplex* xp_d // [N x K x nb], vector-major layout: ((b*K + k)*N + n)
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (n >= N || k >= K || b >= nb) {
        return;
    }

    const int d = d_start + b;
    const float fd = doppler_d[d + k * Lf];
    const float phase = -2.0f * CUDART_PI_F * fd * (static_cast<float>(n) / fs_dec);
    float s, c;
    sincosf(phase, &s, &c);

    const cufftComplex x = signal_d[n + k * N];
    const cufftComplex p = make_cuFloatComplex(c, s);

    const cufftComplex out = make_cuFloatComplex(
        x.x * p.x - x.y * p.y,
        x.x * p.y + x.y * p.x
    );

    xp_d[(b * K + k) * N + n] = out;
}

__global__ void freqMultiplyKernel(
    cufftComplex* Xf_d,
    const cufftComplex* Cf_conj_d, // [N x K], column-major per segment
    int N,
    int K,
    int nb
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (n >= N || k >= K || b >= nb) {
        return;
    }

    const cufftComplex c = Cf_conj_d[n + k * N];
    cufftComplex& x = Xf_d[(b * K + k) * N + n];

    const float xr = x.x;
    const float xi = x.y;
    x.x = xr * c.x - xi * c.y;
    x.y = xr * c.y + xi * c.x;
}

__global__ void powerAccumulateKernel(
    const cufftComplex* y_ifft_d,
    int N,
    int K,
    int Lt,
    int d_start,
    int nb,
    float* ddm_accum_d // [Lt x Lf], column-major in MATLAB output
) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (l >= Lt || k >= K || b >= nb) {
        return;
    }

    const cufftComplex y = y_ifft_d[(b * K + k) * N + l];
    //const float p = (y.x * y.x + y.y * y.y) / static_cast<float>(N);
    const float p = (y.x * y.x + y.y * y.y) / (static_cast<float>(N) * static_cast<float>(N) * static_cast<float>(N));
    const int d = d_start + b;

    atomicAdd(&ddm_accum_d[l + d * Lt], p / static_cast<float>(K));
}

__global__ void conjugateKernel(cufftComplex* data_d, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    data_d[i].y = -data_d[i].y;
}

} // namespace

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
    cudaStream_t stream) {
    const dim3 block(16, 8, 1);
    const dim3 grid((N + block.x - 1) / block.x, (K + block.y - 1) / block.y, nb);
    dopplerCompKernel<<<grid, block, 0, stream>>>(
        signal_d, doppler_d, N, K, Lf, d_start, nb, fs_dec, xp_d);
}

void launchFreqMultiplyKernel(
    cufftComplex* Xf_d,
    const cufftComplex* Cf_conj_d,
    int N,
    int K,
    int nb,
    cudaStream_t stream) {
    const dim3 block(16, 8, 1);
    const dim3 grid((N + block.x - 1) / block.x, (K + block.y - 1) / block.y, nb);
    freqMultiplyKernel<<<grid, block, 0, stream>>>(Xf_d, Cf_conj_d, N, K, nb);
}

void launchPowerAccumulateKernel(
    const cufftComplex* y_ifft_d,
    int N,
    int K,
    int Lt,
    int d_start,
    int nb,
    float* ddm_accum_d,
    cudaStream_t stream) {
    const dim3 block(16, 8, 1);
    const dim3 grid((Lt + block.x - 1) / block.x, (K + block.y - 1) / block.y, nb);
    powerAccumulateKernel<<<grid, block, 0, stream>>>(
        y_ifft_d, N, K, Lt, d_start, nb, ddm_accum_d);
}

void launchConjugateKernel(cufftComplex* data_d, int count, cudaStream_t stream) {
    const int block = 256;
    const int grid = (count + block - 1) / block;
    conjugateKernel<<<grid, block, 0, stream>>>(data_d, count);
}
