#include "mex.h"
#include "matrix.h"

#include "cacode_gps_ca.h"
#include "coh_corr_cuda_kernels.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr float kFc = 1.023e6f;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (call);                                                       \
        if (_err != cudaSuccess) {                                                       \
            mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:CUDA", "%s failed: %s", #call,     \
                              cudaGetErrorString(_err));                                 \
        }                                                                                \
    } while (0)

#define CUFFT_CHECK(call)                                                                \
    do {                                                                                 \
        cufftResult _err = (call);                                                       \
        if (_err != CUFFT_SUCCESS) {                                                     \
            mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:CUFFT", "%s failed (code %d)",     \
                              #call, static_cast<int>(_err));                            \
        }                                                                                \
    } while (0)

struct PersistentContext {
    bool initialized = false;
    bool at_exit_registered = false;
    bool mex_locked = false;
    bool cf_valid = false;

    int N = 0;
    int K = 0;
    int Lt = 0;
    int Lf = 0;
    int max_nb = 0;
    int prn = 0;

    float fs_dec = 0.0f;
    float Ti = 0.0f;

    cufftComplex* signal_d = nullptr;   // [N x K]
    float* doppler_d = nullptr;         // [Lf x K]
    cufftComplex* Cf_conj_d = nullptr;  // [N x K]
    cufftComplex* work_d = nullptr;     // [N x K x max_nb]
    float* ddm_d = nullptr;             // [Lt x Lf]

    cufftHandle plan_code = 0;      // batch = K
    cufftHandle plan_fwd_full = 0;  // batch = K * max_nb
    cufftHandle plan_inv_full = 0;  // batch = K * max_nb

    cufftHandle plan_fwd_tail = 0;  // optional, only for last smaller chunk
    cufftHandle plan_inv_tail = 0;  // optional
    int tail_nb = 0;                // nb for tail plan

    std::array<float, 1023> ca_code{};
    std::vector<float> last_delay_vec;
    std::vector<cufftComplex> code_h_cache; // [N x K] host reusable buffer
};

static PersistentContext g_ctx;

double getScalarDouble(const mxArray* a, const char* name) {
    if (!mxIsNumeric(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "%s must be a real numeric scalar.", name);
    }
    return mxGetScalar(a);
}

int getScalarInt(const mxArray* a, const char* name) {
    const double v = getScalarDouble(a, name);
    if (std::isnan(v) || std::isinf(v) || std::floor(v) != v) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "%s must be an integer scalar.", name);
    }
    return static_cast<int>(v);
}

std::vector<float> toFloatRealVector(const mxArray* a, const char* name) {
    if (!mxIsNumeric(a) || mxIsComplex(a)) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "%s must be a real numeric array.", name);
    }
    const mwSize n = mxGetNumberOfElements(a);
    std::vector<float> out(n);
    if (mxIsSingle(a)) {
        const float* p = static_cast<const float*>(mxGetData(a));
        std::copy(p, p + n, out.begin());
    } else {
        const double* p = mxGetPr(a);
        for (mwSize i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    }
    return out;
}

std::vector<cufftComplex> toComplexFloatMatrix(const mxArray* a, int& rows, int& cols) {
    if (!mxIsNumeric(a) || !mxIsComplex(a)) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "signal must be a complex numeric matrix [N x K].");
    }
    if (mxGetNumberOfDimensions(a) != 2) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "signal must be 2-D [N x K].");
    }

    rows = static_cast<int>(mxGetM(a));
    cols = static_cast<int>(mxGetN(a));
    const mwSize n = static_cast<mwSize>(rows) * static_cast<mwSize>(cols);
    std::vector<cufftComplex> out(n);

#if MX_HAS_INTERLEAVED_COMPLEX
    if (mxIsSingle(a)) {
        const mxComplexSingle* p = mxGetComplexSingles(a);
        for (mwSize i = 0; i < n; ++i) {
            out[i] = make_cuFloatComplex(p[i].real, p[i].imag);
        }
    } else {
        const mxComplexDouble* p = mxGetComplexDoubles(a);
        for (mwSize i = 0; i < n; ++i) {
            out[i] = make_cuFloatComplex(static_cast<float>(p[i].real),
                                         static_cast<float>(p[i].imag));
        }
    }
#else
    const double* pr = mxGetPr(a);
    const double* pi = mxGetPi(a);
    for (mwSize i = 0; i < n; ++i) {
        const float re = static_cast<float>(pr[i]);
        const float im = pi ? static_cast<float>(pi[i]) : 0.0f;
        out[i] = make_cuFloatComplex(re, im);
    }
#endif
    return out;
}

bool sameFloatVector(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void destroyHandleIfValid(cufftHandle& h) {
    if (h) {
        cufftDestroy(h);
        h = 0;
    }
}

void destroyContext(PersistentContext& ctx) {
    destroyHandleIfValid(ctx.plan_code);
    destroyHandleIfValid(ctx.plan_fwd_full);
    destroyHandleIfValid(ctx.plan_inv_full);
    destroyHandleIfValid(ctx.plan_fwd_tail);
    destroyHandleIfValid(ctx.plan_inv_tail);
    ctx.tail_nb = 0;

    if (ctx.signal_d)   { cudaFree(ctx.signal_d);   ctx.signal_d = nullptr; }
    if (ctx.doppler_d)  { cudaFree(ctx.doppler_d);  ctx.doppler_d = nullptr; }
    if (ctx.Cf_conj_d)  { cudaFree(ctx.Cf_conj_d);  ctx.Cf_conj_d = nullptr; }
    if (ctx.work_d)     { cudaFree(ctx.work_d);     ctx.work_d = nullptr; }
    if (ctx.ddm_d)      { cudaFree(ctx.ddm_d);      ctx.ddm_d = nullptr; }

    ctx.last_delay_vec.clear();
    ctx.code_h_cache.clear();
    ctx.cf_valid = false;

    ctx.N = 0;
    ctx.K = 0;
    ctx.Lt = 0;
    ctx.Lf = 0;
    ctx.max_nb = 0;
    ctx.prn = 0;
    ctx.fs_dec = 0.0f;
    ctx.Ti = 0.0f;
    ctx.initialized = false;

    if (ctx.mex_locked) {
        mexUnlock();
        ctx.mex_locked = false;
    }
}

void atExitCleanup() {
    destroyContext(g_ctx);
}

bool needsRebuild(const PersistentContext& ctx,
                  int N, int K, int Lt, int Lf, int max_nb,
                  int prn, float fs_dec, float Ti) {
    if (!ctx.initialized) return true;
    if (ctx.N != N) return true;
    if (ctx.K != K) return true;
    if (ctx.Lt != Lt) return true;
    if (ctx.Lf != Lf) return true;
    if (ctx.max_nb != max_nb) return true;
    if (ctx.prn != prn) return true;
    if (ctx.fs_dec != fs_dec) return true;
    if (ctx.Ti != Ti) return true;
    return false;
}

void createPlanCode(PersistentContext& ctx) {
    destroyHandleIfValid(ctx.plan_code);

    int rank = 1;
    int nfft[1] = {ctx.N};
    int inembed[1] = {ctx.N};
    int onembed[1] = {ctx.N};
    int istride = 1;
    int ostride = 1;
    int idist = ctx.N;
    int odist = ctx.N;
    int batch = ctx.K;

    CUFFT_CHECK(cufftPlanMany(&ctx.plan_code, rank, nfft,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch));
}

void createFullPlans(PersistentContext& ctx) {
    destroyHandleIfValid(ctx.plan_fwd_full);
    destroyHandleIfValid(ctx.plan_inv_full);

    int rank = 1;
    int nfft[1] = {ctx.N};
    int inembed[1] = {ctx.N};
    int onembed[1] = {ctx.N};
    int istride = 1;
    int ostride = 1;
    int idist = ctx.N;
    int odist = ctx.N;
    int batch = ctx.K * ctx.max_nb;

    CUFFT_CHECK(cufftPlanMany(&ctx.plan_fwd_full, rank, nfft,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch));

    CUFFT_CHECK(cufftPlanMany(&ctx.plan_inv_full, rank, nfft,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch));
}

void createTailPlans(PersistentContext& ctx, int nb) {
    if (nb <= 0 || nb >= ctx.max_nb) return;
    if (ctx.tail_nb == nb && ctx.plan_fwd_tail && ctx.plan_inv_tail) return;

    destroyHandleIfValid(ctx.plan_fwd_tail);
    destroyHandleIfValid(ctx.plan_inv_tail);
    ctx.tail_nb = nb;

    int rank = 1;
    int nfft[1] = {ctx.N};
    int inembed[1] = {ctx.N};
    int onembed[1] = {ctx.N};
    int istride = 1;
    int ostride = 1;
    int idist = ctx.N;
    int odist = ctx.N;
    int batch = ctx.K * nb;

    CUFFT_CHECK(cufftPlanMany(&ctx.plan_fwd_tail, rank, nfft,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch));

    CUFFT_CHECK(cufftPlanMany(&ctx.plan_inv_tail, rank, nfft,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, batch));
}

void initContext(PersistentContext& ctx,
                 int N, int K, int Lt, int Lf, int max_nb,
                 int prn, float fs_dec, float Ti) {
    destroyContext(ctx);

    ctx.N = N;
    ctx.K = K;
    ctx.Lt = Lt;
    ctx.Lf = Lf;
    ctx.max_nb = max_nb;
    ctx.prn = prn;
    ctx.fs_dec = fs_dec;
    ctx.Ti = Ti;

    if (!generateGpsCaCode(prn, ctx.ca_code)) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Unsupported PRN=%d (expected 1..32).", prn);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.signal_d),
                          sizeof(cufftComplex) * static_cast<size_t>(N) * static_cast<size_t>(K)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.doppler_d),
                          sizeof(float) * static_cast<size_t>(Lf) * static_cast<size_t>(K)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.Cf_conj_d),
                          sizeof(cufftComplex) * static_cast<size_t>(N) * static_cast<size_t>(K)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.work_d),
                          sizeof(cufftComplex) * static_cast<size_t>(N) * static_cast<size_t>(K) * static_cast<size_t>(max_nb)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.ddm_d),
                          sizeof(float) * static_cast<size_t>(Lt) * static_cast<size_t>(Lf)));

    ctx.code_h_cache.resize(static_cast<size_t>(N) * static_cast<size_t>(K));

    createPlanCode(ctx);
    createFullPlans(ctx);

    ctx.cf_valid = false;
    ctx.initialized = true;

    if (!ctx.at_exit_registered) {
        mexAtExit(atExitCleanup);
        ctx.at_exit_registered = true;
    }
    if (!ctx.mex_locked) {
        mexLock();
        ctx.mex_locked = true;
    }
}

void ensureContext(PersistentContext& ctx,
                   int N, int K, int Lt, int Lf, int max_nb,
                   int prn, float fs_dec, float Ti) {
    if (needsRebuild(ctx, N, K, Lt, Lf, max_nb, prn, fs_dec, Ti)) {
        initContext(ctx, N, K, Lt, Lf, max_nb, prn, fs_dec, Ti);
    }
}

void updateCodeReplicaIfNeeded(PersistentContext& ctx, const std::vector<float>& delay_vec_h) {
    if (static_cast<int>(delay_vec_h.size()) != ctx.K) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "length(delay_vec) must equal K.");
    }

    const bool changed = (!ctx.cf_valid) || (!sameFloatVector(delay_vec_h, ctx.last_delay_vec));
    if (!changed) return;

    // Build the local C/A code replica on the host.
    for (int k = 0; k < ctx.K; ++k) {
        for (int n = 0; n < ctx.N; ++n) {
            const float chip = std::floor((static_cast<float>(n) / ctx.fs_dec) * kFc - delay_vec_h[k]);
            int idx = static_cast<int>(chip) % 1023;
            if (idx < 0) idx += 1023;

            const float c = ctx.ca_code[static_cast<size_t>(idx)];
            ctx.code_h_cache[static_cast<size_t>(n) +
                             static_cast<size_t>(k) * static_cast<size_t>(ctx.N)] =
                make_cuFloatComplex(c, 0.0f);
        }
    }

    CUDA_CHECK(cudaMemcpy(ctx.Cf_conj_d, ctx.code_h_cache.data(),
                          sizeof(cufftComplex) * static_cast<size_t>(ctx.N) * static_cast<size_t>(ctx.K),
                          cudaMemcpyHostToDevice));

    // Compute conj(FFT(C)) once and cache it on the device.
    CUFFT_CHECK(cufftExecC2C(ctx.plan_code, ctx.Cf_conj_d, ctx.Cf_conj_d, CUFFT_FORWARD));
    launchConjugateKernel(ctx.Cf_conj_d, ctx.N * ctx.K, 0);
    CUDA_CHECK(cudaGetLastError());

    ctx.last_delay_vec = delay_vec_h;
    ctx.cf_valid = true;
}

std::string getCommandString(const mxArray* a) {
    if (!mxIsChar(a)) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Command input must be a char array.");
    }
    char* c = mxArrayToString(a);
    if (!c) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Failed to parse command string.");
    }
    std::string out(c);
    mxFree(c);
    return out;
}

void handleCommandMode(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input", "Invalid command mode.");
    }

    const std::string cmd = getCommandString(prhs[0]);

    if (cmd == "reset") {
        if (nlhs > 0) {
            mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Output",
                              "'reset' returns no outputs.");
        }
        destroyContext(g_ctx);
        return;
    }

    mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                      "Unknown command '%s'. Supported: 'reset'.",
                      cmd.c_str());
}

} // namespace

// MATLAB signatures:
// ddm_batch = coh_corr_cuda_ddm_mex(signal, fs_dec, PRN, Doppler, delay0, Ti, delay_vec, doppler_chunk)
// coh_corr_cuda_ddm_mex('reset')
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Command mode
    if (nrhs >= 1 && mxIsChar(prhs[0])) {
        handleCommandMode(nlhs, plhs, nrhs, prhs);
        return;
    }

    if (nrhs < 7 || nrhs > 8) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Expected 7 or 8 inputs in compute mode.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Output",
                          "At most 1 output is supported.");
    }

    const mxArray* signal_mx = prhs[0];
    const float fs_dec = static_cast<float>(getScalarDouble(prhs[1], "fs_dec"));
    const int prn = getScalarInt(prhs[2], "PRN");
    const mxArray* doppler_mx = prhs[3];
    const mxArray* delay0_mx = prhs[4];
    const float Ti = static_cast<float>(getScalarDouble(prhs[5], "Ti"));
    const mxArray* delay_vec_mx = prhs[6];
    int doppler_chunk = (nrhs == 8) ? getScalarInt(prhs[7], "doppler_chunk") : 8;
    doppler_chunk = std::max(1, doppler_chunk);

    int Nsig = 0;
    int K = 0;
    std::vector<cufftComplex> signal_h = toComplexFloatMatrix(signal_mx, Nsig, K);

    std::vector<float> doppler_h = toFloatRealVector(doppler_mx, "Doppler");
    if (mxGetNumberOfDimensions(doppler_mx) != 2) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Doppler must be 2-D [Lf x K].");
    }
    const int Lf = static_cast<int>(mxGetM(doppler_mx));
    const int Kd = static_cast<int>(mxGetN(doppler_mx));
    if (Kd != K) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "size(Doppler,2) must equal size(signal,2)=K.");
    }

    std::vector<float> delay0_h = toFloatRealVector(delay0_mx, "delay0");
    const int Lt = static_cast<int>(delay0_h.size());
    if (Lt <= 0) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "delay0 must be non-empty.");
    }

    std::vector<float> delay_vec_h = toFloatRealVector(delay_vec_mx, "delay_vec");
    if (static_cast<int>(delay_vec_h.size()) != K) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "length(delay_vec) must equal K.");
    }

    const int N = static_cast<int>(
        std::floor((std::floor((Ti - 1e-5f) / 1e-3f) + 1.0f) * 1e-3f * fs_dec)
    );
    if (N <= 0) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Computed N must be positive.");
    }
    if (N != Nsig) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "size(signal,1)=%d does not match expected N=%d from Ti/fs_dec.",
                          Nsig, N);
    }
    if (Lt > N) {
        mexErrMsgIdAndTxt("coh_corr_cuda_ddm_mex:Input",
                          "Lt=length(delay0) must be <= N.");
    }

    const int max_nb = std::min(doppler_chunk, Lf);

    ensureContext(g_ctx, N, K, Lt, Lf, max_nb, prn, fs_dec, Ti);

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(g_ctx.signal_d, signal_h.data(),
                          sizeof(cufftComplex) * static_cast<size_t>(N) * static_cast<size_t>(K),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(g_ctx.doppler_d, doppler_h.data(),
                          sizeof(float) * static_cast<size_t>(Lf) * static_cast<size_t>(K),
                          cudaMemcpyHostToDevice));

    // Rebuild local code spectrum only when delay_vec changes
    updateCodeReplicaIfNeeded(g_ctx, delay_vec_h);

    // Clear output accumulation buffer
    CUDA_CHECK(cudaMemset(g_ctx.ddm_d, 0,
                          sizeof(float) * static_cast<size_t>(Lt) * static_cast<size_t>(Lf)));

    for (int d0 = 0; d0 < Lf; d0 += g_ctx.max_nb) {
        const int nb = std::min(g_ctx.max_nb, Lf - d0);

        launchDopplerCompKernel(g_ctx.signal_d, g_ctx.doppler_d,
                                N, K, Lf, d0, nb, fs_dec, g_ctx.work_d, 0);
        CUDA_CHECK(cudaGetLastError());

        if (nb == g_ctx.max_nb) {
            CUFFT_CHECK(cufftExecC2C(g_ctx.plan_fwd_full, g_ctx.work_d, g_ctx.work_d, CUFFT_FORWARD));

            launchFreqMultiplyKernel(g_ctx.work_d, g_ctx.Cf_conj_d, N, K, nb, 0);
            CUDA_CHECK(cudaGetLastError());

            CUFFT_CHECK(cufftExecC2C(g_ctx.plan_inv_full, g_ctx.work_d, g_ctx.work_d, CUFFT_INVERSE));
        } else {
            createTailPlans(g_ctx, nb);

            CUFFT_CHECK(cufftExecC2C(g_ctx.plan_fwd_tail, g_ctx.work_d, g_ctx.work_d, CUFFT_FORWARD));

            launchFreqMultiplyKernel(g_ctx.work_d, g_ctx.Cf_conj_d, N, K, nb, 0);
            CUDA_CHECK(cudaGetLastError());

            CUFFT_CHECK(cufftExecC2C(g_ctx.plan_inv_tail, g_ctx.work_d, g_ctx.work_d, CUFFT_INVERSE));
        }

        launchPowerAccumulateKernel(g_ctx.work_d, N, K, Lt, d0, nb, g_ctx.ddm_d, 0);
        CUDA_CHECK(cudaGetLastError());
    }

    mxArray* out = mxCreateNumericMatrix(static_cast<mwSize>(Lt),
                                         static_cast<mwSize>(Lf),
                                         mxSINGLE_CLASS, mxREAL);
    float* out_p = static_cast<float*>(mxGetData(out));

    CUDA_CHECK(cudaMemcpy(out_p, g_ctx.ddm_d,
                          sizeof(float) * static_cast<size_t>(Lt) * static_cast<size_t>(Lf),
                          cudaMemcpyDeviceToHost));

    plhs[0] = out;
}