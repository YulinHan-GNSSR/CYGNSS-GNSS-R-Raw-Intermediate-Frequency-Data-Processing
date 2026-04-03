function build_mex()
% Build script for coh_corr_cuda_ddm_mex (Windows + NVIDIA + MATLAB PCT).
%
% Usage:
%   >> build_mex

    src = {
        'coh_corr_cuda_ddm_mex.cu', ...
        'coh_corr_cuda_kernels.cu', ...
        'cacode_gps_ca.cpp'
    };

    mexcuda('-R2018a', ...
        '-output', 'coh_corr_cuda_ddm_mex', ...
        src{:}, ...
        '-lcufft');


    fprintf('Built coh_corr_cuda_ddm_mex successfully.\n');
end