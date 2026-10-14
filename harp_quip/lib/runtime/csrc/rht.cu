#include "hadamard.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <int L, int NT, typename IT, typename OT, bool PRE, bool FINAL>
__global__ void fht_boundary(const IT *x, OT *y, const c10::Half *scale, int K, float norm) {
    constexpr int NE = 4, NC = L / (NT * NE), WS = (NT < 32 ? NT : 32), NW = NT / WS;
    __shared__ uint4 smem[L / 4];
    float v[NC][NE];
    int base = blockIdx.x * L, sb = (blockIdx.x % K) * L;
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int i = 0; i < NE; i++) {
            int q = (c * NT + threadIdx.x) * NE + i;
            float z = float(x[base + q]);
            if constexpr (PRE)
                z = __fmul_rn(z, float(scale[sb + q]));
            v[c][i] = z;
        }
    }
    hadamard_mult_thread<2, NC>(v);
    hadamard_mult_warp<cilog2(WS), 0, NC, NE>(v);
    if constexpr (NW > 1) {
        exchange_smem_pre<NC, NC, NE, WS, NW, true, uint4>(v, smem);
        hadamard_mult_warp<cilog2(NW), 0, NC, NE>(v);
        exchange_smem_pre<NC, NC, NE, WS, NW, false, uint4>(v, smem);
    }
    if constexpr (NC > 1) {
        float vt[NE][NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int i = 0; i < NE; i++)
                vt[i][c] = v[c][i];
        }
        hadamard_mult_thread<cilog2(NC), NE>(vt);
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int i = 0; i < NE; i++)
                v[c][i] = vt[i][c];
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int i = 0; i < NE; i++) {
            int q = (c * NT + threadIdx.x) * NE + i;
            float z = __fmul_rn(v[c][i], norm);
            if constexpr (FINAL) {
                if constexpr (PRE)
                    z = __fmul_rn(z, 1.f / 32.f);
                else
                    z = __fmul_rn(__fmul_rn(z, float(scale[sb + q])), 32.f);
            }
            y[base + q] = OT(z);
        }
    }
}
template <typename OT, bool PRE>
__global__ void finish_boundary(const float *x, OT *y, const c10::Half *s, int n, int total) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < total) {
        float z = x[q];
        if constexpr (PRE)
            z = __fmul_rn(z, 1.f / 32.f);
        else
            z = __fmul_rn(__fmul_rn(z, float(s[q % n])), 32.f);
        y[q] = OT(z);
    }
}
template <typename IT, typename OT, bool PRE, bool FINAL>
void launch(torch::Tensor x, torch::Tensor y, torch::Tensor s, int K, float norm,
            cudaStream_t stream) {
    int L = x.size(1) / K, blocks = x.size(0) * K;
#define CASE(N, T)                                                                                 \
    case N:                                                                                        \
        fht_boundary<N, T, IT, OT, PRE, FINAL><<<blocks, T, 0, stream>>>(                          \
            x.data_ptr<IT>(), y.data_ptr<OT>(), s.data_ptr<c10::Half>(), K, norm);                 \
        break
    switch (L) {
        CASE(32, 8);
        CASE(64, 8);
        CASE(128, 16);
        CASE(256, 32);
        CASE(512, 64);
        CASE(1024, 128);
        CASE(2048, 256);
        CASE(4096, 256);
        CASE(8192, 256);
    default:
        TORCH_CHECK(false, "Unsupported FHT subwidth");
    }
#undef CASE
}
template <typename IT, typename OT, bool PRE>
torch::Tensor run(torch::Tensor x, torch::Tensor s, torch::Tensor had, int K, cudaStream_t stream) {
    int n = x.size(1);
    float norm = float(1.0 / std::sqrt(double(n)));
    auto out = torch::empty(
        x.sizes(),
        x.options().dtype(std::is_same<OT, float>::value ? torch::kFloat32 : torch::kFloat16));
    if (K == 1)
        launch<IT, OT, PRE, true>(x, out, s, K, norm, stream);
    else {
        auto tmp = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
        launch<IT, float, PRE, false>(x, tmp, s, K, norm, stream);

        auto prod = at::matmul(had, tmp.view({x.size(0), K, n / K})).reshape(x.sizes());
        finish_boundary<OT, PRE><<<(x.numel() + 255) / 256, 256, 0, stream>>>(
            prod.data_ptr<float>(), out.data_ptr<OT>(), s.data_ptr<c10::Half>(), n, x.numel());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor apply(torch::Tensor x, torch::Tensor s, torch::Tensor had, int K, bool pre,
                    bool halfout) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.size(0) > 0 &&
                x.size(0) <= 8 && x.numel() < 2147483647);
    TORCH_CHECK(s.is_cuda() && s.device() == x.device() && s.is_contiguous() &&
                s.scalar_type() == torch::kFloat16 && s.numel() == x.size(1));
    TORCH_CHECK(K > 0 && K <= x.size(1));
    TORCH_CHECK(x.size(1) % K == 0);
    if (K > 1)
        TORCH_CHECK(had.is_cuda() && had.device() == x.device() &&
                    had.scalar_type() == torch::kFloat32 && had.dim() == 2 && had.size(0) == K &&
                    had.size(1) == K);
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(IT)                                                                                    \
    if (pre)                                                                                       \
        return run<IT, c10::Half, true>(x, s, had, K, stream);                                     \
    if (halfout)                                                                                   \
        return run<IT, c10::Half, false>(x, s, had, K, stream);                                    \
    return run<IT, float, false>(x, s, had, K, stream)
    if (x.scalar_type() == torch::kFloat16) {
        RUN(c10::Half);
    }
    TORCH_CHECK(x.scalar_type() == torch::kFloat32);
    RUN(float);
#undef RUN
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("apply", &apply); }
