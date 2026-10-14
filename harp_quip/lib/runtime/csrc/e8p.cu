#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cuda_pipeline.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/extension.h>
#include <torch/types.h>

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xffffffff
#define HALF_MASK 0x0000ffff

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    do {                                                                                           \
        CHECK_CUDA(x);                                                                             \
        CHECK_CONTIGUOUS(x);                                                                       \
    } while (false)
#define gpuErrchk(ans) C10_CUDA_CHECK(ans)

__device__ static inline uint32_t add_as_half2(uint32_t x, uint32_t y) {
    uint32_t z;
    asm("add.f16x2 %0,%1,%2;" : "=r"(z) : "r"(x), "r"(y));
    return z;
}

__device__ static inline uint32_t mask_lop3(uint32_t x, uint32_t m0, uint32_t m1) {
    uint32_t y;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(y) : "r"(x), "r"(m0), "r"(m1));
    return y;
}

#define BASE_OFFSET 0xd080d080
#define XMASK 0x00f000f0
#define WMASK 0x50085008

__global__ static void

decode_batch_e8p_kernel(float *__restrict__ output, const uint2 *__restrict__ input,
                        const uint2 *__restrict__ weights_compressed,
                        const uint32_t *__restrict__ codebook_abs, int N, int K, int B) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;

    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {

        float z0 = 0.0;
        float z1 = 0.0;
        float z2 = 0.0;
        float z3 = 0.0;

        for (int iik = warpId; iik < (K >> 6); iik += 32) {

            uint2 w_compr = weights_compressed[laneId + 32 * iik + (K >> 1) * iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;

            uint32_t s = b;
            s = s ^ (s >> 4);
            s = s ^ (s >> 8);
            s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);

            const uint32_t *row_input =
                reinterpret_cast<const uint32_t *>(input) + (laneId >> 2) * (K >> 1) + 32 * iik;
            uint32_t shifted_laneId = (laneId & 3) << 3;

            {
                uint32_t x = codebook_abs[(a >> 0) & 255];
                x = x ^ ((s & 0x11111111) * 14);

                uint32_t o = BASE_OFFSET | ((sb & 0x00010001) << 4);

                uint32_t w00 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
                uint32_t w01 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
                uint32_t w02 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
                uint32_t w03 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

                x = codebook_abs[(a >> 8) & 255];
                x = x ^ ((s & 0x22222222) * 7);

                o = BASE_OFFSET | ((sb & 0x00020002) << 3);

                uint32_t w10 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
                uint32_t w11 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
                uint32_t w12 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
                uint32_t w13 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

                uint32_t x_in0 = ((laneId >> 2) < B ? row_input[shifted_laneId | 0] : 0u);
                uint32_t x_in1 = ((laneId >> 2) < B ? row_input[shifted_laneId | 1] : 0u);

                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                    " { %0, %1, %2, %3 },"
                    " { %4, %5, %6, %7 },"
                    " { %8, %9 },"
                    " { %0, %1, %2, %3 };"
                    : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                    : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(x_in0), "r"(x_in1));

                x_in0 = ((laneId >> 2) < B ? row_input[shifted_laneId | 2] : 0u);
                x_in1 = ((laneId >> 2) < B ? row_input[shifted_laneId | 3] : 0u);

                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                    " { %0, %1, %2, %3 },"
                    " { %4, %5, %6, %7 },"
                    " { %8, %9 },"
                    " { %0, %1, %2, %3 };"
                    : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                    : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(x_in0), "r"(x_in1));
            }

            {
                uint32_t x = codebook_abs[(a >> 16) & 255];
                s = s >> 2;
                x = x ^ ((s & 0x11111111) * 14);

                uint32_t o = BASE_OFFSET | ((sb & 0x00040004) << 2);

                uint32_t w00 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
                uint32_t w01 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
                uint32_t w02 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
                uint32_t w03 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

                x = codebook_abs[(a >> 24) & 255];
                x = x ^ ((s & 0x22222222) * 7);

                o = BASE_OFFSET | ((sb & 0x00080008) << 1);

                uint32_t w10 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
                uint32_t w11 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
                uint32_t w12 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
                uint32_t w13 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

                uint32_t x_in0 = ((laneId >> 2) < B ? row_input[shifted_laneId | 4] : 0u);
                uint32_t x_in1 = ((laneId >> 2) < B ? row_input[shifted_laneId | 5] : 0u);

                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                    " { %0, %1, %2, %3 },"
                    " { %4, %5, %6, %7 },"
                    " { %8, %9 },"
                    " { %0, %1, %2, %3 };"
                    : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                    : "r"(w00), "r"(w10), "r"(w01), "r"(w11), "r"(x_in0), "r"(x_in1));

                x_in0 = ((laneId >> 2) < B ? row_input[shifted_laneId | 6] : 0u);
                x_in1 = ((laneId >> 2) < B ? row_input[shifted_laneId | 7] : 0u);

                asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                    " { %0, %1, %2, %3 },"
                    " { %4, %5, %6, %7 },"
                    " { %8, %9 },"
                    " { %0, %1, %2, %3 };"
                    : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                    : "r"(w02), "r"(w12), "r"(w03), "r"(w13), "r"(x_in0), "r"(x_in1));
            }
        }

        int feature = (iin << 4) + 2 * (laneId >> 2);
        int batch = 2 * (laneId & 3);
        if (batch < B) {
            atomicAdd(output + batch * N + feature, z0);
            atomicAdd(output + batch * N + feature + 1, z2);
        }
        if (batch + 1 < B) {
            atomicAdd(output + (batch + 1) * N + feature, z1);
            atomicAdd(output + (batch + 1) * N + feature + 1, z3);
        }
    }
}

__host__ extern torch::Tensor decode_batch_e8p(torch::Tensor x, torch::Tensor weights_compressed,
                                               torch::Tensor codebook_abs) {

    CHECK_INPUT(x);
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % alignof(uint2) == 0,
                "Input storage must be aligned for uint2 loads");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8);
    TORCH_CHECK(x.device() == weights_compressed.device() && x.device() == codebook_abs.device());
    TORCH_CHECK(weights_compressed.dim() == 4);
    TORCH_CHECK(weights_compressed.size(3) == 4);
    TORCH_CHECK(weights_compressed.size(2) == 8);
    TORCH_CHECK(codebook_abs.dim() == 1);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt32);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(1) << 6);
    TORCH_CHECK(codebook_abs.size(-1) == 256);

    int64_t N = weights_compressed.size(0) * 16;
    int64_t K = x.size(-1);

    TORCH_CHECK(K % 64 == 0, "K is not divisible by 64");
    TORCH_CHECK(N % 16 == 0, "N is not divisible by 16");

    TORCH_CHECK(K < 65536, "K is not too large");
    TORCH_CHECK(N < 65536, "N is not too large");

    at::DeviceGuard guard(x.device());
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .layout(torch::kStrided)
                                       .device(torch::kCUDA)
                                       .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{x.size(0), N}, options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, x.get_device());
    int64_t grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    const dim3 block_size(32, 32);

    decode_batch_e8p_kernel<<<grid_size, block_size, 0, stream>>>(
        output.data_ptr<float>(), (const uint2 *)x.data_ptr<c10::Half>(),
        (const uint2 *)weights_compressed.data_ptr<int64_t>(),
        (const uint32_t *)codebook_abs.data_ptr<int32_t>(), N, K, x.size(0));

    gpuErrchk(cudaPeekAtLastError());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode_batch_e8p", &decode_batch_e8p); }
