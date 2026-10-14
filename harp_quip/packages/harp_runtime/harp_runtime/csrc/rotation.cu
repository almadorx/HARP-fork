#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <torch/extension.h>
__device__ __forceinline__ float dot_b8_s1(const float *x, const float *k, int xs, int ks) {
    float p0 = __fmul_rn(x[0 * xs], k[0 * ks]);
    float p1 = __fmul_rn(x[1 * xs], k[1 * ks]);
    float p2 = __fmul_rn(x[2 * xs], k[2 * ks]);
    float p3 = __fmul_rn(x[3 * xs], k[3 * ks]);
    float p4 = __fmul_rn(x[4 * xs], k[4 * ks]);
    float p5 = __fmul_rn(x[5 * xs], k[5 * ks]);
    float p6 = __fmul_rn(x[6 * xs], k[6 * ks]);
    float p7 = __fmul_rn(x[7 * xs], k[7 * ks]);
    p0 = __fmaf_rn(x[0 * xs], k[0 * ks], p4);
    p1 = __fmaf_rn(x[1 * xs], k[1 * ks], p5);
    p2 = __fmaf_rn(x[2 * xs], k[2 * ks], p6);
    p3 = __fmaf_rn(x[3 * xs], k[3 * ks], p7);
    p0 = __fadd_rn(p0, p2);
    p1 = __fadd_rn(p1, p3);
    p0 = __fadd_rn(p0, p1);
    return p0;
}
__device__ __forceinline__ float dot_b8_s8(const float *x, const float *k, int xs, int ks) {
    float p0 = __fmul_rn(x[0 * xs], k[0 * ks]);
    float p1 = __fmul_rn(x[1 * xs], k[1 * ks]);
    float p2 = __fmul_rn(x[2 * xs], k[2 * ks]);
    float p3 = __fmul_rn(x[3 * xs], k[3 * ks]);
    float p4 = __fmul_rn(x[4 * xs], k[4 * ks]);
    float p5 = __fmul_rn(x[5 * xs], k[5 * ks]);
    float p6 = __fmul_rn(x[6 * xs], k[6 * ks]);
    float p7 = __fmul_rn(x[7 * xs], k[7 * ks]);
    p0 = __fmaf_rn(x[0 * xs], k[0 * ks], p2);
    p1 = __fmaf_rn(x[1 * xs], k[1 * ks], p3);
    p4 = __fmaf_rn(x[4 * xs], k[4 * ks], p6);
    p5 = __fmaf_rn(x[5 * xs], k[5 * ks], p7);
    p0 = __fadd_rn(p0, p1);
    p4 = __fadd_rn(p4, p5);
    p0 = __fadd_rn(p0, p4);
    return p0;
}
__device__ __forceinline__ float dot_b8_s16(const float *x, const float *k, int xs, int ks) {
    float p0 = __fmul_rn(x[0 * xs], k[0 * ks]);
    float p1 = __fmul_rn(x[1 * xs], k[1 * ks]);
    float p2 = __fmul_rn(x[2 * xs], k[2 * ks]);
    float p3 = __fmul_rn(x[3 * xs], k[3 * ks]);
    float p4 = __fmul_rn(x[4 * xs], k[4 * ks]);
    float p5 = __fmul_rn(x[5 * xs], k[5 * ks]);
    float p6 = __fmul_rn(x[6 * xs], k[6 * ks]);
    float p7 = __fmul_rn(x[7 * xs], k[7 * ks]);
    p0 = __fmaf_rn(x[0 * xs], k[0 * ks], p1);
    p2 = __fmaf_rn(x[2 * xs], k[2 * ks], p3);
    p4 = __fmaf_rn(x[4 * xs], k[4 * ks], p5);
    p6 = __fmaf_rn(x[6 * xs], k[6 * ks], p7);
    p0 = __fadd_rn(p0, p4);
    p2 = __fadd_rn(p2, p6);
    p0 = __fadd_rn(p0, p2);
    return p0;
}
__device__ __forceinline__ float dot_b43(const float *x, const float *k, int xs, int ks) {
    float p0 = __fmul_rn(x[0 * xs], k[0 * ks]);
    float p1 = __fmul_rn(x[1 * xs], k[1 * ks]);
    float p2 = __fmul_rn(x[2 * xs], k[2 * ks]);
    float p3 = __fmul_rn(x[3 * xs], k[3 * ks]);
    float p4 = __fmul_rn(x[4 * xs], k[4 * ks]);
    float p5 = __fmul_rn(x[5 * xs], k[5 * ks]);
    float p6 = __fmul_rn(x[6 * xs], k[6 * ks]);
    float p7 = __fmul_rn(x[7 * xs], k[7 * ks]);
    float p8 = __fmul_rn(x[8 * xs], k[8 * ks]);
    float p9 = __fmul_rn(x[9 * xs], k[9 * ks]);
    float p10 = __fmul_rn(x[10 * xs], k[10 * ks]);
    float p11 = __fmul_rn(x[11 * xs], k[11 * ks]);
    float p12 = __fmul_rn(x[12 * xs], k[12 * ks]);
    float p13 = __fmul_rn(x[13 * xs], k[13 * ks]);
    float p14 = __fmul_rn(x[14 * xs], k[14 * ks]);
    float p15 = __fmul_rn(x[15 * xs], k[15 * ks]);
    float p16 = __fmul_rn(x[16 * xs], k[16 * ks]);
    float p17 = __fmul_rn(x[17 * xs], k[17 * ks]);
    float p18 = __fmul_rn(x[18 * xs], k[18 * ks]);
    float p19 = __fmul_rn(x[19 * xs], k[19 * ks]);
    float p20 = __fmul_rn(x[20 * xs], k[20 * ks]);
    float p21 = __fmul_rn(x[21 * xs], k[21 * ks]);
    float p22 = __fmul_rn(x[22 * xs], k[22 * ks]);
    float p23 = __fmul_rn(x[23 * xs], k[23 * ks]);
    float p24 = __fmul_rn(x[24 * xs], k[24 * ks]);
    float p25 = __fmul_rn(x[25 * xs], k[25 * ks]);
    float p26 = __fmul_rn(x[26 * xs], k[26 * ks]);
    float p27 = __fmul_rn(x[27 * xs], k[27 * ks]);
    float p28 = __fmul_rn(x[28 * xs], k[28 * ks]);
    float p29 = __fmul_rn(x[29 * xs], k[29 * ks]);
    float p30 = __fmul_rn(x[30 * xs], k[30 * ks]);
    float p31 = __fmul_rn(x[31 * xs], k[31 * ks]);
    float p32 = __fmul_rn(x[32 * xs], k[32 * ks]);
    float p33 = __fmul_rn(x[33 * xs], k[33 * ks]);
    float p34 = __fmul_rn(x[34 * xs], k[34 * ks]);
    float p35 = __fmul_rn(x[35 * xs], k[35 * ks]);
    float p36 = __fmul_rn(x[36 * xs], k[36 * ks]);
    float p37 = __fmul_rn(x[37 * xs], k[37 * ks]);
    float p38 = __fmul_rn(x[38 * xs], k[38 * ks]);
    float p39 = __fmul_rn(x[39 * xs], k[39 * ks]);
    float p40 = __fmul_rn(x[40 * xs], k[40 * ks]);
    float p41 = __fmul_rn(x[41 * xs], k[41 * ks]);
    float p42 = __fmul_rn(x[42 * xs], k[42 * ks]);
    float p43 = 0.0f;
    float p44 = 0.0f;
    float p45 = 0.0f;
    float p46 = 0.0f;
    float p47 = 0.0f;
    float p48 = 0.0f;
    float p49 = 0.0f;
    float p50 = 0.0f;
    float p51 = 0.0f;
    float p52 = 0.0f;
    float p53 = 0.0f;
    float p54 = 0.0f;
    float p55 = 0.0f;
    float p56 = 0.0f;
    float p57 = 0.0f;
    float p58 = 0.0f;
    float p59 = 0.0f;
    float p60 = 0.0f;
    float p61 = 0.0f;
    float p62 = 0.0f;
    float p63 = 0.0f;
    p0 = __fmaf_rn(x[0 * xs], k[0 * ks], p8);
    p1 = __fmaf_rn(x[1 * xs], k[1 * ks], p9);
    p2 = __fmaf_rn(x[2 * xs], k[2 * ks], p10);
    p3 = __fmaf_rn(x[3 * xs], k[3 * ks], p11);
    p4 = __fmaf_rn(x[4 * xs], k[4 * ks], p12);
    p5 = __fmaf_rn(x[5 * xs], k[5 * ks], p13);
    p6 = __fmaf_rn(x[6 * xs], k[6 * ks], p14);
    p7 = __fmaf_rn(x[7 * xs], k[7 * ks], p15);
    p16 = __fmaf_rn(x[16 * xs], k[16 * ks], p24);
    p17 = __fmaf_rn(x[17 * xs], k[17 * ks], p25);
    p18 = __fmaf_rn(x[18 * xs], k[18 * ks], p26);
    p19 = __fmaf_rn(x[19 * xs], k[19 * ks], p27);
    p20 = __fmaf_rn(x[20 * xs], k[20 * ks], p28);
    p21 = __fmaf_rn(x[21 * xs], k[21 * ks], p29);
    p22 = __fmaf_rn(x[22 * xs], k[22 * ks], p30);
    p23 = __fmaf_rn(x[23 * xs], k[23 * ks], p31);
    p32 = __fmaf_rn(x[32 * xs], k[32 * ks], p40);
    p33 = __fmaf_rn(x[33 * xs], k[33 * ks], p41);
    p34 = __fmaf_rn(x[34 * xs], k[34 * ks], p42);
    p35 = __fmaf_rn(x[35 * xs], k[35 * ks], p43);
    p36 = __fmaf_rn(x[36 * xs], k[36 * ks], p44);
    p37 = __fmaf_rn(x[37 * xs], k[37 * ks], p45);
    p38 = __fmaf_rn(x[38 * xs], k[38 * ks], p46);
    p39 = __fmaf_rn(x[39 * xs], k[39 * ks], p47);
    p48 = __fadd_rn(p48, p56);
    p49 = __fadd_rn(p49, p57);
    p50 = __fadd_rn(p50, p58);
    p51 = __fadd_rn(p51, p59);
    p52 = __fadd_rn(p52, p60);
    p53 = __fadd_rn(p53, p61);
    p54 = __fadd_rn(p54, p62);
    p55 = __fadd_rn(p55, p63);
    p0 = __fadd_rn(p0, p4);
    p1 = __fadd_rn(p1, p5);
    p2 = __fadd_rn(p2, p6);
    p3 = __fadd_rn(p3, p7);
    p16 = __fadd_rn(p16, p20);
    p17 = __fadd_rn(p17, p21);
    p18 = __fadd_rn(p18, p22);
    p19 = __fadd_rn(p19, p23);
    p32 = __fadd_rn(p32, p36);
    p33 = __fadd_rn(p33, p37);
    p34 = __fadd_rn(p34, p38);
    p35 = __fadd_rn(p35, p39);
    p48 = __fadd_rn(p48, p52);
    p49 = __fadd_rn(p49, p53);
    p50 = __fadd_rn(p50, p54);
    p51 = __fadd_rn(p51, p55);
    p0 = __fadd_rn(p0, p2);
    p1 = __fadd_rn(p1, p3);
    p16 = __fadd_rn(p16, p18);
    p17 = __fadd_rn(p17, p19);
    p32 = __fadd_rn(p32, p34);
    p33 = __fadd_rn(p33, p35);
    p48 = __fadd_rn(p48, p50);
    p49 = __fadd_rn(p49, p51);
    p0 = __fadd_rn(p0, p1);
    p16 = __fadd_rn(p16, p17);
    p32 = __fadd_rn(p32, p33);
    p48 = __fadd_rn(p48, p49);
    p0 = __fadd_rn(p0, p32);
    p16 = __fadd_rn(p16, p48);
    p0 = __fadd_rn(p0, p16);
    return p0;
}
__device__ __forceinline__ float dot_b4(const float *x, const float *k, int xs, int ks) {
    float p0 = __fmul_rn(x[0 * xs], k[0 * ks]);
    float p1 = __fmul_rn(x[1 * xs], k[1 * ks]);
    float p2 = __fmul_rn(x[2 * xs], k[2 * ks]);
    float p3 = __fmul_rn(x[3 * xs], k[3 * ks]);
    p0 = __fadd_rn(p0, p2);
    p1 = __fadd_rn(p1, p3);
    p0 = __fadd_rn(p0, p1);
    return p0;
}
__device__ __forceinline__ float dot_b3(const float *x, const float *k, int xs, int ks) {
    float p0 = __fmul_rn(x[0 * xs], k[0 * ks]);
    float p1 = __fmul_rn(x[1 * xs], k[1 * ks]);
    float p2 = __fmul_rn(x[2 * xs], k[2 * ks]);
    float p3 = 0.0f;
    p0 = __fadd_rn(p0, p2);
    p1 = __fadd_rn(p1, p3);
    p0 = __fadd_rn(p0, p1);
    return p0;
}

__host__ __device__ __forceinline__ bool specialized_dot(int b, int s) {
    return (b == 8 && (s == 1 || s == 2 || s == 4 || s == 8 || s == 16 || s == 64 || s == 512)) ||
           (b == 43 && (s == 256 || s == 512)) || ((b == 3 || b == 4) && s >= 32);
}

template <int PAD, int LANE_BITS>
__device__ __forceinline__ float tiled_dot(const float *x, const float *k, int xs, int ks, int b) {
    float p[PAD];
#pragma unroll
    for (int i = 0; i < PAD; i++)
        p[i] = i < b ? __fmul_rn(x[i * xs], k[i * ks]) : 0.0f;
#pragma unroll
    for (int step = 0; step < __ffs(PAD) - 1; step++) {
        int bit = step < LANE_BITS ? LANE_BITS - 1 - step : (__ffs(PAD) - 2) - (step - LANE_BITS);
#pragma unroll
        for (int i = 0; i < PAD; i++) {
            int prior = step < LANE_BITS
                            ? ((1 << LANE_BITS) - (1 << (bit + 1)))
                            : ((1 << LANE_BITS) - 1) | ((PAD - 1) ^ ((1 << (bit + 1)) - 1));
            if (!(i & prior) && !(i & (1 << bit))) {
                if (step == 0 && LANE_BITS > 0 && i < b)
                    p[i] = __fmaf_rn(x[i * xs], k[i * ks], p[i + (1 << bit)]);
                else
                    p[i] = __fadd_rn(p[i], p[i + (1 << bit)]);
            }
        }
    }
    return p[0];
}
template <int PAD>
__device__ __forceinline__ float generic_tiled_dot(const float *x, const float *k, int xs, int ks,
                                                   int b, int s) {
    int bs = 1;
    while (bs * 2 <= 128 / PAD && bs * 2 <= s)
        bs *= 2;
    int lanes = min(PAD, 32 / bs), bits = 0;
    while ((1 << bits) < lanes)
        bits++;
    switch (bits) {
    case 0:
        return tiled_dot<PAD, 0>(x, k, xs, ks, b);
    case 1:
        return tiled_dot<PAD, 1>(x, k, xs, ks, b);
    case 2:
        if constexpr (PAD >= 4)
            return tiled_dot<PAD, 2>(x, k, xs, ks, b);
    case 3:
        if constexpr (PAD >= 8)
            return tiled_dot<PAD, 3>(x, k, xs, ks, b);
    case 4:
        if constexpr (PAD >= 16)
            return tiled_dot<PAD, 4>(x, k, xs, ks, b);
    default:
        if constexpr (PAD >= 32)
            return tiled_dot<PAD, 5>(x, k, xs, ks, b);
    }
    return 0;
}
__device__ __noinline__ float generic_dot(const float *x, const float *k, int xs, int ks, int b,
                                          int s) {
    if (b == 1)
        return __fmul_rn(x[0], k[0]);
    if (b <= 2)
        return generic_tiled_dot<2>(x, k, xs, ks, b, s);
    if (b <= 4)
        return generic_tiled_dot<4>(x, k, xs, ks, b, s);
    if (b <= 8)
        return generic_tiled_dot<8>(x, k, xs, ks, b, s);
    if (b <= 16)
        return generic_tiled_dot<16>(x, k, xs, ks, b, s);
    if (b <= 32)
        return generic_tiled_dot<32>(x, k, xs, ks, b, s);
    if (b <= 64)
        return generic_tiled_dot<64>(x, k, xs, ks, b, s);
    if (b <= 128)
        return generic_tiled_dot<128>(x, k, xs, ks, b, s);
    float acc = 0.0f;
    for (int i = 0; i < b; i++)
        acc = __fmaf_rn(x[i * xs], k[i * ks], acc);
    return acc;
}
template <bool GENERIC>
__device__ __forceinline__ float exact_dot(const float *x, const float *k, int xs, int ks, int b,
                                           int s) {
    if constexpr (GENERIC) {
        if (!specialized_dot(b, s))
            return generic_dot(x, k, xs, ks, b, s);
    }
    if (b == 8) {
        if (s <= 4)
            return dot_b8_s1(x, k, xs, ks);
        if (s == 8)
            return dot_b8_s8(x, k, xs, ks);
        return dot_b8_s16(x, k, xs, ks);
    }
    if (b == 43)
        return dot_b43(x, k, xs, ks);
    if (b == 4)
        return dot_b4(x, k, xs, ks);
    return dot_b3(x, k, xs, ks);
}

template <bool INV, int FIRST, int LAST, bool GENERIC>
__global__ void boundary_pair(const void *xx, const float *k1, const float *k2, void *yy,
                              const c10::Half *scale, int n, int a, int b, int s,
                              float multiplier) {
    extern __shared__ float sm[];
    int p = a * b;
    int f = blockIdx.x % (n / p), row = blockIdx.x / (n / p), c = f / s, u = f % s;
    int j = threadIdx.x, d = j / a, v = j % a, base = row * n + c * p * s + u;
    const float *x = (const float *)xx;
    float *mid = sm;
    const float *source = x + base;
    int stride = s;
    if constexpr (FIRST != 0) {
        if (j < p) {
            float val;
            if constexpr (FIRST == 2)
                val = x[base + j * s];
            else
                val = float(((const c10::Half *)xx)[base + j * s]);
            if constexpr (FIRST == 1 || FIRST == 2)
                val = __fmul_rn(val, float(scale[(base + j * s) % n]));
            sm[p + j] = val;
        }
        __syncthreads();
        source = sm + p;
        stride = 1;
    }
    if (j < p) {
        float acc;
        if (b == 1) {
            int kg = (c * s + u) * a * a;
            acc =
                exact_dot<GENERIC>(source, k1 + kg + (INV ? v : v * a), stride, INV ? a : 1, a, s);
        } else if (!INV) {
            int kg = ((c * b + d) * s + u) * a * a;
            acc = exact_dot<GENERIC>(source + d * a * stride, k1 + kg + v * a, stride, 1, a, s);
        } else {
            int kg = (c * a * s + v * s + u) * b * b;
            acc = exact_dot<GENERIC>(source + v * stride, k2 + kg + d, a * stride, b, b, a * s);
        }
        mid[j] = acc;
    }
    __syncthreads();
    if (j < p) {
        float acc = mid[j];
        if (b != 1) {
            if (!INV) {
                int kg = (c * a * s + v * s + u) * b * b;
                acc = exact_dot<GENERIC>(mid + v, k2 + kg + d * b, a, 1, b, a * s);
            } else {
                int kg = ((c * b + d) * s + u) * a * a;
                acc = exact_dot<GENERIC>(mid + d * a, k1 + kg + v, 1, a, a, s);
            }
        }
        if constexpr (LAST == 1)
            ((c10::Half *)yy)[base + j * s] = c10::Half(__fdiv_rn(acc, multiplier));
        else if constexpr (LAST == 2)
            ((c10::Half *)yy)[base + j * s] =
                c10::Half(__fmul_rn(__fmul_rn(acc, float(scale[(base + j * s) % n])), multiplier));
        else if constexpr (LAST == 3)
            ((float *)yy)[base + j * s] =
                __fmul_rn(__fmul_rn(acc, float(scale[(base + j * s) % n])), multiplier);
        else
            ((float *)yy)[base + j * s] = acc;
    }
}
struct Op {
    torch::Tensor k1, k2;
    int a, b, s;
};
class RotationPlan {
    std::vector<Op> ops;
    int n;
    torch::Device dev;
    template <bool INV, int FIRST, int LAST>
    void launch(const torch::Tensor &x, torch::Tensor &y, const torch::Tensor &scale, const Op &op,
                cudaStream_t stream, double multiplier) {
        int p = op.a * op.b, nt = 32;
        while (nt < p)
            nt *= 2;
#define LAUNCH_PAIR(G)                                                                             \
    boundary_pair<INV, FIRST, LAST, G>                                                             \
        <<<x.size(0) * (n / p), nt, p * sizeof(float) * (FIRST ? 2 : 1), stream>>>(                \
            x.data_ptr(), op.k1.data_ptr<float>(), op.b == 1 ? nullptr : op.k2.data_ptr<float>(),  \
            y.data_ptr(), scale.defined() ? scale.data_ptr<c10::Half>() : nullptr, n, op.a, op.b,  \
            op.s, float(multiplier))
        if (specialized_dot(op.a, op.s) && (op.b == 1 || specialized_dot(op.b, op.a * op.s))) {
            LAUNCH_PAIR(false);
        } else {
            LAUNCH_PAIR(true);
        }
#undef LAUNCH_PAIR
    }
    template <bool INV>
    torch::Tensor run(torch::Tensor x, torch::Tensor scale, int mode, double multiplier) {
        c10::cuda::CUDAGuard guard(x.device());
        auto opts = x.options().dtype(torch::kFloat32);
        auto tmp0 = torch::empty(x.sizes(), opts), tmp1 = torch::empty(x.sizes(), opts);
        auto out = torch::empty(
            x.sizes(), opts.dtype((mode == 1 || mode == 2) ? torch::kFloat16 : torch::kFloat32));
        auto stream = at::cuda::getCurrentCUDAStream();
        auto cur = x;
        for (int i = 0; i < (int)ops.size(); i++) {
            const auto &op = ops[INV ? ops.size() - 1 - i : i];
            bool first = i == 0, last = i == (int)ops.size() - 1;
            auto target = last ? out : (i % 2 ? tmp1 : tmp0);
            int f =
                first ? (mode == 1
                             ? (x.scalar_type() == torch::kFloat16 ? 1 : 2)
                             : ((mode == 2 || mode == 3) && x.scalar_type() == torch::kFloat16 ? 3
                                                                                               : 0))
                      : 0;
            int l = last ? mode : 0;
#define LAUNCH_LAST(F)                                                                             \
    do {                                                                                           \
        if (l == 1)                                                                                \
            launch<INV, F, 1>(cur, target, scale, op, stream, multiplier);                         \
        else if (l == 2)                                                                           \
            launch<INV, F, 2>(cur, target, scale, op, stream, multiplier);                         \
        else if (l == 3)                                                                           \
            launch<INV, F, 3>(cur, target, scale, op, stream, multiplier);                         \
        else                                                                                       \
            launch<INV, F, 0>(cur, target, scale, op, stream, multiplier);                         \
    } while (0)
            if (f == 1) {
                LAUNCH_LAST(1);
            } else if (f == 2) {
                LAUNCH_LAST(2);
            } else if (f == 3) {
                LAUNCH_LAST(3);
            } else {
                LAUNCH_LAST(0);
            }
#undef LAUNCH_LAST
            cur = target;
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

  public:
    RotationPlan(std::vector<torch::Tensor> k, std::vector<int64_t> strides,
                 std::vector<bool> paired)
        : n(0), dev(torch::kCPU) {
        TORCH_CHECK(!k.empty() && k[0].dim() == 3);
        n = k[0].size(0) * k[0].size(1);
        dev = k[0].device();
        TORCH_CHECK(k.size() == strides.size() && k.size() == paired.size());
        for (int t = 0; t < (int)k.size(); t++) {
            auto first = k[t];
            auto second = paired[t] ? k.at(t + 1) : torch::Tensor();
            int a = first.size(1), b = paired[t] ? second.size(1) : 1, s = strides[t];
            TORCH_CHECK(first.is_cuda() && first.is_contiguous() &&
                        first.scalar_type() == torch::kFloat32 && first.device() == dev &&
                        first.dim() == 3 && first.size(2) == a && first.size(0) == n / a);
            TORCH_CHECK(s > 0 && n % (a * b * s) == 0 && a * b <= 1024);
            TORCH_CHECK(a >= 1 && b >= 1);
            if (b != 1) {
                TORCH_CHECK(second.is_contiguous() && second.device() == dev &&
                            second.scalar_type() == torch::kFloat32 && second.dim() == 3 &&
                            second.size(2) == b && second.size(0) == n / b);
                t++;
            }
            ops.push_back({first, second, a, b, s});
        }
    }
    torch::Tensor apply(torch::Tensor x, torch::Tensor scale, bool inverse, int mode,
                        double multiplier) {
        TORCH_CHECK(x.device() == dev && x.dim() == 2 && x.size(1) == n && x.is_contiguous() &&
                    x.size(0) > 0 && x.size(0) <= 8);
        TORCH_CHECK(mode >= 0 && mode <= 3 &&
                    (x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16));
        TORCH_CHECK(mode || x.scalar_type() == torch::kFloat32);
        TORCH_CHECK(std::isfinite(multiplier) && multiplier > 0);
        if (mode)
            TORCH_CHECK(scale.device() == dev && scale.numel() == n &&
                        scale.scalar_type() == torch::kFloat16 && scale.is_contiguous());
        return inverse ? run<true>(x, scale, mode, multiplier)
                       : run<false>(x, scale, mode, multiplier);
    }
};

class DependencyGuard {
    struct Mapping {
        pybind11::object key, expected;
    };
    struct Owner {
        pybind11::object weak;
        std::vector<Mapping> mappings, values;
    };
    std::vector<torch::Tensor> tensors;
    std::vector<int64_t> versions;
    std::vector<Owner> owners;

  public:
    DependencyGuard(std::vector<torch::Tensor> ts, pybind11::list bs, pybind11::list vs)
        : tensors(std::move(ts)) {
        for (const auto &t : tensors)
            versions.push_back(t._version());
        auto add = [&](pybind11::handle item, bool mapping) {
            auto row = pybind11::reinterpret_borrow<pybind11::tuple>(item);
            auto raw = PyWeakref_GetObject(row[0].ptr());
            size_t i = 0;
            while (i < owners.size() && PyWeakref_GetObject(owners[i].weak.ptr()) != raw)
                i++;
            if (i == owners.size())
                owners.push_back({pybind11::reinterpret_borrow<pybind11::object>(row[0]), {}, {}});
            (mapping ? owners[i].mappings : owners[i].values)
                .push_back({pybind11::reinterpret_borrow<pybind11::object>(row[1]),
                            pybind11::reinterpret_borrow<pybind11::object>(row[2])});
        };
        for (auto item : bs)
            add(item, true);
        for (auto item : vs)
            add(item, false);
    }
    bool unchanged() {
        for (size_t i = 0; i < tensors.size(); i++)
            if (tensors[i]._version() != versions[i])
                return false;
        for (auto &entry : owners) {
            auto owner = PyWeakref_GetObject(entry.weak.ptr());
            if (owner == Py_None)
                return false;

            auto dict = pybind11::reinterpret_steal<pybind11::object>(
                PyObject_GenericGetDict(owner, nullptr));
            if (!dict) {
                PyErr_Clear();
                return false;
            }
            for (auto &record : entry.mappings) {
                auto mapping = PyDict_GetItemWithError(dict.ptr(), record.key.ptr());
                if (!mapping || !PyDict_Check(mapping) ||
                    PyDict_Size(mapping) != PyDict_Size(record.expected.ptr()))
                    return false;
                PyObject *key, *expected;
                Py_ssize_t pos = 0;
                while (PyDict_Next(record.expected.ptr(), &pos, &key, &expected)) {
                    auto found = PyDict_GetItemWithError(mapping, key);
                    auto value = expected == Py_None ? Py_None : PyWeakref_GetObject(expected);
                    if (!found || found != value)
                        return false;
                }
            }
            for (auto &record : entry.values) {
                auto value = PyDict_GetItemWithError(dict.ptr(), record.key.ptr());
                if (!value || PyObject_RichCompareBool(value, record.expected.ptr(), Py_EQ) != 1)
                    return false;
            }
        }
        return true;
    }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<RotationPlan>(m, "RotationPlan")
        .def(pybind11::init<std::vector<torch::Tensor>, std::vector<int64_t>, std::vector<bool>>())
        .def("apply", &RotationPlan::apply)
        .def("apply_guarded",
             [](RotationPlan &plan, torch::Tensor x, torch::Tensor scale, bool inverse, int mode,
                double multiplier, DependencyGuard &guard) {
                 return guard.unchanged() ? plan.apply(x, scale, inverse, mode, multiplier)
                                          : torch::Tensor();
             });
    pybind11::class_<DependencyGuard>(m, "DependencyGuard")
        .def(pybind11::init<std::vector<torch::Tensor>, pybind11::list, pybind11::list>())
        .def("unchanged", &DependencyGuard::unchanged);
}
