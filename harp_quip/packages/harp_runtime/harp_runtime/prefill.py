"""Exact index permutation between stages."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _autosort(
    X,
    K,
    Y,
    N: tl.constexpr,
    M,
    B: tl.constexpr,
    S: tl.constexpr,
    INV: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    BG: tl.constexpr,
):
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    r = tl.program_id(1) * BM + tl.arange(0, BM)
    i = tl.arange(0, BS)
    j = tl.arange(0, BS)
    C: tl.constexpr = N // (B * S)
    kg = (g % C) * S + g // C
    if INV:
        xc = i[None, :] * (N // B) + g[:, None]
        yc = g[:, None] * B + j[None, :]
        ko = i[:, None] * B + j[None, :]
    else:
        xc = g[:, None] * B + i[None, :]
        yc = j[None, :] * (N // B) + g[:, None]
        ko = j[None, :] * B + i[:, None]
    x = tl.load(
        X + r[None, :, None] * N + xc[:, None, :],
        (g[:, None, None] < N // B) & (r[None, :, None] < M) & (i[None, None, :] < B),
        other=0,
    )
    k = tl.load(
        K + kg[:, None, None] * (B * B) + ko[None, :, :],
        (g[:, None, None] < N // B) & (i[None, :, None] < B) & (j[None, None, :] < B),
        other=0,
    )
    y = tl.dot(x, k, input_precision="tf32x3")
    tl.store(
        Y + r[None, :, None] * N + yc[:, None, :],
        y,
        (g[:, None, None] < N // B) & (r[None, :, None] < M) & (j[None, None, :] < B),
    )


def stage(x, k, s, inv=False, bm=16, bg=8):
    m, n = x.shape
    b = k.shape[-1]
    k = k.contiguous()
    y = torch.empty_like(x)
    _autosort[(triton.cdiv(n // b, bg), triton.cdiv(m, bm))](
        x,
        k,
        y,
        n,
        m,
        b,
        s,
        inv,
        bm,
        max(16, triton.next_power_of_2(b)),
        bg,
        num_warps=8,
        num_stages=1,
    )
    return y


def autosort(x, stages, inv=False, bm=16, bg=16):
    y = x
    for k, s in (reversed(stages) if inv else stages):
        y = stage(y, k, s, inv, bm, bg if k.shape[-1] <= 8 else max(1, bg // 4))
    return y


@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
def _grouped_stage(
    X,
    K,
    Y,
    N: tl.constexpr,
    M,
    BT: tl.constexpr,
    STEP: tl.constexpr,
    INV: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    BG: tl.constexpr,
):
    groups = tl.program_id(0) * BG + tl.arange(0, BG)
    row = tl.program_id(1) * BM + tl.arange(0, BM)
    i = tl.arange(0, BS)
    j = tl.arange(0, BS)
    base = (groups // STEP) * (BT * STEP) + groups % STEP
    x = tl.load(
        X + row[None, :, None] * N + base[:, None, None] + i[None, None, :] * STEP,
        (groups[:, None, None] < N // BT)
        & (row[None, :, None] < M)
        & (i[None, None, :] < BT),
        other=0,
    )
    offsets = i[:, None] * BT + j[None, :] if INV else j[None, :] * BT + i[:, None]
    k = tl.load(
        K + groups[:, None, None] * (BT * BT) + offsets[None, :, :],
        (groups[:, None, None] < N // BT)
        & (i[None, :, None] < BT)
        & (j[None, None, :] < BT),
        other=0,
    )
    y = tl.dot(x, k, input_precision="tf32x3")
    tl.store(
        Y + row[None, :, None] * N + base[:, None, None] + j[None, None, :] * STEP,
        y,
        (groups[:, None, None] < N // BT)
        & (row[None, :, None] < M)
        & (j[None, None, :] < BT),
    )


def grouped_stage(x, k, b, step, inverse=False, bm=16):
    assert (
        x.is_cuda
        and x.dtype == torch.float32
        and k.dtype == torch.float32
        and not torch.is_grad_enabled()
    )
    rows, n = x.shape
    assert n % (b * step) == 0 and k.shape == (n // b, b, b)
    bg = 8 if b <= 8 else 2
    x = x.contiguous()
    k = k.contiguous()
    out = torch.empty_like(x)
    _grouped_stage[(triton.cdiv(n // b, bg), triton.cdiv(rows, bm))](
        x,
        k,
        out,
        n,
        rows,
        b,
        step,
        inverse,
        bm,
        max(16, triton.next_power_of_2(b)),
        bg,
        num_warps=8,
        num_stages=1,
    )
    return out


def apply(x, stages, inverse=False):
    if len(x) >= 128:
        return autosort(x, stages, inverse)
    y = x
    for k, s in reversed(stages) if inverse else stages:
        y = grouped_stage(y, k, k.shape[-1], s, inverse)
    return y
