import os
import numpy as np
import cupy as cp

# Standalone .cu for GPU kernel for easier developing (linting)
with open(os.path.join(os.path.dirname(__file__), "seladd.cu")) as sel_add_source_file:
    sel_add_source = "".join(sel_add_source_file.readlines())

sel_kern = cp.RawModule(
    code=sel_add_source,
    options=("-std=c++11",),
)

sel_add_kern = sel_kern.get_function("sel_add_kern")
afa_multiply_kern = sel_kern.get_function("afa_multiply_kern")
delta_w_bar_and_fa_kern = sel_kern.get_function("delta_w_bar_and_fa_kern")
beta_inversion_kern = sel_kern.get_function("beta_inversion_kern")
beye_kern = sel_kern.get_function("beye_kern")
stable_det_kern = sel_kern.get_function("stable_det_kern")
bar_F_multiply_kern = sel_kern.get_function("bar_F_multiply_kern")


def sel_call(delta, e_accept_indices, dest, kern, mask=False):
    # avoid silent error caused by mismatched type
    if mask:
        assert e_accept_indices.dtype == np.bool_
    else:
        assert e_accept_indices.dtype == cp.uint32
    assert delta.dtype == cp.float64
    assert dest.dtype == cp.float64
    N_to_be_add = len(e_accept_indices)
    if N_to_be_add != 0:
        other_dim = int(np.prod(dest.shape[1:]))
        total_to_process = N_to_be_add * other_dim
        block_num = total_to_process // 1024 + 1
        if block_num == 1:
            thread_num = total_to_process
        else:
            thread_num = 1024
        # 2**31 是 cuda 能开的 block 数目的上限; 1024 是 thread 数上限，不同设备最优的 thread 数目不一致，但影响不大。
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
        block_num = np.min([block_num, 2**31 - 1])
        # Kernel 里面使用 unsigned int 计算索引，需要预防整数溢出
        assert total_to_process + block_num * thread_num < 2**32
        kern(
            (block_num,),
            (thread_num,),
            (delta, e_accept_indices, dest, other_dim, total_to_process),
        )


def sel_add(delta, e_accept_indices, dest):
    sel_call(delta, e_accept_indices, dest, sel_add_kern)
        
def afa_multiply(afa, af, ti, frac):
    assert afa.dtype == cp.float64
    assert af.dtype == cp.float64
    assert frac.dtype == cp.float64
    assert ti.dtype == cp.int32
    l_s = ti.shape[1]
    N = af.shape[0]
    if N != 0:
        block_num = N // 1024 + 1
        l_ex_t = af.shape[2] - 1
        if block_num == 1:
            thread_num = N
        else:
            thread_num = 1024
        block_num = np.min([block_num, 2**31 - 1])
        afa_multiply_kern((block_num,), (thread_num,), (afa, ti, frac, af, N, l_s, l_ex_t))

def delta_w_bar_and_fa(Δw_bar, ti, frac, bundle_move, fa, F, l_A, sigs, e_accept_indices):
    assert Δw_bar.dtype == cp.float64
    assert fa.dtype == cp.float64
    assert F.dtype == cp.float64
    assert frac.dtype == cp.float64
    assert ti.dtype == cp.int32
    assert l_A.dtype == cp.int32
    assert bundle_move.dtype == cp.float64
    assert sigs.dtype == cp.float64
    assert e_accept_indices.dtype == cp.uint32
    N = e_accept_indices.shape[0]
    if N != 0:
        block_num = N // 1024 + 1
        l_ex_t = F.shape[1] - 1
        if block_num == 1:
            thread_num = N
        else:
            thread_num = 1024
        block_num = np.min([block_num, 2**31 - 1])
        delta_w_bar_and_fa_kern((block_num,), (thread_num,), (Δw_bar, ti, frac, bundle_move, fa, F, l_A, sigs, e_accept_indices, N, l_ex_t))
        
def beta_inversion(B_inv, B):
    assert B_inv.dtype == cp.float64
    assert B.dtype == cp.float64
    N = B.shape[0]
    if N != 0:
        block_num = N // 1024 + 1
        if block_num == 1:
            thread_num = N
        else:
            thread_num = 1024
        block_num = np.min([block_num, 2**31 - 1])
        beta_inversion_kern((block_num,), (thread_num,), (B_inv, B, N))
        
def stable_det(A):
    assert A.dtype == cp.float64
    assert A.ndim == 3
    assert A.shape[1] == 2
    assert A.shape[2] == 2
    N = A.shape[0]
    det_A = cp.zeros(N, dtype=cp.float64)
    if N != 0:
        block_num = N // 1024 + 1
        if block_num == 1:
            thread_num = N
        else:
            thread_num = 1024
        block_num = np.min([block_num, 2**31 - 1])
        stable_det_kern((block_num,), (thread_num,), (det_A, A, N))
    return det_A

def bar_F_multiply(ti, frac, F, l_A, e_accept_indices):
    assert F.dtype == cp.float64
    assert frac.dtype == cp.float64
    assert ti.dtype == cp.int32
    assert l_A.dtype == cp.int32
    if e_accept_indices is not None:
        assert e_accept_indices.dtype == cp.uint32
    l_s = ti.shape[1]
    N = ti.shape[0] * l_s
    bar_F = cp.zeros((ti.shape[0], l_s, F.shape[1]), dtype=cp.float64)
    if N != 0:
        block_num = N // 1024 + 1
        l_ex_t = F.shape[1] - 1
        if block_num == 1:
            thread_num = N
        else:
            thread_num = 1024
        block_num = np.min([block_num, 2**31 - 1])
        if e_accept_indices is None:
            e_accept_indices = 0 # 变为nullptr
        bar_F_multiply_kern((block_num,), (thread_num,), (bar_F, ti, frac, F, l_A, e_accept_indices, N, l_s, l_ex_t))
    return bar_F


def beye(dest, length, coef):
    assert dest.dtype == cp.float64
    assert length.dtype == cp.uint32
    assert coef.dtype == cp.float64
    batch, width, _ = dest.shape
    N = batch * width
    if N != 0:
        block_num = N // 1024 + 1
        if block_num == 1:
            thread_num = N
        else:
            thread_num = 1024
        block_num = np.min([block_num, 2**31 - 1])
        beye_kern((block_num,), (thread_num,), (dest, length, coef, batch, width))
