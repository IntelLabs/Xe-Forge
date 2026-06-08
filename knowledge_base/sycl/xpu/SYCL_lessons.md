# SYCL Lessons

## Lesson: Named Kernels for `parallel_for`

### Symptom

Compilation can fail with an error similar to:

```text
unnamed type '(lambda ...)' is invalid; provide a kernel name, or use '-fsycl-unnamed-lambda'
```

### Cause

Some SYCL toolchains reject unnamed kernel lambdas in `parallel_for`.

### Fix

Prefer an explicitly named kernel:

```cpp
h.parallel_for<class ScalarMulKernel>(
    sycl::nd_range<1>(global_size, WG_SIZE),
    [=](sycl::nd_item<1> item) {
        // kernel body
    }
);
```

### Fallback

If a code path cannot be easily updated to named kernels, add this compile flag:

```text
-fsycl-unnamed-lambda
```

### Practical Note

For portability and future compiler compatibility, named kernels are the safer default.

---

## Lesson: Sub-group Coalesced Memory Access for Intel Xe GPUs

### Symptom

A kernel performs element-wise operations on large tensors but runs at only 0.66× PyTorch speed, despite using vectorized loads and a reasonable work-group size.

### Root Cause

**Poor memory coalescing**. If each work-item processes a contiguous block of N elements independently, lanes within a sub-group access memory N elements apart. With N=32 and sub-group size=16, consecutive lanes load from offsets 0, 32, 64, ... — hitting 16 separate cache lines per SIMD instruction, wasting 75% of fetched data.

### Fix: Sub-group Aware Layout

Reorganize the computation so **consecutive lanes access consecutive memory addresses**. For bf16 pairs, treat them as `uint32` and distribute work by lane:

```cpp
enum { WG = 512, ITERS = 128, SG_SZ = 16, SGS_PER_WG = WG / SG_SZ };
enum { PAIRS_PER_SG = SG_SZ * ITERS };  // 2048 pairs per sub-group

q->submit([&](sycl::handler& h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(global, WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = item.get_sub_group();
            int lane = sg.get_local_linear_id();
            int64_t sg_id = item.get_group(0) * SGS_PER_WG + sg.get_group_linear_id();
            int64_t sg_base = sg_id * PAIRS_PER_SG;

            #pragma unroll
            for (int i = 0; i < ITERS; i++) {
                int64_t idx = sg_base + i * SG_SZ + lane;
                // Lane 0 reads idx=sg_base+0
                // Lane 1 reads idx=sg_base+1
                // ... Lane 15 reads idx=sg_base+15
                // All lanes read from same cache line!
                uint32_t data = load_u32(idx);
                process_and_store(data, idx);
            }
        });
});
```

**Key points:**
1. All 16 lanes in a sub-group load from consecutive offsets: `[sg_base, sg_base+1, ..., sg_base+15]`.
2. Together they fill exactly **one 64-byte cache line** per iteration.
3. Use `[[sycl::reqd_sub_group_size(16)]]` to guarantee the lane mapping.
4. Use `enum` instead of `constexpr` for compile-time constants to avoid DPC++ host/device capture mismatch.

### Expected Impact

- **Memory bandwidth utilization**: ~100% (vs. 25% with poor coalescing).
- **Performance gain**: 0.66× → 0.97× (47% improvement toward parity with optimized reference).

### Reference (oneAPI GPU Optimization Guide)

- Thread Mapping and GPU Occupancy (Section 6)
- Memory Hierarchy and Access Patterns (Section 7)
- Sub-group sizes: 16 or 32 for most Xe GPUs; use `get_sub_group()` API to query and adapt at runtime.

---

## Lesson: Speeding Up Row-Scale Kernels (diag(A) * B)

### Symptom

The row-wise broadcast multiply kernel `C[i,j] = A[i] * B[i,j]` is correct but only about 0.62x as fast as `torch.compile()`.

### Root Causes

- Per-element integer division (`row = idx / M`) in the hot loop
- Scalar bf16 loads/stores and scalar conversion overhead
- Weak memory coalescing pattern across lanes

### Fix Pattern

1. Map one work-group to one row
- Use `row = item.get_group(0)`.
- Remove integer division from the inner loop entirely.

2. Process bf16 pairs as `uint32`
- Load/store two bf16 values at once.
- This doubles useful bytes per memory op with naturally aligned 32-bit accesses.

3. Use coalesced lane mapping along columns
- Let consecutive local IDs handle consecutive pair indices.
- Stride by work-group size in the column loop.

4. Use explicit sub-group size on Xe
- Add `[[sycl::reqd_sub_group_size(16)]]` for predictable lane behavior.

5. Convert bf16 manually with bit operations
- `bf16 -> f32`: left-shift by 16 and bit-cast.
- `f32 -> bf16`: round-to-nearest-even using bias and sticky-bit logic.

### Minimal Kernel Structure

```cpp
const int pairs_per_row = M / 2;
enum { WG = 256 };

q.submit([&](sycl::handler& h) {
    h.parallel_for<class RowScaleKernel>(
        sycl::nd_range<1>((size_t)N * WG, WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            int row = item.get_group(0);
            int lid = item.get_local_id(0);

            float a_val = bf16_to_f32(a[row]);
            const uint32_t* b_row = b32 + row * pairs_per_row;
            uint32_t* c_row = c32 + row * pairs_per_row;

            for (int pi = lid; pi < pairs_per_row; pi += WG) {
                uint32_t p = b_row[pi];
                c_row[pi] = mul_pair_bf16(a_val, p);
            }
        });
});
```

### Measured Impact

- Versus eager PyTorch: improved to about 1.46x
- Versus `torch.compile()`: improved from about 0.62x to about 0.93x

### oneAPI Guide Alignment

- Section 6: Thread mapping and occupancy (work-group decomposition)
- Section 7: Memory hierarchy and coalesced access
- Section 11: Prefer wide/coalesced accesses and remove expensive operations from inner loops
