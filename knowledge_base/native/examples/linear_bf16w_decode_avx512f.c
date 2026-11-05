/* Example: written by an Xe-Forge session; verified in an engine against its reference. */
#include "xf.h"
#include <immintrin.h>

static inline __m512 ld_bf16(const uint16_t *p)
{
    __m256i h = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

static inline __m512 ld_bf16_mask(const uint16_t *p, __mmask16 mk)
{
    __m256i h = _mm256_maskz_loadu_epi16(mk, p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

static inline void rows4(const float *xm, const uint16_t *w, float *ym, int64_t K)
{
    const uint16_t *w0 = w, *w1 = w + K, *w2 = w + 2 * K, *w3 = w + 3 * K;
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m512 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
        __m512 x0 = _mm512_loadu_ps(xm + k), x1 = _mm512_loadu_ps(xm + k + 16);
        a0 = _mm512_fmadd_ps(x0, ld_bf16(w0 + k), a0);
        a1 = _mm512_fmadd_ps(x0, ld_bf16(w1 + k), a1);
        a2 = _mm512_fmadd_ps(x0, ld_bf16(w2 + k), a2);
        a3 = _mm512_fmadd_ps(x0, ld_bf16(w3 + k), a3);
        b0 = _mm512_fmadd_ps(x1, ld_bf16(w0 + k + 16), b0);
        b1 = _mm512_fmadd_ps(x1, ld_bf16(w1 + k + 16), b1);
        b2 = _mm512_fmadd_ps(x1, ld_bf16(w2 + k + 16), b2);
        b3 = _mm512_fmadd_ps(x1, ld_bf16(w3 + k + 16), b3);
    }
    for (; k < K; k += 16) {
        int64_t r = K - k;
        __mmask16 mk = r >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << r) - 1);
        __m512 x0 = _mm512_maskz_loadu_ps(mk, xm + k);
        a0 = _mm512_fmadd_ps(x0, ld_bf16_mask(w0 + k, mk), a0);
        a1 = _mm512_fmadd_ps(x0, ld_bf16_mask(w1 + k, mk), a1);
        a2 = _mm512_fmadd_ps(x0, ld_bf16_mask(w2 + k, mk), a2);
        a3 = _mm512_fmadd_ps(x0, ld_bf16_mask(w3 + k, mk), a3);
    }
    ym[0] = _mm512_reduce_add_ps(_mm512_add_ps(a0, b0));
    ym[1] = _mm512_reduce_add_ps(_mm512_add_ps(a1, b1));
    ym[2] = _mm512_reduce_add_ps(_mm512_add_ps(a2, b2));
    ym[3] = _mm512_reduce_add_ps(_mm512_add_ps(a3, b3));
}

static inline float row1(const float *xm, const uint16_t *w, int64_t K)
{
    __m512 a = _mm512_setzero_ps(), b = a;
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
        a = _mm512_fmadd_ps(_mm512_loadu_ps(xm + k), ld_bf16(w + k), a);
        b = _mm512_fmadd_ps(_mm512_loadu_ps(xm + k + 16), ld_bf16(w + k + 16), b);
    }
    for (; k < K; k += 16) {
        int64_t r = K - k;
        __mmask16 mk = r >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << r) - 1);
        a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mk, xm + k), ld_bf16_mask(w + k, mk), a);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a, b));
}

void xf_linear_bf16w_decode_avx512f(const float *x, const uint16_t *w, float *y,
                int64_t M, int64_t N, int64_t K, const xf_ctx *ctx)
{
    int64_t nb = (N + 3) / 4;
    #pragma omp parallel for schedule(static) num_threads(ctx->nthreads)
    for (int64_t b = 0; b < nb; b++) {
        int64_t n = b * 4;
        if (n + 4 <= N) {
            for (int64_t m = 0; m < M; m++) {
                float t[4];
                rows4(x + m * K, w + n * K, t, K);
                for (int j = 0; j < 4; j++) y[m * N + n + j] = t[j];
            }
        } else {
            for (int64_t nn = n; nn < N; nn++)
                for (int64_t m = 0; m < M; m++)
                    y[m * N + nn] = row1(x + m * K, w + nn * K, K);
        }
    }
}
