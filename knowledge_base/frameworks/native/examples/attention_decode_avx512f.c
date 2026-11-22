/* Example: written by an Xe-Forge session; verified in an engine against its reference. */
#include "xf.h"
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define KB 16
#define SPLIT_MIN 8
#define GB 2

static inline __m512 xexp2(__m512 x)
{
    __mmask16 live = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-125.f), _CMP_GE_OQ);
    x = _mm512_max_ps(x, _mm512_set1_ps(-125.f));
    __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(x, n);
    __m512 p = _mm512_set1_ps(1.535336188319500e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.339887440266574e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.618437357674640e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.550332471162809e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.402264791363012e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.931472028550421e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_maskz_scalef_ps(live, p, n);
}

static inline __mmask16 tailm(int64_t r)
{
    return r >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << r) - 1);
}

/* One key chunk [j0, j1) of kv head kvh for G query heads h0 .. h0+G-1 of token t.
 * Writes per head [m, l, pad.., acc[D]] at pp + g * GS, m in log2 units. */
static inline __attribute__((always_inline)) void chunk_run(
    const float *q, const float *k_cache, const float *v_cache, int64_t H, int64_t KVH,
    const int64_t D, const int G, int64_t t, int64_t kvh, int64_t h0, int64_t j0, int64_t j1,
    float qs, float *pp, int64_t GS)
{
    const float *qv[GB];
    float *acc[GB];
    float m[GB], l[GB];
    for (int g = 0; g < G; g++) {
        qv[g] = q + (t * H + h0 + g) * D;
        acc[g] = pp + g * GS + 16;
        m[g] = -INFINITY;
        l[g] = 0.f;
        for (int64_t i = 0; i < D; i += 16) _mm512_mask_storeu_ps(acc[g] + i, tailm(D - i), _mm512_setzero_ps());
    }
    for (int64_t jb = j0; jb < j1; jb += KB) {
        int64_t kc = j1 - jb < KB ? j1 - jb : KB;
        float s[GB][KB];
        for (int64_t jj = 0; jj < kc; jj++) {
            const float *kr = k_cache + ((jb + jj) * KVH + kvh) * D;
            __m512 a0[GB], a1[GB];
            for (int g = 0; g < G; g++) a0[g] = a1[g] = _mm512_setzero_ps();
            int64_t i = 0;
            for (; i + 32 <= D; i += 32) {
                __m512 k0 = _mm512_loadu_ps(kr + i), k1 = _mm512_loadu_ps(kr + i + 16);
                for (int g = 0; g < G; g++) {
                    a0[g] = _mm512_fmadd_ps(_mm512_loadu_ps(qv[g] + i), k0, a0[g]);
                    a1[g] = _mm512_fmadd_ps(_mm512_loadu_ps(qv[g] + i + 16), k1, a1[g]);
                }
            }
            for (; i < D; i += 16) {
                __mmask16 mk = tailm(D - i);
                __m512 k0 = _mm512_maskz_loadu_ps(mk, kr + i);
                for (int g = 0; g < G; g++)
                    a0[g] = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mk, qv[g] + i), k0, a0[g]);
            }
            for (int g = 0; g < G; g++) s[g][jj] = _mm512_reduce_add_ps(_mm512_add_ps(a0[g], a1[g]));
        }
        __mmask16 live = tailm(kc);
        float p[GB][KB];
        float alpha[GB];
        for (int g = 0; g < G; g++) {
            __m512 sv = _mm512_mul_ps(_mm512_maskz_loadu_ps(live, s[g]), _mm512_set1_ps(qs));
            sv = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), live, sv);
            float bm = _mm512_reduce_max_ps(sv);
            float mn = bm > m[g] ? bm : m[g];
            __m512 e = xexp2(_mm512_sub_ps(_mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m[g]),
                                           _mm512_set1_ps(mn)));
            alpha[g] = _mm512_cvtss_f32(e);
            __m512 pv = xexp2(_mm512_sub_ps(sv, _mm512_set1_ps(mn)));
            _mm512_storeu_ps(p[g], pv);
            l[g] = l[g] * alpha[g] + _mm512_reduce_add_ps(pv);
            m[g] = mn;
        }
        for (int64_t i = 0; i < D; i += 16) {
            __mmask16 mk = tailm(D - i);
            __m512 o[GB];
            for (int g = 0; g < G; g++)
                o[g] = _mm512_mul_ps(_mm512_maskz_loadu_ps(mk, acc[g] + i), _mm512_set1_ps(alpha[g]));
            for (int64_t jj = 0; jj < kc; jj++) {
                __m512 vv = _mm512_maskz_loadu_ps(mk, v_cache + ((jb + jj) * KVH + kvh) * D + i);
                for (int g = 0; g < G; g++) o[g] = _mm512_fmadd_ps(_mm512_set1_ps(p[g][jj]), vv, o[g]);
            }
            for (int g = 0; g < G; g++) _mm512_mask_storeu_ps(acc[g] + i, mk, o[g]);
        }
    }
    for (int g = 0; g < G; g++) {
        pp[g * GS] = m[g];
        pp[g * GS + 1] = l[g];
    }
}

static void chunk_any(const float *q, const float *k_cache, const float *v_cache, int64_t H,
                      int64_t KVH, int64_t D, int G, int64_t t, int64_t kvh, int64_t h0,
                      int64_t j0, int64_t j1, float qs, float *pp, int64_t GS)
{
    if (D == 128 && G == 2)
        chunk_run(q, k_cache, v_cache, H, KVH, 128, 2, t, kvh, h0, j0, j1, qs, pp, GS);
    else if (G == 2)
        chunk_run(q, k_cache, v_cache, H, KVH, D, 2, t, kvh, h0, j0, j1, qs, pp, GS);
    else
        chunk_run(q, k_cache, v_cache, H, KVH, D, 1, t, kvh, h0, j0, j1, qs, pp, GS);
}

void xf_attention_decode_avx512f(const float *q, float *k_cache, float *v_cache, float *y,
                int64_t T, int64_t POS, int64_t CTX, int64_t H, int64_t KVH, int64_t D,
                float scale, const xf_ctx *ctx)
{
    if (POS + T > CTX) xf_fatal("attention: %lld + %lld rows exceed the cache (%lld)",
                                (long long)POS, (long long)T, (long long)CTX);
    if (T <= 0) return;
    int64_t group = H / KVH;
    int nth = ctx->nthreads;
    int64_t nmax = POS + T;
    int64_t S = nth / (2 * T * KVH);
    if (S < 1) S = 1;
    if (S > (nmax + SPLIT_MIN - 1) / SPLIT_MIN) S = (nmax + SPLIT_MIN - 1) / SPLIT_MIN;
    if (S < 1) S = 1;
    int64_t GS = 16 + ((D + 15) & ~(int64_t)15);
    int64_t ntask = T * KVH * S;
    float *part = xf_scratch(T * H * S * GS + T * KVH * 16 + 16);
    int *cnt = (int *)(part + T * H * S * GS);
    for (int64_t i = 0; i < T * KVH; i++) cnt[i * 16] = 0;
    float qs = scale * 1.4426950408889634f;

    #pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num();
        int64_t stride = ntask * 2 <= nth ? 2 : 1;
        for (int64_t task = tid % stride ? ntask : tid / stride; task < ntask; task += nth / stride) {
            int64_t sp = task % S, tk = task / S, kvh = tk % KVH, t = tk / KVH;
            int64_t n = POS + t + 1;
            int64_t chunk = (n + S - 1) / S;
            int64_t j0 = sp * chunk, j1 = j0 + chunk < n ? j0 + chunk : n;
            if (j0 > j1) j0 = j1;
            for (int64_t j = j0; j < j1; j++) {
                const char *kr = (const char *)(k_cache + (j * KVH + kvh) * D);
                for (int64_t o = 0; o < D * 4; o += 64) _mm_prefetch(kr + o, _MM_HINT_T0);
            }
            for (int64_t j = j0; j < j1; j++) {
                const char *vr = (const char *)(v_cache + (j * KVH + kvh) * D);
                for (int64_t o = 0; o < D * 4; o += 64) _mm_prefetch(vr + o, _MM_HINT_T0);
            }
            for (int64_t g0 = 0; g0 < group; g0 += GB) {
                int G = group - g0 < GB ? (int)(group - g0) : GB;
                chunk_any(q, k_cache, v_cache, H, KVH, D, G, t, kvh, kvh * group + g0, j0, j1, qs,
                          part + ((t * H + kvh * group + g0) * S + sp) * GS, S * GS);
            }
            if (__atomic_add_fetch(&cnt[(t * KVH + kvh) * 16], 1, __ATOMIC_ACQ_REL) != S) continue;
            for (int64_t g = 0; g < group; g++) {
                int64_t r = t * H + kvh * group + g;
                const float *pr = part + r * S * GS;
                float M = -INFINITY;
                for (int64_t s2 = 0; s2 < S; s2++) if (pr[s2 * GS] > M) M = pr[s2 * GS];
                float w[S];
                float L = 0.f;
                for (int64_t s2 = 0; s2 < S; s2++) {
                    w[s2] = pr[s2 * GS] == -INFINITY ? 0.f : exp2f(pr[s2 * GS] - M);
                    L += w[s2] * pr[s2 * GS + 1];
                }
                float inv = 1.f / L;
                for (int64_t i = 0; i < D; i += 16) {
                    __mmask16 mk = tailm(D - i);
                    __m512 o = _mm512_setzero_ps();
                    for (int64_t s2 = 0; s2 < S; s2++)
                        o = _mm512_fmadd_ps(_mm512_set1_ps(w[s2]), _mm512_maskz_loadu_ps(mk, pr + s2 * GS + 16 + i), o);
                    _mm512_mask_storeu_ps(y + r * D + i, mk, _mm512_mul_ps(o, _mm512_set1_ps(inv)));
                }
            }
        }
    }
}
