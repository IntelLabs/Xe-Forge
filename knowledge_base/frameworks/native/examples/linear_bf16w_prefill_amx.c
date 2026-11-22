/* Example: written by an Xe-Forge session; verified in an engine against its reference. */
#include "xf.h"
#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

/* AMX bf16: y^T tile[n][m] += W[n][k] . X[k][m], x split into bf16 hi + lo.
   Threads split m into GM groups and n into nthreads/GM groups; each packs its own x rows. */

typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __attribute__((packed)) tilecfg;

#define GM 4  /* m groups */

/* dst row r = column r of the 16x16 u32 matrix v */
static inline void tr16_store(__m512i v[16], uint32_t *dst)
{
    __m512i t[16], u[16];
    for (int i = 0; i < 8; i++) {
        t[2 * i] = _mm512_unpacklo_epi32(v[2 * i], v[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(v[2 * i], v[2 * i + 1]);
    }
    for (int i = 0; i < 4; i++) {
        u[4 * i + 0] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        u[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    for (int j = 0; j < 4; j++) {
        const __m512i p0 = _mm512_shuffle_i32x4(u[j], u[4 + j], 0x44);
        const __m512i p1 = _mm512_shuffle_i32x4(u[j], u[4 + j], 0xEE);
        const __m512i q0 = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0x44);
        const __m512i q1 = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0xEE);
        _mm512_store_si512(dst + (0 + j) * 16, _mm512_shuffle_i32x4(p0, q0, 0x88));
        _mm512_store_si512(dst + (4 + j) * 16, _mm512_shuffle_i32x4(p0, q0, 0xDD));
        _mm512_store_si512(dst + (8 + j) * 16, _mm512_shuffle_i32x4(p1, q1, 0x88));
        _mm512_store_si512(dst + (12 + j) * 16, _mm512_shuffle_i32x4(p1, q1, 0xDD));
    }
}
#define PFD 8 /* W prefetch distance in k-chunks */

void xf_linear_bf16w_prefill_amx(const float *x, const uint16_t *w, float *y,
               int64_t M, int64_t N, int64_t K, const xf_ctx *ctx)
{
    syscall(SYS_arch_prctl, 0x1023, 18);
    const int64_t nkc = (K + 31) / 32;
    const int64_t nmp = (M + 31) / 32, nnp = (N + 31) / 32;
    const int64_t gm = nmp < GM ? nmp : GM;
    const int64_t gn = ctx->nthreads / gm;

    #pragma omp parallel num_threads(ctx->nthreads)
    {
        const int64_t tid = omp_get_thread_num();
        if (tid < gm * gn) {
            const int64_t im = tid % gm, in = tid / gm;
            const int64_t mp0 = im * nmp / gm, mp1 = (im + 1) * nmp / gm;
            const int64_t np0 = in * nnp / gn, np1 = (in + 1) * nnp / gn;
            const int64_t lmb = (mp1 - mp0) * 2;
            uint16_t *xp = (uint16_t *)xf_scratch(nkc * lmb * 512);

            if (np1 > np0) {
                for (int64_t t = 0; t < nkc * lmb; t++) {
                    const int64_t kc = t / lmb, mb = t % lmb, k0 = kc * 32;
                    uint32_t *hi = (uint32_t *)(xp + t * 2 * 512), *lo = hi + 256;
                    const int64_t kr = K - k0 < 32 ? K - k0 : 32;
                    const __mmask16 ma = kr >= 16 ? 0xFFFF : (__mmask16)((1u << kr) - 1);
                    const __mmask16 mb_ = kr >= 32 ? 0xFFFF : kr <= 16 ? 0 : (__mmask16)((1u << (kr - 16)) - 1);
                    __m512i vh[16], vl[16];
                    for (int64_t mi = 0; mi < 16; mi++) {
                        const int64_t m = mp0 * 32 + mb * 16 + mi;
                        __m512 a = _mm512_setzero_ps(), b = _mm512_setzero_ps();
                        if (m < M) {
                            a = _mm512_maskz_loadu_ps(ma, x + m * K + k0);
                            b = _mm512_maskz_loadu_ps(mb_, x + m * K + k0 + 16);
                        }
                        __m256bh ha = _mm512_cvtneps_pbh(a), hb = _mm512_cvtneps_pbh(b);
                        __m512 ra = _mm512_sub_ps(a, _mm512_cvtpbh_ps(ha));
                        __m512 rb = _mm512_sub_ps(b, _mm512_cvtpbh_ps(hb));
                        const __m512i h = _mm512_inserti64x4(_mm512_castsi256_si512((__m256i)ha), (__m256i)hb, 1);
                        const __m512i l = (__m512i)_mm512_cvtne2ps_pbh(rb, ra);
                        vh[mi] = h;
                        vl[mi] = l;
                    }
                    tr16_store(vh, hi);
                    tr16_store(vl, lo);
                }
            }

            tilecfg cfg;
            memset(&cfg, 0, sizeof cfg);
            cfg.palette_id = 1;
            for (int t = 0; t < 8; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
            _tile_loadconfig(&cfg);
            uint16_t apad[16 * 32] __attribute__((aligned(64)));
            float cbuf[16 * 16] __attribute__((aligned(64)));

#define LOAD_A(T, NB0, NR)                                                         \
            do {                                                                   \
                if ((NR) == 16 && k0 + 32 <= K) {                                  \
                    _tile_loadd(T, w + (NB0) * K + k0, K * 2);                     \
                } else {                                                           \
                    const int64_t kr = K - k0 < 32 ? K - k0 : 32;                  \
                    memset(apad, 0, sizeof apad);                                  \
                    for (int64_t i = 0; i < (NR); i++)                             \
                        memcpy(apad + i * 32, w + ((NB0) + i) * K + k0, kr * 2);   \
                    _tile_loadd(T, apad, 64);                                      \
                }                                                                  \
            } while (0)
#define STORE_C(T, NB0, NR, M0)                                                    \
            do {                                                                   \
                _tile_stored(T, cbuf, 64);                                         \
                for (int64_t j = 0; j < 16 && (M0) + j < M; j++)                   \
                    for (int64_t i = 0; i < (NR); i++)                             \
                        y[((M0) + j) * N + (NB0) + i] = cbuf[i * 16 + j];          \
            } while (0)

            for (int64_t np = np0; np < np1; np++) {
                const int64_t n0 = np * 32, n1 = n0 + 16;
                const int64_t r0 = N - n0 < 16 ? N - n0 : 16;
                const int64_t r1 = N - n1 < 0 ? 0 : N - n1 < 16 ? N - n1 : 16;
                const int64_t nrow = r0 + r1;
                for (int64_t mp = mp0; mp < mp1; mp++) {
                    const int64_t lb = (mp - mp0) * 2;
                    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                    for (int64_t kc = 0; kc < nkc; kc++) {
                        const int64_t k0 = kc * 32;
                        if (kc + PFD < nkc)
                            for (int64_t i = 0; i < nrow; i++)
                                _mm_prefetch((const char *)(w + (n0 + i) * K + k0 + 32 * PFD), _MM_HINT_T1);
                        LOAD_A(4, n0, r0);
                        LOAD_A(5, n1, r1);
                        const uint16_t *b = xp + ((kc * lmb + lb) * 2) * 512;
                        _tile_loadd(6, b + 0 * 512, 64);
                        _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(2, 5, 6);
                        _tile_loadd(7, b + 1 * 512, 64);
                        _tile_dpbf16ps(0, 4, 7); _tile_dpbf16ps(2, 5, 7);
                        _tile_loadd(6, b + 2 * 512, 64);
                        _tile_dpbf16ps(1, 4, 6); _tile_dpbf16ps(3, 5, 6);
                        _tile_loadd(7, b + 3 * 512, 64);
                        _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(3, 5, 7);
                    }
                    STORE_C(0, n0, r0, mp * 32);
                    STORE_C(1, n0, r0, mp * 32 + 16);
                    STORE_C(2, n1, r1, mp * 32);
                    STORE_C(3, n1, r1, mp * 32 + 16);
                }
            }
            _tile_release();
        }
    }
}
