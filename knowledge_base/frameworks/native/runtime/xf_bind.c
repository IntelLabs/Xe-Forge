/* Bind the generated weight table: find, check shape against the config, widen to fp32. */
#include "xf.h"
#include "xf_internal.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void xf_fatal(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("xf: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(2);
}

float *xf_scratch(int64_t n)
{
    static _Thread_local float *buf;
    static _Thread_local int64_t cap;
    if (n > cap) {
        free(buf);
        buf = aligned_alloc(64, (size_t)((n + 15) / 16 * 16) * sizeof(float));
        if (!buf) xf_fatal("out of memory (scratch of %lld floats)", (long long)n);
        cap = (n + 15) / 16 * 16;
    }
    return buf;
}

static float half_to_float(uint16_t h)
{
    int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(m | 1024), e - 25);
    return s ? -v : v;
}

static uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);  /* NaN stays NaN */
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);                /* nearest even */
}

/* item e of a weight: its checkpoint name (a printf pattern) resolved */
static void item_name(const xf_weight_spec *sp, int e, char *out, size_t cap)
{
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wformat-nonliteral"
    snprintf(out, cap, sp->name, e);
    #pragma GCC diagnostic pop
}

/* one checkpoint tensor into dst, stored as dtype */
static void load_item(const xf_st *st, const xf_st_tensor *t, int dtype, void *dst, int64_t numel)
{
    if (dtype == XF_U8 || (dtype == XF_BF16 && t->dtype == XF_ST_BF16) ||
        (dtype == XF_F32 && t->dtype == XF_ST_F32)) {
        xf_st_read(st, t, dst);   /* as the checkpoint holds it: exact (a packed format the kernel decodes) */
        return;
    }
    void *raw = malloc((size_t)t->nbytes);
    if (!raw) xf_fatal("out of memory binding %s", t->name);
    xf_st_read(st, t, raw);
    for (int64_t k = 0; k < numel; k++) {
        float v = t->dtype == XF_ST_F32 ? ((float *)raw)[k]
                : t->dtype == XF_ST_BF16 ? xf_bf16_to_f32(((uint16_t *)raw)[k])
                : half_to_float(((uint16_t *)raw)[k]);
        if (dtype == XF_BF16) ((uint16_t *)dst)[k] = f32_to_bf16(v);
        else ((float *)dst)[k] = v;
    }
    free(raw);
}

void **xf_bind(xf_st *st, const xf_weight_spec *spec, int n)
{
    void **w = calloc((size_t)n, sizeof *w);
    int bad = 0;
    char name[512];
    for (int i = 0; i < n; i++)
        for (int e = 0; e < spec[i].count; e++) {
            item_name(&spec[i], e, name, sizeof name);
            const xf_st_tensor *t = xf_st_find(st, name);
            if (!t) { fprintf(stderr, "xf: weight %s not in the checkpoint\n", name); bad = 1; continue; }
            int same = t->ndim == spec[i].ndim;
            for (int d = 0; same && d < t->ndim; d++) same = t->shape[d] == spec[i].shape[d];
            int bytes = t->dtype == XF_ST_U8 || t->dtype == XF_ST_I8;
            if (!same || t->dtype == XF_ST_OTHER || bytes != (spec[i].dtype == XF_U8)) {
                fprintf(stderr, "xf: weight %s has shape/dtype other than the model's\n", name);
                bad = 1;
            }
        }
    if (bad) xf_fatal("the checkpoint does not match this engine");
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        int64_t numel = 1;
        for (int d = 0; d < spec[i].ndim; d++) numel *= spec[i].shape[d];
        size_t esize = spec[i].dtype == XF_U8 ? 1 : spec[i].dtype == XF_BF16 ? 2 : 4;
        char *dst = aligned_alloc(64, (size_t)((numel * spec[i].count + 31) / 32 * 32) * esize);
        if (!dst) xf_fatal("out of memory binding %s", spec[i].name);
        for (int e = 0; e < spec[i].count; e++) {
            char nm[512];
            item_name(&spec[i], e, nm, sizeof nm);
            load_item(st, xf_st_find(st, nm), spec[i].dtype, dst + (size_t)e * (size_t)numel * esize, numel);
        }
        w[i] = dst;
    }
    return w;
}

void xf_unbind(void **w, int n)
{
    for (int i = 0; i < n; i++) free(w[i]);
    free(w);
}
