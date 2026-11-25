/* The engine's shared header: context, runtime services, model API. Add kernel prototypes. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct xf_hooks xf_hooks;

typedef struct {          /* passed to every kernel; read-only for kernels */
    int nthreads;         /* kernels never set thread counts themselves */
    uint32_t isa;
    xf_hooks *hooks;      /* NULL unless XF_DUMP or XF_PROFILE is set */
} xf_ctx;

void xf_fatal(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));

/* bf16 is stored as its upper 16 bits; widening is exact. */
static inline float xf_bf16_to_f32(uint16_t v)
{
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
float *xf_scratch(int64_t n);   /* per-thread temporary of n floats, kept between calls */

/* safetensors (xf_st.c) */
typedef struct xf_st xf_st;
xf_st *xf_st_open_dir(const char *dir);
void xf_st_close(xf_st *st);

/* weight binding (xf_bind.c): every entry loaded, shape-checked and stored as its dtype --
   fp32, bf16 where every kernel reading it takes bf16, or raw bytes for a packed format */
enum { XF_F32 = 0, XF_BF16 = 1, XF_U8 = 2 };   /* XF_U8: the checkpoint's bytes, as stored */
/* count > 1: a stack of count tensors, name a printf pattern of the item index, bound as one
   contiguous [count, *shape] buffer */
typedef struct { const char *name; int ndim; int64_t shape[4]; int dtype; int count; } xf_weight_spec;
void **xf_bind(xf_st *st, const xf_weight_spec *spec, int n);
void xf_unbind(void **w, int n);

/* hooks (xf_hooks.c): XF_DUMP=dir writes step outputs, XF_PROFILE=file per-step times */
xf_hooks *xf_hooks_from_env(void);
void xf_hooks_close(xf_hooks *h);
void xf_hook_begin(xf_hooks *h, int64_t T, int64_t pos);
void xf_hook_step(xf_hooks *h, const char *block, int64_t layer, const char *step,
                  const char *op, const char *module, const float *y, int64_t numel, int64_t rows);
void xf_hook_end(xf_hooks *h);

/* kernels: one prototype per op the engine implements */

/* model (model.c) */
typedef struct xf_model xf_model;
xf_model *xf_model_load(const char *weights_dir, int nthreads);
const float *xf_forward(xf_model *m, const int32_t *ids, int64_t T, int64_t pos);
void xf_reset(xf_model *m);
void xf_model_free(xf_model *m);
int64_t xf_model_ctx(void);
int64_t xf_model_out_numel(void);
