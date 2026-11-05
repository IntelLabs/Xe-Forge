/* Runtime-internal declarations; kernels never include this. */
#pragma once
#include "xf.h"

enum { XF_ST_OTHER = 0, XF_ST_F32, XF_ST_BF16, XF_ST_F16, XF_ST_U8, XF_ST_I8 };

typedef struct {
    char *name;
    int file, dtype, ndim;
    int64_t shape[4], offset, nbytes;
} xf_st_tensor;

const xf_st_tensor *xf_st_find(const xf_st *st, const char *name);
void xf_st_read(const xf_st *st, const xf_st_tensor *t, void *dst);
