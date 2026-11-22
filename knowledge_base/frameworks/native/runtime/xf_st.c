/* safetensors: index every *.safetensors file of a directory; pread on demand. */
#include "xf.h"
#include "xf_internal.h"
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct xf_st { int nfiles; int *fd; int n, cap; xf_st_tensor *t; };

static const char *ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++; return p; }

static const char *str(const char *p, char *out, size_t cap)
{
    if (*p != '"') xf_fatal("safetensors header: expected a string");
    size_t n = 0;
    for (p++; *p && *p != '"'; p++) {
        if (*p == '\\' && p[1]) p++;
        if (n + 1 < cap) out[n++] = *p;
    }
    if (*p != '"') xf_fatal("safetensors header: unterminated string");
    out[n] = 0;
    return p + 1;
}

static const char *skip_value(const char *p)
{
    char buf[8];
    p = ws(p);
    if (*p == '"') return str(p, buf, sizeof buf);
    if (*p == '{' || *p == '[') {
        int depth = 0;
        for (; *p; p++) {
            if (*p == '"') { p = str(p, buf, sizeof buf) - 1; continue; }
            if (*p == '{' || *p == '[') depth++;
            if (*p == '}' || *p == ']') { if (--depth == 0) return p + 1; }
        }
        xf_fatal("safetensors header: unbalanced");
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

static const char *ints(const char *p, int64_t *v, int max, int *n)
{
    p = ws(p);
    if (*p != '[') xf_fatal("safetensors header: expected a list");
    *n = 0;
    for (p = ws(p + 1); *p != ']'; p = ws(p)) {
        char *end;
        long long x = strtoll(p, &end, 10);
        if (end == p) xf_fatal("safetensors header: expected an integer");
        if (*n < max) v[*n] = x;
        (*n)++;
        p = ws(end);
        if (*p == ',') p++;
    }
    return p + 1;
}

static int dtype_of(const char *s)
{
    if (!strcmp(s, "F32")) return XF_ST_F32;
    if (!strcmp(s, "BF16")) return XF_ST_BF16;
    if (!strcmp(s, "F16")) return XF_ST_F16;
    if (!strcmp(s, "U8")) return XF_ST_U8;
    if (!strcmp(s, "I8")) return XF_ST_I8;
    return XF_ST_OTHER;
}

static void index_file(xf_st *st, int file, const char *path)
{
    int fd = st->fd[file];
    uint64_t len;
    if (pread(fd, &len, 8, 0) != 8) xf_fatal("%s: not a safetensors file", path);
    char *h = malloc(len + 1);
    if (!h || pread(fd, h, len, 8) != (ssize_t)len) xf_fatal("%s: header unreadable", path);
    h[len] = 0;
    const char *p = ws(h);
    if (*p != '{') xf_fatal("%s: header is not an object", path);
    for (p = ws(p + 1); *p && *p != '}'; p = ws(p)) {
        char key[512];
        p = ws(str(p, key, sizeof key));
        if (*p++ != ':') xf_fatal("%s: header syntax", path);
        p = ws(p);
        if (!strcmp(key, "__metadata__")) { p = ws(skip_value(p)); if (*p == ',') p++; continue; }
        if (st->n == st->cap) {
            st->cap = st->cap ? 2 * st->cap : 1024;
            st->t = realloc(st->t, (size_t)st->cap * sizeof *st->t);
        }
        xf_st_tensor *t = &st->t[st->n++];
        memset(t, 0, sizeof *t);
        t->name = strdup(key);
        t->file = file;
        if (*p++ != '{') xf_fatal("%s: tensor %s is not an object", path, key);
        for (p = ws(p); *p != '}'; p = ws(p)) {
            char field[32], val[32];
            int n;
            int64_t off[2];
            p = ws(str(p, field, sizeof field));
            p = ws(p + 1);
            if (!strcmp(field, "dtype")) { p = str(p, val, sizeof val); t->dtype = dtype_of(val); }
            else if (!strcmp(field, "shape")) { p = ints(p, t->shape, 4, &n); t->ndim = n; }
            else if (!strcmp(field, "data_offsets")) {
                p = ints(p, off, 2, &n);
                t->offset = 8 + (int64_t)len + off[0];
                t->nbytes = off[1] - off[0];
            } else p = skip_value(p);
            p = ws(p);
            if (*p == ',') p++;
        }
        p = ws(p + 1);
        if (*p == ',') p++;
    }
    free(h);
}

xf_st *xf_st_open_dir(const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) xf_fatal("cannot open weights directory %s", dir);
    xf_st *st = calloc(1, sizeof *st);
    struct dirent *e;
    char path[4096];
    while ((e = readdir(d))) {
        size_t l = strlen(e->d_name);
        if (l < 12 || strcmp(e->d_name + l - 12, ".safetensors")) continue;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        st->fd = realloc(st->fd, (size_t)(st->nfiles + 1) * sizeof(int));
        st->fd[st->nfiles] = open(path, O_RDONLY);
        if (st->fd[st->nfiles] < 0) xf_fatal("cannot open %s", path);
        index_file(st, st->nfiles++, path);
    }
    closedir(d);
    if (!st->nfiles) xf_fatal("no .safetensors file in %s", dir);
    return st;
}

const xf_st_tensor *xf_st_find(const xf_st *st, const char *name)
{
    for (int i = 0; i < st->n; i++)
        if (!strcmp(st->t[i].name, name)) return &st->t[i];
    return NULL;
}

void xf_st_read(const xf_st *st, const xf_st_tensor *t, void *dst)
{
    char *p = dst;
    int64_t done = 0;
    while (done < t->nbytes) {
        ssize_t r = pread(st->fd[t->file], p + done, (size_t)(t->nbytes - done), t->offset + done);
        if (r <= 0) xf_fatal("short read of %s", t->name);
        done += r;
    }
}

void xf_st_close(xf_st *st)
{
    for (int i = 0; i < st->nfiles; i++) close(st->fd[i]);
    for (int i = 0; i < st->n; i++) free(st->t[i].name);
    free(st->t);
    free(st->fd);
    free(st);
}
