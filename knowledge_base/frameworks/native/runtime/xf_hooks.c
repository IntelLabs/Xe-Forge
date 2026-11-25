/* XF_DUMP=dir: step outputs of the first XF_DUMP_CALLS (default 1) forward calls as raw fp32
   plus dir/index.jsonl.  XF_PROFILE=file: one JSON row per (block, step, T) with summed time. */
#include "xf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct { char key[160]; char op[32]; int64_t T, calls; double us; } xf_prof_row;

struct xf_hooks {
    const char *dump_dir, *prof_path;
    FILE *index;
    int dump_calls, call;
    int64_t T;
    double last;
    xf_prof_row *rows;
    int nrows, caprows;
};

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

xf_hooks *xf_hooks_from_env(void)
{
    const char *dump = getenv("XF_DUMP"), *prof = getenv("XF_PROFILE");
    if (!dump && !prof) return NULL;
    xf_hooks *h = calloc(1, sizeof *h);
    h->dump_dir = dump;
    h->prof_path = prof;
    h->dump_calls = getenv("XF_DUMP_CALLS") ? atoi(getenv("XF_DUMP_CALLS")) : 1;
    if (dump) {
        char path[4096];
        mkdir(dump, 0755);
        snprintf(path, sizeof path, "%s/index.jsonl", dump);
        h->index = fopen(path, "w");
        if (!h->index) xf_fatal("cannot write %s", path);
    }
    return h;
}

void xf_hook_begin(xf_hooks *h, int64_t T, int64_t pos)
{
    (void)pos;
    if (!h) return;
    h->T = T;
    h->last = now_us();
}

void xf_hook_step(xf_hooks *h, const char *block, int64_t layer, const char *step,
                  const char *op, const char *module, const float *y, int64_t numel, int64_t rows)
{
    if (!h) return;
    double t = now_us();
    if (h->prof_path) {
        char key[160];
        snprintf(key, sizeof key, "%s.%s", block, step);
        int i = 0;
        while (i < h->nrows && !(h->rows[i].T == h->T && !strcmp(h->rows[i].key, key))) i++;
        if (i == h->nrows) {
            if (h->nrows == h->caprows) {
                h->caprows = h->caprows ? 2 * h->caprows : 64;
                h->rows = realloc(h->rows, (size_t)h->caprows * sizeof *h->rows);
            }
            memset(&h->rows[i], 0, sizeof h->rows[i]);
            snprintf(h->rows[i].key, sizeof h->rows[i].key, "%s", key);
            snprintf(h->rows[i].op, sizeof h->rows[i].op, "%s", op);
            h->rows[i].T = h->T;
            h->nrows++;
        }
        h->rows[i].calls++;
        h->rows[i].us += t - h->last;
    }
    if (h->dump_dir && y && h->call < h->dump_calls) {
        char name[512], path[4096];
        snprintf(name, sizeof name, "c%d.%s.%lld.%s.f32", h->call, block, (long long)layer, step);
        snprintf(path, sizeof path, "%s/%s", h->dump_dir, name);
        FILE *f = fopen(path, "wb");
        if (!f || fwrite(y, sizeof(float), (size_t)numel, f) != (size_t)numel) xf_fatal("cannot write %s", path);
        fclose(f);
        fprintf(h->index, "{\"call\": %d, \"block\": \"%s\", \"layer\": %lld, \"step\": \"%s\", \"op\": \"%s\", "
                "\"module\": \"%s\", \"rows\": %lld, \"numel\": %lld, \"file\": \"%s\"}\n",
                h->call, block, (long long)layer, step, op, module, (long long)rows, (long long)numel, name);
    }
    h->last = now_us();   /* the dump is not charged to the next step */
}

void xf_hook_end(xf_hooks *h)
{
    if (h) h->call++;
}

void xf_hooks_close(xf_hooks *h)
{
    if (!h) return;
    if (h->index) fclose(h->index);
    if (h->prof_path) {
        FILE *f = fopen(h->prof_path, "w");
        if (!f) xf_fatal("cannot write %s", h->prof_path);
        for (int i = 0; i < h->nrows; i++)
            fprintf(f, "{\"step\": \"%s\", \"op\": \"%s\", \"T\": %lld, \"calls\": %lld, \"us\": %.3f}\n",
                    h->rows[i].key, h->rows[i].op, (long long)h->rows[i].T, (long long)h->rows[i].calls, h->rows[i].us);
        fclose(f);
    }
    free(h->rows);
    free(h);
}
