/* xf-run WEIGHTS_DIR --ids 1,2,3|@FILE [--new N] [--recompute] [--logits FILE] [--threads N]
   Greedy decode. Prints TOKENS, PREFILL_MS, DECODE_MS_PER_TOKEN, then DONE. */
#include "xf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int32_t argmax(const float *v, int64_t n)
{
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return (int32_t)best;
}

int main(int argc, char **argv)
{
    const char *weights = NULL, *ids_arg = NULL, *logits_path = NULL;
    int new_tokens = 1, recompute = 0, threads = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_arg = argv[++i];
        else if (!strcmp(argv[i], "--new") && i + 1 < argc) new_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logits_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--recompute")) recompute = 1;
        else if (argv[i][0] != '-' && !weights) weights = argv[i];
        else xf_fatal("usage: %s WEIGHTS_DIR --ids 1,2,3 [--new N] [--recompute] [--logits FILE] [--threads N]", argv[0]);
    }
    if (!weights || !ids_arg) xf_fatal("usage: %s WEIGHTS_DIR --ids 1,2,3 [--new N]", argv[0]);
    int64_t ctx = xf_model_ctx(), nout = xf_model_out_numel();
    int32_t *seq = calloc((size_t)ctx, sizeof *seq);
    int64_t n = 0;
    /* --ids @FILE reads the comma-separated ids from FILE: a long prompt does not fit in one argument */
    char *ids_text = NULL;
    if (ids_arg[0] == '@') {
        FILE *f = fopen(ids_arg + 1, "rb");
        if (!f) xf_fatal("cannot read %s", ids_arg + 1);
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        ids_text = calloc((size_t)len + 1, 1);
        if (!ids_text || fread(ids_text, 1, (size_t)len, f) != (size_t)len) xf_fatal("cannot read %s", ids_arg + 1);
        fclose(f);
    }
    for (char *s = ids_text ? ids_text : strdup(ids_arg), *tok = strtok(s, ",\n"); tok; tok = strtok(NULL, ",\n")) {
        if (n == ctx) xf_fatal("prompt longer than the engine's context (%lld)", (long long)ctx);
        seq[n++] = atoi(tok);
    }
    if (n + new_tokens > ctx) xf_fatal("prompt + new tokens exceed the context (%lld)", (long long)ctx);
    FILE *lf = logits_path ? fopen(logits_path, "wb") : NULL;
    if (logits_path && !lf) xf_fatal("cannot write %s", logits_path);

    double t0 = now_ms();
    xf_model *m = xf_model_load(weights, threads);
    printf("LOAD_MS: %.1f\n", now_ms() - t0);
    int64_t prompt = n;
    double prefill_ms = 0, decode_ms = 0;
    for (int step = 0; step < new_tokens; step++) {
        double s = now_ms();
        const float *logits;
        if (recompute) { xf_reset(m); logits = xf_forward(m, seq, n, 0); }
        else if (step == 0) logits = xf_forward(m, seq, n, 0);
        else logits = xf_forward(m, seq + n - 1, 1, n - 1);
        double e = now_ms() - s;
        if (step == 0) prefill_ms = e; else decode_ms += e;
        if (lf) fwrite(logits, sizeof(float), (size_t)nout, lf);
        seq[n++] = argmax(logits, nout);
    }
    if (lf) fclose(lf);
    printf("TOKENS:");
    for (int64_t i = prompt; i < n; i++) printf(" %d", seq[i]);
    printf("\nPREFILL_MS: %.3f\n", prefill_ms);
    if (new_tokens > 1) printf("DECODE_MS_PER_TOKEN: %.3f\n", decode_ms / (new_tokens - 1));
    xf_model_free(m);
    printf("DONE\n");
    return 0;
}
