# <framework>

## 1. What it is

One sentence on what the framework is, with its upstream repository link.

## 2. Workloads

Which `--workload` values it runs (`text`, `vision`, `action`), and what kind of model each is.

## 3. Command

One complete command a person can copy, using only flags in `xe-forge run --help`:

```
xe-forge run --framework <framework> --model <hub id> \
    --framework-source <repository URL>@<tag> \
    --batch 4 --input-len 512 --output-len 32 --prompt-file <text file> \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove \
    -o runs/<name>
```

- `--framework-source`, `--framework-venv` or `--framework-python`: which fits, and why.
- `--framework-note`: only when the capture agent cannot discover a constraint itself; give
  the note and the reason.
- The workload's inputs: the flags of section 2's workload, and the form `--model` takes
  (a hub id, a local path, a converted model).
- `--reference-python`: when the framework's venv has no GPU torch and transformers for the
  patch check's reference.

## 4. What the setup agent builds

What `<venv>/xe-forge-setup.json` records: the source and how it was installed, what was
dropped (`dropped`) and why, the `env` and `prepend` additions, and the `notes` handed to the
capture agent. Say "generic" when nothing beyond the project's own install docs was needed.

## 5. Kernel repositories

The repositories the naming agent ends up in, so a person can clone them under
`--kernel-repos` ahead of the run: name, upstream link, what lives there (kernels in which DSL,
library calls), and which route they take (`kernel`, `library`, `call`).

## 6. Known limits

What the framework does that a run cannot see or change: ops it keeps on the CPU, compiled
paths the eager capture does not run, outputs the checks cannot judge. Only what is known.

## 7. Status

The status from the [index](README.md), in one line.

Last checked: <date>, Xe-Forge <short HEAD sha>
