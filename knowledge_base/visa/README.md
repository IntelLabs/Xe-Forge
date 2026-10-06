# vISA knowledge for AI lowering

Knowledge the `xe-forge --lower visa` agent may be given, selected per kernel by
`xe_forge.lowering.visa.retrieval` (which ops a Triton kernel uses decides which
entries it sees).

- `common/` -- the vISA language: file structure, declarations, operands and
  regions, execution masks and predicates, arithmetic, conversions, control flow,
  memory access, kernel arguments.
- `xe2/` -- what is particular to Xe2 (Battlemage): execution widths, the register
  file, LSC, DPAS, BF16/FP16, restrictions, finalizer behaviour.
- `examples/` -- complete, hand-written kernels, each with the Triton kernel it
  implements and a spec. An example is used only after it has been finalized and
  verified on a device (`tests/gpu/test_visa_examples.py`).

## Provenance

Entries are condensed from IGC's own vISA documentation, at the release the
finalizer is built from (`intel-graphics-compiler` tag `v2.41.5`,
`documentation/visa/`); each entry names its source section. Statements marked
*observed* were established on a Xe2 device with this finalizer.

**No entry or example is derived from compiler output.** No IGC dump of any
kernel, and no Triton intermediate form, was used to write anything here. The
lowering experiment depends on this: the agent must lower without having seen a
compiler lower the kernel. Examples are written by hand from the documentation
above and from the kernel ABI that Xe-Forge publishes in every contract.
