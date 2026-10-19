# Python kernel guidelines

These instructions apply to the cuTile kernel modules in this directory. The target is a small-memory SM120 GPU, with kernels exported to cubin for the C++ runtime. Keep the kernel correct and measurable before treating an optimization as complete.

## Module responsibilities

- Keep the cuTile kernels, their kernel-specific decode or quantization helpers, lightweight Python launch wrappers, and kernel-specific benchmark case definitions in the kernel module.
- Put reusable GGUF loading, autotuner execution, result reporting, and test reference code in utilities outside this directory. The Python launch wrappers support development and tests; C++ owns production launches and memory management.
- Keep separate activation quantization and matmul kernels unless fusion is explicitly requested. A standalone decoder may be retained for correctness checks even when the matmul decodes weights inside its tile.

## Interfaces and data layout

- Name parameters for the data they contain. Derive sizes and loop bounds from tensor shapes when possible; avoid redundant arguments, unexplained abbreviations, and implicit copies or layout conversions.
- State the required dtype, shape, stride, and packed-block layout at the interface. Support valid strided inputs when the kernel can read them directly.
- Use `ct.cdiv` for grid sizing and respect cuTile's power-of-two tile-size requirements. Keep tile sizes that need specialization as compile-time constants.

## Packed data and computation

- For packed-weight matmul, load a packed tile, decode it within the working tile, and consume the decoded values directly in the matmul when practical.
- Use contiguous tile loads for contiguous packed fields when possible. Use `pack_to_bytes` / `unpack_from_bytes` for byte-width reinterpretation and same-width `bitcast` where appropriate; verify the generated instructions for performance claims.
- Pre-encode exact lookup-table values when this removes per-element conversion work. Upload small device tables lazily and reuse them. Document packed-block composition next to the decoder.

## Autotuning and maintained configurations

- Use the benchmark utility's cuTile autotuner on the **current target GPU**, with the **real model tensor shape** and runtime input layout. Tune representative `medium` (**4 tokens**) and `large` (**128 tokens**) cases separately. If a matmul consumes prequantized activations, prepare them before timing that matmul.
- Search relevant tile dimensions, TMA choices for loads and stores, and occupancy hints. Check cubin/SASS when an optimization depends on a specific instruction, such as SM120 QMMA or TMA. Exclude a configuration only after measurements or generated code show why it is unsuitable.
- Manually copy each measured winner, including all tuned options, into a `BEST_CONFIGS` variable in the kernel module. The top-level key is the GPU SM version (for example, `sm120`, obtained from CuPy's compute capability); the next level contains `medium` and `large`. The autotuner reports results but never edits `BEST_CONFIGS` automatically.
- Have the launch wrapper select from `BEST_CONFIGS` using the current GPU and token count. Make the token-size boundary explicit, and report an unsupported SM instead of silently using a configuration measured on another GPU. Treat the table as shape- and hardware-specific: retune when either changes materially.

## Validation

- Compare decoding, quantization, and final output against independent references with tolerances appropriate to the quantization paths. Keep reusable reference logic in test utilities.
- Compile representative specializations to cubin and inspect the generated SASS when claiming tensor-core or TMA use. Run meaningful GPU correctness checks after changing kernel behavior or launch selection.
