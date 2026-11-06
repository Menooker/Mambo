# AGENTS

目标硬件是sm120 小显存GPU能跑通并且优化性能。使用cutile在python编写kernel，保证正确性。使用cutile 编译完cubin之后通过C++调用。这个项目的主体部分是C++，来进行kernel调用，内存管理等等

## Use of subagents

Actively delegate independent work such as compiler source research, CPU compilation and process supervision, IR / SASS audits, and independent reference and correctness checks.

- The primary agent owns task decomposition, integration, optimization selection, and the final correctness and performance report. Give each subagent a concrete question, input artifacts, completion criteria, and explicit file or directory ownership.
- Run independent source research and IR / SASS audits in parallel. Reuse agents with relevant context and share findings and artifacts to avoid duplicate compilation or investigation.
- Assign one explicit owner to serialize GPU correctness tests, autotuning, benchmarks, and profiling so experiments do not interfere. Other agents can continue CPU work in parallel.
- Agents share the workspace. Edit only assigned files, accommodate others' changes, and never revert their work.
- Report source or artifact paths, compiler and config, validation results, and unresolved hypotheses. Static IR / SASS analysis does not replace GPU performance measurements.
