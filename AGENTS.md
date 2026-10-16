# AGENTS

目标硬件是sm120 小显存GPU能跑通并且优化性能。使用cutile在python编写kernel，保证正确性。使用cutile 编译完cubin之后通过C++调用。这个项目的主体部分是C++，来进行kernel调用，内存管理等等

## Use of subagents

You are encouraged to use subagents to supervise processes for compiling and do research work.