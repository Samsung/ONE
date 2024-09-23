# Origin of source code

It is ggml part in llama.cpp https://github.com/ggerganov/llama.cpp/

# Version

b3542: https://github.com/ggerganov/llama.cpp/tree/b3542

# Background

It is part of ggml, not all code to reduce the binary size.

C code marking

- `#if 0 // [FIX] disable` & `#endif // [FIX] end` pair: Manually disable unused code

CMake marking
- `# [FIX] comment~ `: Manually fix for build
