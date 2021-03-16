This is lightweight replacement for C++ std::filesystem library.

Motivation and rationale:
1) Source code of codegen should comply with c++14 (no standard fs library available)
2) Codegen need to manipulate directories
3) Codegen need to avoid unnecessary dependencies from third-party components

This library provides subset of functions implemented in std::filesystem
to provide basic functionality to manipulate directories and files.
