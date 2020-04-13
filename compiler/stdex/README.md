# stdex

`stdex` is an extension over standard C++ libraries.

# How to use

Please read each header files.

One example of `stdex::make_unique(..)` in `compiler/stdex/Memory.h` is as follows:

```cpp
#include <stdex/Memory.h>

using stdex::make_unique;

class A { ... };

...

std::unique_ptr<A> a = make_unique<A>(); // Note: std::make_unique is not supported in C++ 11

```
