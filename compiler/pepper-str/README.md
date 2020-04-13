# pepper-str

Let us simulate string interpolation in C++!

## HOW TO USE

```cxx
#include <pepper/str.h>

int main(int argc, char **argv)
{
  std::cout << pepper::str("There are ", argc, " arguments") << std::endl;
  return 0;
}
```
