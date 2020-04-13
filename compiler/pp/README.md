# pp

`pp` is a library which provides various helper functions and classes for pretty-printing.
This was originted while writing C/C++ code generator.

# Function (Feature)

With `pp`, the following can be built:
- multi-line structure with easy indentation, where each line can be accessed by index
- indented string
- concating `string`, `int`, etc., without user's explicit type conversion
- multi-line string
an so on.

# How to use

- Some of examples are listed below:
  - `pp::fmt`

    ```cpp
    std::cout << pp::fmt("Hello ", 2) << "\n"; // "Hello 2"
    std::cout << pp::fmt("Hello ", "Good ", "World") << "\n"; // ""Hello Good World"
    ```
  - `pp::IndentedStringBuilder`

    ```cpp
    pp::IndentedStringBuilder builder{};

    std::cout << builder.build("A") << "\n"; // "A"
    builder.increase();
    std::cout << builder.build("B") << "\n"; // "  B"
    builder.decrease();
    std::cout << builder.build("C") << "\n"; // "C"
    ```
  - For more usage and examples, please refer to `*.test.cpp` under `pp/src`.
