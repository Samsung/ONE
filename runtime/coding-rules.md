## Coding Rules

This is recommended coding rules for the project.

### Naming Conventions

- Class name: PascalCase
  - Struct, Union, Enum, Interface, etc. should also follow this rule.
- Method name: camelCase
- Function name: snake_case
- Variable name: snake_case
  - Parameter name: snake_case
  - Class member variable name
    - _snake_case (starts with underscore) for private member variables
    - snake_case for public member variables
- Constant name: UPPER_SNAKE_CASE
- Macro name: UPPER_SNAKE_CASE
- Namespace: snake_case

### STL Usage

- Use `std::` prefix to avoid ambiguity
- Use `std::vector` as default container
- `std::array` is preferred over `std::vector` when size is fixed
- Use `std::unordered_map` instead of `std::map` for performance reason
  - Use `std::map` when order is required

### Code Style

- Follow .clang-format file in the repository.
- You can use `./nnas format` command to apply clang-format to entire project.
- You can use `./nnas format --diff-only` command to apply clang-format to only modified files.
