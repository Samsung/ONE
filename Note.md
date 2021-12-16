## Reduce constant duplication via castom source

Let's refactor GraphBuilderSource and delegate embedded constants building to castom GraphBuilderSource inheritor.

Plan:

1. Refactor GraphBuilderSource in [luci/import]: Add builder for inputs, outputs and buffers
2. Add extesnion to [luci-interpreter], which allows import buffers without constants copying.
3. Add support of new node to [luci-interpeter].
4. Provide example as [model-import-verifier] module.


After this draft importer can import models without constant copying:

```cpp
  // Load model from const pointer, import without copying, execute and save to output_buffer
  std::vector<model_dtype> output_data_3;
  {
    auto custom_source = luci_interpreter::source_without_constant_copying();
    auto module = import_model_constant_buffer(conv_model_const_pointer, custom_source.get());
    if (not module)
      throw std::runtime_error("Fail to import model");

    output_data_3 = interpret_module_and_write_to_output(module);
  }
```

This example can be found in EvalDriver of [model-import-verifier]. To verify code works well you need to build and launch *mode_import_verifier* executable without any args. Expected msg: `[TEST PASSED]`.
