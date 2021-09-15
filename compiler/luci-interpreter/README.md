# luci-interpreter

`luci-interpreter` is an interpreter engine for luci IR models (see compiler/luci module for details).

It provides:
- Basic inference interfaces, setting input and getting output data
- Basic analysis interfaces, like getting intermediate actiavation data
- Customization mechanisms to adopt interpreter to specific platforms, like MCUs

Public interface headers placed in `luci-interpreter/include/luci_interpreter` directory

### Basic usage

Minimal usage requires:
- Setting input data
- Inference
- Getting inference results

Interpreter object is reusable and can run multiple inferences.

Usage example (assuming model has only one input and output):

``` c++
luci_interpreter::Interpreter interpreter(luci_module);

// Set inputs
const auto input_nodes = loco::input_nodes(module->graph());

const auto *input_node = dynamic_cast<const luci::CircleInput *>(input_nodes[0]);
std::vector<char> input_data(getTensorSize(input_node));
// Initialize input data here

interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());

// Start inference
interpreter.interpret();

// Fetch inference results
const auto output_nodes = loco::output_nodes(module->graph());
const auto *output_node = dynamic_cast<const luci::CircleOutput *>(output_nodes[0]);
std::vector<char> output_data(getTensorSize(output_node));
interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());
```

### Inspecting intermediate state

Interpreter provides interfaces to investigate internal state of interpreter during inference.

This is done by "Observer" mechanism:
- `Interpreter` class has `attachObserver` method, which takes pointer to `ExecutionObserver` object
- `ExecutionObserver` defines several callback methods which user can override to inject custom code

ExecutionObserver provides three callbacks:
- `postTensorWrite` checks contents of output tensor after operation execution
- `preOperatorExecute` notifies that operation is going to execute
- `postOperatorExecute` notifies that operation is finished execution

See `luci-interpreter/include/luci_interpreter/Interpreter.h` for implementation details.

Usage example:
``` c++
class CustomExecutionObserver: public luci_interpreter::ExecutionObserver
{
public:
  void postTensorWrite(const luci::CircleNode *node, const Tensor *tensor) override
  {
    if (tensor->element_type() != loco::DataType::FLOAT32)
      return;
    for (int i = 0; i < tensor->shape().num_elements(); ++i)
      std::cout << tensor->data<float>[i] << ", ";
  }

  // User observer can override only needed methods,
  // others will inherit empty implementation from base observer.

  // void preOperatorExecute(const luci::CircleNode *node);
  // void postOperatorExecute(const luci::CircleNode *node);
};

luci_interpreter::Interpreter interpreter(module);
CustomExecutionObserver observer;
interpreter.attachObserver(observer);

// initialize input_data
interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());

interpreter.interpret();
```

### Customizing inference

#### Memory manager

Interpreter provides handle to alter default memory management mechanisms.

This is done by MemoryManger interface, see `luci-interpreter/include/luci_interpreter/MemoryManager.h` for details.

This header contains IMemoryManamger abstract class which is responsible for allocation and dealocation of tensors memory.

User can construct interpreter with one of predefined Memory managers or it's own custom memory manager.

List of existing memory managers:
- `SimpleMemoryManager` This is simple wrapper around new/delete, default manager.
- `TestMemoryManager` Memorizing all allocated memory and releases it in Manager desctuctor. Used in kernel tests.
- `BuddyMemoryManager` Implements Buddy algorithm, uses beffer for tensor data allocations.
- `StaticMemoryManger` Requires models preparation with MemoryPlanner. Could improve memory consumption in restricted environments, like MCUs.

Usage example:
``` c++
luci_interpreter::BuddyMemoryManager mm;

luci_interpreter::Interpreter interpreter(module, &mm);

// initialize input_data
interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());

interpreter.interpret();
```

StaticMemoryManager usage example:
``` c++
TBD when it is merged
```

### Further reading

If you want to participate in development, please read DEVELOPER.md for SW architecture details.