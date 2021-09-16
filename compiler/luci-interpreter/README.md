# luci-interpreter

`luci-interpreter` is an inference engine for neural networks represented by luci IR.
See `compiler/luci/lang` directory for details about IR.
You can find useful infrastructure, like importer/exporter, optimizations in `compiler/luci`.

luci-interpreter provides:
- Basic inference functionality, input setters and output getters
- Interface for inspecting hidden interpreter state, like activation values during inference
- Customization mechanisms to adaptate interpreter to specific platforms, like MCUs

Public interface headers are placed in `luci-interpreter/include/luci_interpreter` directory

### Basic usage

Minimal usage includes:
- Setting input data
- Run inference
- Fetching inference results

Interpreter object is reusable and can run multiple inferences.
Elements in tensors (input/output/internal) are stored contiguously and has C-like layout:
This means for tensor t=[[0, 1],[2, 3]], t[0,1] == 1.

Input and output tensors are enumerated and have same order in origianl luci model. 

Usage example:
``` c++
// Note getTensorSize is a function that computes tensor size,
// it is not part of interpreter and should be implemented by user 

luci_interpreter::Interpreter interpreter(luci_module);

// Set inputs
// assuming model has only one input and one output
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

This is done by "observer" mechanism:
- `Interpreter` class has `attachObserver` method, which takes pointer to `ExecutionObserver` object
- `ExecutionObserver` defines several callback methods user can override to inject custom code

ExecutionObserver provides three callbacks:
- `postTensorWrite` checks contents of output tensor after operation execution
- `preOperatorExecute` notifies that interpreter is going to execute operation
- `postOperatorExecute` notifies that interpreter finished operation execution

See `luci-interpreter/include/luci_interpreter/Interpreter.h` for this interface details.

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
interpreter.attachObserver(&observer);

// initialize input_data
interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());

interpreter.interpret();
```

### Customizing inference

#### Memory manager

Interpreter provides handle to alter default memory management mechanisms.

This is done by `MemoryManger` interface, see `luci-interpreter/include/luci_interpreter/MemoryManager.h` for implementation details.

Header contains `IMemoryManager` abstract class which is responsible for allocation and dealocation of tensors memory.

User can construct interpreter with one of predefined memory mmanagers or it's own custom memory manager.

List of predefined memory managers:
- `SimpleMemoryManager` This is simple wrapper around new/delete, default one.
- `TestMemoryManager` Memorizing all allocated memory and releases it in Manager desctuctor, used in kernel unit tests.
- `BuddyMemoryManager` Implements Buddy algorithm, uses external buffer for tensor data allocations, do not need new/delete.
- `StaticMemoryManger` Uses precomputed memory allocation plan. Requires preparation with MemoryPlanner, but could improve memory consumption in restricted environments (like MCUs).

Usage example:
``` c++
luci_interpreter::BuddyMemoryManager mm;

luci_interpreter::Interpreter interpreter(module, &mm);

// initialize input_data
interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());

interpreter.interpret();
...
```

StaticMemoryManager usage example:
``` c++
TBD when it is merged
```

### Further reading

If you want to participate in development, please read `DEVELOPER.md` for SW architecture details.
