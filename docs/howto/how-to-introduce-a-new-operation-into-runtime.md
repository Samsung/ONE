# How To Introduce a New Operation Into Runtime

**ONE**'s runtime has three main modules: **core**, **frontend** and **backend**. This document
provides some lightweight guidance about how to introduce a new operation into these modules to make
onert support the operation.

## Index

- [How To Introduce a New Operation Into Runtime](#how-to-introduce-a-new-operation-into-runtime)
  - [Index](#index)
  - [Core](#core)
  - [Frontend](#frontend)
    - [Loaders](#loaders)
      - [Base Loader](#base-loader)
      - [TFLite Loader](#tflite-loader)
      - [Circle Loader](#circle-loader)
    - [NNAPI](#nnapi)
  - [Backend](#backend)
    - [ShapeFixer](#shapefixer)
      - [acl_cl](#acl_cl)
      - [acl_neon](#acl_neon)
      - [cpu](#cpu)
    - [KernelGenerator](#kernelgenerator)
      - [acl_cl](#acl_cl-1)
      - [acl_neon](#acl_neon-1)
      - [cpu](#cpu-1)
    - [ConstantInitializer (in some cases)](#constantinitializer-in-some-cases)
      - [cpu](#cpu-2)
  - [Samples (to be updated)](#samples-to-be-updated)

## Core

This module has graph-based IR(intermediate representation). You have to add IR for the new
operation.

1. Add name of new operation at [Operations.lst](/runtime/onert/core/include/ir/Operations.lst)

```cpp
OP(Select)
```

2. Create a class for node of new operation in [here](/runtime/onert/core/include/ir/operation/)

```cpp
#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

class Select : public Operation
{
public:
  enum Input
  {
    COND = 0,
    INPUT1 = 1,
    INPUT2 = 2
  };

  enum Output
  {
    OUTPUT = 0,
  };

public:
  Select(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs);

public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::Select; }
};

} // namespace operation
} // namespace ir
} // namespace onert
```

You can also define the class in other source file like below

```cpp
#include "ir/operation/Select.h"

#include "ir/OperationVisitor.h"

namespace onert
{
namespace ir
{
namespace operation
{

void Select::accept(OperationVisitor &v) const { v.visit(*this); }

Select::Select(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs)
    : Operation{OperandConstraint::createExact(3u), inputs, outputs}
{
}
```
  - [Operations.Include.h](/runtime/onert/core/include/ir/Operations.Include.h)

```cpp
#include "ir/operation/Select.h"
```

3. Add to the OperationValidator to check if the node is valid.
  - [OperationValidator.h](/runtime/onert/core/src/compiler/OperationValidator.h)

```cpp
void visit(const operation::Select &node) override;
```

  - [OperationValidator.cc](/runtime/onert/core/src/compiler/OperationValidator.cc)

```cpp
void OperationValidator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(ir::operation::Select::Output::OUTPUT)};
  const auto cond_index{node.getInputs().at(ir::operation::Select::Input::COND)};
  const auto input1_index{node.getInputs().at(ir::operation::Select::Input::INPUT1)};
  const auto input2_index{node.getInputs().at(ir::operation::Select::Input::INPUT2)};

  UNUSED_RELEASE(output_index);
  UNUSED_RELEASE(cond_index);
  UNUSED_RELEASE(input1_index);
  UNUSED_RELEASE(input2_index);

  const auto output_type = _ctx.at(output_index).typeInfo();
  const auto cond_type = _ctx.at(cond_index).typeInfo();
  const auto input1_type = _ctx.at(input1_index).typeInfo();
  const auto input2_type = _ctx.at(input2_index).typeInfo();

  UNUSED_RELEASE(output_type);
  UNUSED_RELEASE(cond_type);
  UNUSED_RELEASE(input1_type);
  UNUSED_RELEASE(input2_type);

  assert(cond_type.type() == ir::DataType::BOOL8);
  assert(output_type.type() == ir::DataType::FLOAT32 || output_type.type() == ir::DataType::INT32 ||
         output_type.type() == ir::DataType::QUANT8_ASYMM);
  assert(output_type.type() == input1_type.type());
  assert(output_type.type() == input2_type.type());

  const auto output_shape = _ctx.at(output_index).shape();
  const auto cond_shape = _ctx.at(cond_index).shape();
  const auto input1_shape = _ctx.at(input1_index).shape();
  const auto input2_shape = _ctx.at(input2_index).shape();

  UNUSED_RELEASE(output_shape);
  UNUSED_RELEASE(cond_shape);
  UNUSED_RELEASE(input1_shape);
  UNUSED_RELEASE(input2_shape);

  assert(output_shape == input1_shape);
  assert(cond_shape == input1_shape);
  assert(input2_shape == input1_shape);
}
```

4. Add to the Dumper to dump IR information of new operation.
- [Dumper.cc](/runtime/onert/core/src/ir/dumper/Dumper.cc)

```cpp
void Dumper::visit(const Select &node)
{
  VERBOSE(LIR) << "* Select" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Cond(" << node.getInputs().at(Select::Input::COND).value()
               << ") Input1" << node.getInputs().at(Select::Input::INPUT1).value() << ") Input2"
               << node.getInputs().at(Select::Input::INPUT2).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(Select::Output::OUTPUT).value()
               << ")" << std::endl;
}
```

5. Add code for shape inference
- ONE runtime tries to calculate shapes and allocate memory during compilation time. For some calculations of output shapes that cannot be done during compilation time, ONE runtime will calculate shapes and allocate memory during execution time.
- Calculation of shapes during compilation time is called _static shape inference_ and calculation of shapes during execution time is called _dynamic shape inference_.
- [`StaticShapeInferer.h`](`/runtime/onert/compiler/StaticShapeInferer.h`)

```CPP
  void visit(const ir::operation::Select &op) override;
```
- [`StaticShapeInferer.cc`](/runtime/onert/core/src/compiler/StaticShapeInferer.cc)
```CPP
void StaticShapeInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx{op.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto &input_cond = _operands.at(input_cond_idx);

  const auto &input_true = ...
  const auto &input_false = ...
  ir::Operand &output = ...

  // Select output shpae
  ir::Shape new_shape = shape_inference::inferSelectShape(
      input_cond.info().shape(), input_true.info().shape(), input_false.info().shape());
  output.info().shape(new_shape);
}
```
- [`DynamicShapeInference.h`](/runtime/onert/core/include/exec/DynamicShapeInference.h)
```CPP
  void visit(const ir::operation::Select &op) override;
```
- [`DynamicShapeInference.cc`](/runtime/onert/core/src/exec/DynamicShapeInference.cc)
```CPP
void DynamicShapeInferer::visit(const ir::operation::Select &op)
{
  const auto input_cond_idx = op.getInputs().at(ir::operation::Select::Input::CONDITION);
  const auto &input_cond = _tensor_registry->getITensor(input_cond_idx);

  const auto &input_true = ...
  const auto &input_false = ...
  auto output = ...

  if ((!input_cond->is_dynamic()) && (!input_true->is_dynamic()) && (!input_false->is_dynamic()))
  {
    return;
  }

  auto input_cond_shape = input_cond->getShape();
  auto input_true_shape = input_true->getShape();
  auto input_false_shape = input_false->getShape();

  // Select output shpae
  ir::Shape new_shape =
      shape_inference::inferSelectShape(input_cond_shape, input_true_shape, input_false_shape);

  output->applyShape(new_shape);
}
```

## Frontend

This module generates IR from a model. There are two kinds of frontend: Loader and NNAPI. First, Loader loads a model file and generates IR from it. Second, NNAPI generates IR from a model set via [Neural Networks API of android](https://developer.android.com/ndk/guides/neuralnetworks)

### Loaders

#### Base Loader

This is where the common parts of loaders are implemented.

1. Add to base_loader to load new operation and to generate IR from it
- [base_loader](/runtime/onert/core/src/loader/base_loader.h)

```cpp
    case BuiltinOperator::BuiltinOperator_SELECT:
      loadSelect(op);
      return;
```

```cpp
template <typename LoaderDomain, typename SpecificLoader>
void BaseLoader<LoaderDomain, SpecificLoader>::loadSelect(const Operator *op)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<ir::Operation> new_op{new ir::operation::Select{inputs, outputs}};
  _graph.addOperation(std::move(new_op));
}
```

#### TFLite Loader

This loads a tflite file.
If you want new operation to be loaded on only TFLite Loader, you only need to implement loading the operation here.

#### Circle Loader

This loads a circle file generated by the compiler.
If you want new operation to be loaded on only Circle Loader, you only need to implement loading the operation here.

### NNAPI

1. Add to the OperationFactory to generate IR of new operation
- [OperationFactory](/runtime/onert/frontend/nnapi/wrapper/OperationFactory.cc)

```cpp
  _map[ANEURALNETWORKS_SELECT] = [](const OperationFactory::Param &init_param, Operands &) {
    assert(init_param.input_count == 3 && init_param.output_count == 1);

    OperandIndexSequence outputs{init_param.outputs[0]};

    // Each input should be interpreted as follows:
    //
    //  0 -> Cond Tensor Index
    //  1 -> Input1 Tensor Index
    //  2 -> Input2 Tensor Index
    OperandIndexSequence inputs;
    for (uint32_t n = 0; n < init_param.input_count; ++n)
    {
      inputs.append(OperandIndex{init_param.inputs[n]});
    }

    return new operation::Select{inputs, outputs};
  };
```

2. If you want that NNAPI supports new operation of TFLite's model, you need to update the things related to the operation in [nnapi_delegate](/runtime/libs/tflite/port/1.13.1/src/nnapi_delegate.cpp) like below

```cpp
      case tflite::BuiltinOperator_SELECT:
        nnapi_version = 12;  // require NNAPI 1.2
        nn_op_type = ANEURALNETWORKS_SELECT;
        break;
```

## Backend

This module generates kernels and tensors of backend such as [ComputeLibrary](https://github.com/ARM-software/ComputeLibrary/) from generated graph-based IR. For this, the runtime fairly works on it internally. But this is not enough because of dependence on backend. So, there are several components that require additional implementation on each backend.

### ShapeFixer

Even for tensors of the same operation, the shape required for each backend can be different. Therefore, this component modifies and fixes shape of tensors of the backend.

#### acl_cl

The kernel of the ACL for the Add operation needs to match the same rank to support the broadcast.
- [ShapeFixer.h](/runtime/onert/backend/acl_cl/ShapeFixer.h)

```cpp
void visit(const ir::operation::Add &) override;
```

- [ShapeFixer.cc](/runtime/onert/backend/acl_cl/ShapeFixer.cc)

```cpp
void ShapeFixer::visit(const ir::operation::Add &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Add::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Add::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}
```

#### acl_neon

Same implementation as acl_cl is required.

#### cpu

This backend doesn't usually require a change of shape.
- [ShapeFixer.h](/runtime/onert/backend/cpu/ShapeFixer.h)

```cpp
void visit(const ir::operation::Select &) override;
```

- [ShapeFixer.cc](/runtime/onert/backend/cpu/ShapeFixer.cc)

```cpp
void ShapeFixer::visit(const ir::operation::Select &) { /* DO NOTHING */}
```

### KernelGenerator

This component generates kernels of backend. You have to generate kernel of new operation. And then append it to execution builder. You can obtain information of the node from IR and necessary tensors from tensor builder.

#### acl_cl

- [KernelGenerator.h](/runtime/onert/backend/acl_cl/KernelGenerator.h)

```cpp
void visit(const ir::operation::Select &) override;
```

- [KernelGenerator.cc](/runtime/onert/backend/acl_cl/KernelGenerator.cc)

```cpp
void KernelGenerator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(ir::operation::Select::Output::OUTPUT)};
  const auto cond_index{node.getInputs().at(ir::operation::Select::Input::COND)};
  const auto input1_index{node.getInputs().at(ir::operation::Select::Input::INPUT1)};
  const auto input2_index{node.getInputs().at(ir::operation::Select::Input::INPUT2)};

  auto output_alloc = _tensor_builder->at(output_index).get();
  auto cond_alloc = _tensor_builder->at(cond_index).get();
  auto input1_alloc = _tensor_builder->at(input1_index).get();
  auto input2_alloc = _tensor_builder->at(input2_index).get();

  auto fn = std::make_unique<::arm_compute::CLSelect>();

  fn->configure(cond_alloc->handle(), input1_alloc->handle(), input2_alloc->handle(),
                output_alloc->handle());

  auto acl_fn = asAclFunction(std::move(fn));

  _execution_builder->append(std::move(acl_fn));
}
```

#### acl_neon

Similar implementation as acl_cl is required.

#### cpu

- [KernelGenerator.h](/runtime/onert/backend/cpu/KernelGenerator.h)

```cpp
void visit(const ir::operation::Select &) override;
```

- [KernelGenerator.cc](/runtime/onert/backend/cpu/KernelGenerator.cc)

```cpp
void KernelGenerator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto condition_tensor = _tensor_reg->getPortableTensor(condition_index);
  auto true_tensor = _tensor_reg->getPortableTensor(true_index);
  auto false_tensor = _tensor_reg->getPortableTensor(false_index);

  auto fn = std::make_unique<ops::SelectLayer>();

  fn->configure(condition_tensor, true_tensor, false_tensor, output_tensor);

  _return_fn = std::move(fn);
}
```

### ConstantInitializer (in some cases)

This component registers function initializing constant tensors and initialize constant tensor
layer. Most tensors will be automatically registered internally. And there are some exceptions.

#### cpu

- [ConstantInitializer.h](/runtime/onert/backend/cpu/ConstantInitializer.h)

```cpp
void visit(const ir::operation::Conv2D &) override;
```

- [ConstantInitializer.cc](/runtime/onert/backend/cpu/ConstantInitializer.cc)

```cpp
void ConstantInitializer::visit(const ir::operation::Conv2D &node)
{
  const auto &kernel_index = node.getInputs().at(ir::operation::Conv2D::KERNEL);
  const auto &kernel_obj = _operands.at(kernel_index);
  registerCopyInitializer(kernel_index, kernel_obj);

  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::BIAS);
  const auto &bias_obj = _operands.at(bias_index);
  registerCopyInitializer(bias_index, bias_obj);
}
```

## Samples (to be updated)

- `Select` operation
  - Simple explanation : `Output[i] = Condition[i] ? input1[i] : input2[i]`
  - PR : https://github.com/Samsung/ONE/pull/XXX
