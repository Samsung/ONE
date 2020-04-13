/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MirInterpreter.h"

#include "ops/Add.h"
#include "ops/Abs.h"
#include "ops/AvgPool2D.h"
#include "ops/CappedReLU.h"
#include "ops/Concat.h"
#include "ops/Conv2D.h"
#include "ops/DeConv2D.h"
#include "ops/DepthwiseConv2D.h"
#include "ops/Div.h"
#include "ops/ELU.h"
#include "ops/Equal.h"
#include "ops/Fill.h"
#include "ops/FullyConnected.h"
#include "ops/Gather.h"
#include "ops/Greater.h"
#include "ops/HardSwish.h"
#include "ops/LeakyReLU.h"
#include "ops/Less.h"
#include "ops/Max.h"
#include "ops/MaxPool2D.h"
#include "ops/Mul.h"
#include "ops/Pad.h"
#include "ops/Quantization.h"
#include "ops/ReduceMean.h"
#include "ops/ReLU.h"
#include "ops/Reshape.h"
#include "ops/Sigmoid.h"
#include "ops/Slice.h"
#include "ops/Softmax.h"
#include "ops/Sqrt.h"
#include "ops/Sub.h"
#include "ops/Tanh.h"
#include "ops/Transpose.h"

#include "ops/Common.h"

#include "mir/OpDefs.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace mir_interpreter
{

using namespace mir;

void MIRInterpreter::setTensor(const Operation::Output *output, TensorVariant tensor)
{
  const auto result = _tensors.emplace(output, std::move(tensor));
  if (!result.second)
  {
    const std::string &name = output->getName();
    throw std::runtime_error("Attempt to overwrite data for tensor \"" + name + "\".");
  }
}

const TensorVariant &MIRInterpreter::getTensor(const Operation::Output *output) const
{
  const auto it = _tensors.find(output);
  if (it == _tensors.end())
  {
    const std::string &name = output->getName();
    throw std::runtime_error("Can't find data for tensor \"" + name + "\".");
  }
  return it->second;
}

std::vector<std::reference_wrapper<const TensorVariant>>
MIRInterpreter::getInputTensors(const Operation &op)
{
  std::vector<std::reference_wrapper<const TensorVariant>> tensors;
  for (const Operation::Output *input : op.getInputs())
  {
    tensors.emplace_back(getTensor(input));
  }
  return tensors;
}

std::vector<std::reference_wrapper<TensorVariant>>
MIRInterpreter::getOutputTensors(const Operation &op)
{
  // Create and register output tensors.
  for (const Operation::Output &res : op.getOutputs())
  {
    assert(res.getElementType() != mir::DataType::UNKNOWN);
    _tensors.emplace(&res, res.getType());
  }

  // Gather references to output tensors.
  std::vector<std::reference_wrapper<TensorVariant>> tensors;
  for (const Operation::Output &output : op.getOutputs())
  {
    tensors.emplace_back(_tensors.at(&output));
  }
  return tensors;
}

// Deprecated, will be removed.
void MIRInterpreter::setOutputTensors(const Operation &op, std::vector<TensorVariant> &&outputs)
{
  assert(outputs.size() == op.getNumOutputs());
  for (std::size_t i = 0; i < op.getNumOutputs(); ++i)
  {
    setTensor(op.getOutput(i), std::move(outputs[i]));
  }
}

// Deprecated, will be removed.
TensorVariant MIRInterpreter::getResult(const Operation::Output *tensor)
{
  return getTensor(tensor);
}

void MIRInterpreter::visit(ops::InputOp &op)
{
  assert(_tensors.find(op.getOutput(0)) != _tensors.end());
}

void MIRInterpreter::visit(ops::AvgPool2DOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  AvgPool2D(op, inputs[0], outputs[0]);
}

void MIRInterpreter::visit(ops::ConstantOp &op) { setOutputTensors(op, {op.getValue()}); }

void MIRInterpreter::visit(ops::ConcatOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Concat(inputs, op.getAxis(), outputs[0]);
}

void MIRInterpreter::visit(ops::Conv2DOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  const mir::TensorVariant *bias = nullptr;
  if (inputs.size() > 2)
  {
    bias = &(inputs[2].get());
  }
  Conv2D(inputs[0], inputs[1], op.getAttributes(), outputs[0], bias);
}

void MIRInterpreter::visit(ops::MaxPool2DOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  MaxPool2D(inputs[0], op, outputs[0]);
}

void MIRInterpreter::visit(ops::ReshapeOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = Reshape(inputs[0], op.getOutputShape(0));
  setOutputTensors(op, std::move(outputs));
}

void MIRInterpreter::visit(ops::ReluOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  ReLU(args[0], results[0]);
}

void MIRInterpreter::visit(ops::SigmoidOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  Sigmoid(args[0], results[0]);
}

void MIRInterpreter::visit(ops::SoftmaxOp &op)
{
  auto inputs = getInputTensors(op);
  assert(inputs.size() == 1);
  auto outputs = getOutputTensors(op);
  Softmax(inputs[0], op.getAxis(), outputs[0]);
}

void MIRInterpreter::visit(ops::FullyConnectedOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  const mir::TensorVariant *bias = nullptr;
  if (inputs.size() > 2)
  {
    bias = &(inputs[3].get());
  }
  FullyConnected(inputs[0], inputs[1], op, outputs[0], bias);
}

void MIRInterpreter::visit(ops::CappedReluOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  CappedReLU(args[0], op.getCap(), results[0]);
}

void MIRInterpreter::visit(ops::DepthwiseConv2DOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  const mir::TensorVariant *bias = nullptr;
  if (inputs.size() > 2)
  {
    bias = &inputs[3].get();
  }
  DepthwiseConv2D(op, inputs[0], inputs[1], outputs[0], bias);
}

void MIRInterpreter::visit(ops::SliceOp &op)
{
  auto inputs = getInputTensors(op);
  auto input = inputs[0];
  auto outputs = getOutputTensors(op);
  Slice(input, op.getStarts(), outputs[0]);
}

void MIRInterpreter::visit(ops::TanhOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  Tanh(args[0], results[0]);
}

void MIRInterpreter::visit(ops::DeConv2DOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  DeConv2D(inputs[0], inputs[1], op, outputs[0]);
}

void MIRInterpreter::visit(ops::EluOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  ELU(args[0], op.getAlpha(), results[0]);
}

void MIRInterpreter::visit(ops::SqueezeOp &op)
{
  auto inputs = getInputTensors(op);
  // Squeeze is just a special case of reshape.
  auto outputs = Reshape(inputs[0], op.getOutputShape(0));
  setOutputTensors(op, std::move(outputs));
}

void MIRInterpreter::visit(ops::PadOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Pad(inputs[0], op, outputs[0]);
}

void MIRInterpreter::visit(ops::SqrtOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  Sqrt(args[0], results[0]);
}

void MIRInterpreter::visit(ops::ResizeOp &op)
{
  // TODO support types other than float32
  auto inputs = getInputTensors(op);
  assert(inputs[0].get().getElementType() == mir::DataType::FLOAT32);
  auto outputs = getOutputTensors(op);

  Tensor<float> input(inputs[0]);
  assert(op.getMode() == ops::ResizeOp::ResizeMethod::nearestNeighbor);

  auto scales = op.getScales();
  Fill(outputs[0], [&scales, &input](const Index &id) {
    Index in_idx;
    in_idx.resize(4);
    for (int i = 0; i < input.getShape().rank(); i++)
    {
      in_idx.at(i) = static_cast<int>(floorf(id.at(i) / scales[i]));
    }
    return input.at(in_idx);
  });
}

void MIRInterpreter::visit(ops::ReduceMeanOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  ReduceMean(inputs[0], op, outputs[0]);
}

void MIRInterpreter::visit(ops::TransposeOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Transpose(inputs[0], op, outputs[0]);
}

void MIRInterpreter::visit(ops::GatherOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Gather(inputs[0], inputs[1], op, outputs[0]);
}

void MIRInterpreter::visit(ops::LeakyReluOp &op)
{
  auto args = getInputTensors(op);
  auto results = getOutputTensors(op);
  LeakyReLU(args[0], op.getAlpha(), results[0]);
}

void MIRInterpreter::visit(ops::OutputOp &op)
{
  assert(_tensors.find(op.getInput(0)) != _tensors.end());
}

void MIRInterpreter::visit(ops::AddOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Add(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::DivOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Div(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::MaxOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Max(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::MulOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Mul(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::SubOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Sub(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::DequantizeOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Dequantize(inputs[0], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::QuantizeOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Quantize(inputs[0], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::HardSwishOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  HardSwish(inputs[0], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::GreaterOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Greater(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::LessOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Less(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::EqualOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Equal(inputs[0], inputs[1], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::AbsOp &op)
{
  auto inputs = getInputTensors(op);
  auto outputs = getOutputTensors(op);
  Abs(inputs[0], outputs[0]);
}

void MIRInterpreter::visit(mir::ops::BroadcastOp &op)
{
  auto inputs = getInputTensors(op);
  TensorVariant tv{inputs[0], op.getOutputShape(0)};
  setOutputTensors(op, {tv});
}

void MIRInterpreter::visit_fallback(mir::Operation &) { throw std::runtime_error("NYI operation"); }

} // namespace mir_interpreter
