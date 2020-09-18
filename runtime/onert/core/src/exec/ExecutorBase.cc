/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ExecutorBase.h"

#include "backend/ITensor.h"
#include "backend/controlflow/UserTensor.h"
#include "backend/cpu_common/Tensor.h"
#include "util/logging.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace exec
{

ExecutorBase::ExecutorBase(std::unique_ptr<compiler::LoweredGraph> &&lowered_graph,
                           const compiler::TensorRegistries &tensor_regs)
    : _lowered_graph{std::move(lowered_graph)}, _graph{_lowered_graph->graph()}, _mutex()
{
  auto build_tensor_list = [&](const auto &ind_seq, auto &tensors) {
    assert(tensors.empty());
    for (auto ind : ind_seq)
    {
      backend::ITensor *tensor = tensor_regs.getITensor(ind);
      assert(tensor != nullptr);
      auto io_tensor = nnfw::misc::polymorphic_downcast<backend::controlflow::IOTensor *>(tensor);
      tensors.push_back(io_tensor);
    }
  };
  build_tensor_list(_graph.getInputs(), _input_tensors);
  build_tensor_list(_graph.getOutputs(), _output_tensors);
}

void ExecutorBase::execute(const std::vector<backend::ITensor *> &src_tensors,
                           const std::vector<backend::ITensor *> &dst_tensors)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  // Deadlock occurs when an Executor is called recursively.
  std::lock_guard<std::mutex> lock(_mutex);

  assert(src_tensors.size() == _graph.getInputs().size());
  assert(src_tensors.size() == _input_tensors.size());
  for (uint32_t n = 0; n < src_tensors.size(); ++n)
  {
    // when user changes input shape, the input tensor is dynamic and its memory is not allocated.
    // This code find the info to allocate dynamic tensor, and allocate memory based on the source
    // tensor's shape set by caller.
    const auto src_tensor = src_tensors[n];
    assert(src_tensor->buffer() != nullptr);
    auto input_tensor = _input_tensors[n];
    assert(input_tensor != nullptr);
    VERBOSE_F() << "XXXXXXXXXX Input  " << n << " " << input_tensor->getShape() << std::endl;
    // if (input_tensor->orig_info().shape() != src_tensor->getShape())
    input_tensor->set_dynamic();
    input_tensor->setTensor(src_tensor);
  }

  // XXX Workaround!
  if (!dst_tensors.empty())
  {
    assert(dst_tensors.size() == _graph.getOutputs().size());
    assert(dst_tensors.size() == _output_tensors.size());
    for (uint32_t n = 0; n < dst_tensors.size(); ++n)
    {
      const auto dst_tensor = dst_tensors[n];
      // assert(dst_tensor->buffer() != nullptr);
      auto output_tensor = _output_tensors[n];
      assert(output_tensor != nullptr);
      VERBOSE_F() << "XXXXXXXXXX Output " << n << std::endl;
      output_tensor->setTensor(dst_tensor);
    }
  }

  executeImpl();
}

void ExecutorBase::execute(const IODescription &desc)
{
  // For thread-safe, use mutex
  // TODO: if all used backends on this executor are thread-safe,
  //       do not need to use mutex (otherwise, use mutex)
  std::lock_guard<std::mutex> lock(_mutex);

  // Set input(s)
  assert(_input_tensors.size() == desc.inputs.size());
  for (uint32_t i = 0; i < _input_tensors.size(); ++i)
  {
    auto tensor = _input_tensors[i];

    // TODO Check if (desc.inputs[i] == nullptr)
    // TODO Better design for ITensor? (we need const_cast as ITensor is writable)
    tensor->setUserTensor(static_cast<uint8_t *>(const_cast<void *>(desc.inputs[i]->buffer)),
                          desc.inputs[i]->size);

    auto input_shape = desc.dynamic_input_shapes.find(ir::IOIndex{i});
    if (input_shape != desc.dynamic_input_shapes.end())
    {
      tensor->set_dynamic();
      tensor->setShape(input_shape->second);
    }

    handleDynamicInputTensor(ir::IOIndex{i}, desc);
  }

  assert(_output_tensors.size() == desc.outputs.size());
  for (uint32_t i = 0; i < _output_tensors.size(); ++i)
  {
    auto tensor = _output_tensors[i];

    if (desc.outputs[i] == nullptr)
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};
    tensor->setUserTensor(static_cast<uint8_t *>(desc.outputs[i]->buffer), desc.outputs[i]->size);
    tensor->set_dynamic(); // It can't be resized but shape could change
  }

  executeImpl();

  // Update output(s) desc
  for (uint32_t n = 0; n < _graph.getOutputs().size(); ++n)
  {
    ir::IOIndex output_index{n};
    // Optional output
    if (desc.outputs.at(n) == nullptr)
    {
      continue;
    }
    auto &output = *desc.outputs.at(n);

    // set shape of outputDesc to tensor shape since tensor can be dynamic
    const auto output_tensor_shape = _output_tensors[n]->getShape();
    output.info.shape(
        convertShape(output_tensor_shape, _output_tensors[n]->layout(), output.layout));
  }
}

/**
 * @brief Changes tensor shape and allocate memory
 *        if input shape was changed by nnfw_set_input_tensorinfo()
 *
 * @note  Cases are:
 *        1) static operand -> nnfw_set_input_tensorinfo() -> execute() -> execute()
 *                                                        (a)          (b)
 *
 *           at (a), operand is static, tensor is static - memory dealloc is not needed
 *                   (DynamicTensorManager cannot dealloc memory allocated by StaticTensorManager)
 *           at (b), operand is static, tensor is dynamic - memory dealloc is needed
 *
 *        2) dynamic operand -> nnfw_set_input_tensorinfo() -> execute() -> execute()
 *                                                         (a)          (b)
 *
 *           at (a), operand is dynamic, tensor is dynamic - memory dealloc is not needed
 *                                                           since it has not been allocated yet
 *           at (b), operand is dynamic, tensor is dynamic - memory dealloc is needed
 */
void ExecutorBase::handleDynamicInputTensor(ir::IOIndex io_ind, const IODescription &desc)
{
  auto shape_sig_found = desc.dynamic_input_shapes.find(io_ind);
  if (shape_sig_found != desc.dynamic_input_shapes.end())
  {
    auto changed_input_shape = shape_sig_found->second;
    _input_tensors[io_ind.value()]->applyShape(changed_input_shape);
  }
}

bool ExecutorBase::hasDynamicInput()
{
  for (auto &tensor : _input_tensors)
  {
    if (tensor->is_dynamic())
      return true;
  }
  return false;
}

} // namespace exec
} // namespace onert
