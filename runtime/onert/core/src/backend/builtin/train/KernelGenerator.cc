/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "KernelGenerator.h"

#include "kernel/PermuteLayer.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{

KernelGenerator::KernelGenerator(const ir::train::TrainableGraph &tgraph,
                                 const std::shared_ptr<TensorRegistry> &tensor_reg,
                                 const std::shared_ptr<ExternalContext> &external_context)
  : KernelGeneratorBase{tgraph}, _tensor_reg{tensor_reg}, _external_context(external_context)
{
}

std::unique_ptr<exec::train::TrainableFnSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::train::TrainableFnSequence>();
  const auto &op = _tgraph.operation(ind);
  op.accept(*this);
  // _return_fn must have been generated
  if (_return_fn == nullptr)
  {
    throw std::runtime_error(op.name() + " op does not supported trainable kernel yet");
  }

  ret->_functions.emplace_back(std::move(_return_fn));

  return ret;
}

void KernelGenerator::visit(const ir::train::operation::Permute &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  // Add PermuteLayer
  std::vector<ITensor *> output_tensors{getTensor(output_index)};
  std::vector<ITensor *> input_tensors{getTensor(input_index)};

  std::vector<ITensor *> output_back_prop_tensors;
  std::vector<ITensor *> input_back_prop_tensors;

  auto input_back_prop_tensor = getBackPropTensor(input_index);
  auto output_back_prop_tensor = getBackPropTensor(output_index);
  output_back_prop_tensors.emplace_back(output_back_prop_tensor);
  input_back_prop_tensors.emplace_back(input_back_prop_tensor);

  // TODO Remove ignore_forward_in_training
  //      If whole graph's output is the same with output index, it need to execute the
  //      forward() function to get the output result so that user can take the output
  //      result.
  bool ignore_forward_in_training = false;
  auto fn = std::make_unique<kernel::PermuteLayer>(
    input_tensors, output_tensors, input_back_prop_tensors, output_back_prop_tensors,
    ignore_forward_in_training, _external_context);

  _return_fn = std::move(fn);
}

backend::ITensor *KernelGenerator::getTensor(const ir::OperandIndex &index)
{
  // Get Tensor from all tensor registries (for Permute op)
  auto ret = _tensor_registries.getITensor(index);
  assert(ret != nullptr);
  return ret;
}

backend::ITensor *KernelGenerator::getBackPropTensor(const ir::OperandIndex &index)
{
  // Get back propagation Tensor from all tensor registries (for Permute op)
  auto ret = _tensor_registries.getBackPropITensor(index);
  return ret;
}

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
