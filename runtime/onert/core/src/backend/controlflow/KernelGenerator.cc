/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <backend/BackendContext.h>
#include <util/Utils.h>
#include "kernel/IfLayer.h"
#include "kernel/WhileLayer.h"
#include "kernel/PermuteLayer.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

KernelGenerator::KernelGenerator(const ir::Operands &operand_ctx,
                                 const ir::Operations &operations_ctx)
    : _operand_ctx{operand_ctx}, _operations_ctx{operations_ctx}, _tensor_builder_set{nullptr},
      _executor_map{nullptr}
{
  UNUSED_RELEASE(_operand_ctx);
  UNUSED_RELEASE(_operations_ctx);
  UNUSED_RELEASE(_tensor_builder_set);
  UNUSED_RELEASE(_executor_map);
}

void KernelGenerator::visit(const ir::OpSequence &op_seq)
{
  assert(!_return_fn_seq);
  _return_fn_seq = std::make_unique<exec::FunctionSequence>();
  for (const auto &op_idx : op_seq.operations())
  {
    const auto &node = _operations_ctx.at(op_idx);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());
  }
}

void KernelGenerator::visit(const ir::operation::If &node)
{
  const auto then_subg_index = node.param().then_subg_index;
  const auto else_subg_index = node.param().else_subg_index;

  std::vector<std::shared_ptr<backend::ITensor>> input_tensors;
  for (const auto input_index : node.getInputs())
  {
    auto input_alloc = getTensor(input_index);

    input_tensors.emplace_back(input_alloc);
  }

  std::vector<std::shared_ptr<backend::ITensor>> output_tensors;
  for (const auto output_index : node.getOutputs())
  {
    auto output_alloc = getTensor(output_index);

    output_tensors.emplace_back(output_alloc);
  }

  // IfLayer just set ExecutorMap instead of then and else executor to avoid complexity of
  // creating executor recusively
  const auto cond_tensor = input_tensors.front();
  input_tensors.erase(input_tensors.begin());
  auto fn = std::make_unique<::onert::backend::controlflow::kernel::IfLayer>(
      cond_tensor, input_tensors, output_tensors, then_subg_index, else_subg_index, _executor_map);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Permute &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  std::vector<std::shared_ptr<ITensor>> output_tensors{getTensor(output_index)};
  std::vector<std::shared_ptr<ITensor>> input_tensors{getTensor(input_index)};

  auto fn = std::make_unique<kernel::PermuteLayer>(input_tensors, output_tensors);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::While &node)
{
  const auto cond_subg_index = node.param().cond_subg_index;
  const auto body_subg_index = node.param().body_subg_index;

  // This op does not support input as a constant, because controlflow backend does not have
  // TensorBuilder
  std::vector<std::shared_ptr<backend::ITensor>> input_tensors;
  for (const auto input_index : node.getInputs())
  {
    auto input_alloc = getTensor(input_index);

    input_tensors.emplace_back(input_alloc);
  }

  std::vector<std::shared_ptr<backend::ITensor>> output_tensors;
  for (const auto output_index : node.getOutputs())
  {
    auto output_alloc = getTensor(output_index);

    output_tensors.emplace_back(output_alloc);
  }

  // WhileLayer just set ExecutorMap instead of cond and body executor to avoid complexity of
  // creating executor recusively
  auto fn = std::make_unique<::onert::backend::controlflow::kernel::WhileLayer>(
      input_tensors, output_tensors, cond_subg_index, body_subg_index, _executor_map);

  _return_fn = std::move(fn);
}

std::shared_ptr<backend::ITensor> KernelGenerator::getTensor(const ir::OperandIndex &index)
{
  std::shared_ptr<backend::ITensor> ret;
  for (auto tensor_builder : _tensor_builder_set)
  {
    auto tensor = tensor_builder->tensorAt(index);
    if (tensor)
    {
      ret = tensor;
    }
  }
  assert(ret != nullptr);
  return ret;
}

} // namespace controlflow
} // namespace backend
} // namespace onert
