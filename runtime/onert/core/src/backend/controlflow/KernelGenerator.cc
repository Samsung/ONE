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
#include "exec/ExecutorBase.h"
#include "exec/FunctionSequence.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

KernelGenerator::KernelGenerator(const ir::Graph &graph,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder,
                                 const std::shared_ptr<TensorRegistry> &tensor_reg)
    : _graph{graph}, _tensor_builder{tensor_builder}, _tensor_reg{tensor_reg},
      _tensor_builder_set{}, _executor_map{nullptr}
{
  UNUSED_RELEASE(_graph);
  UNUSED_RELEASE(_tensor_builder_set);
  UNUSED_RELEASE(_executor_map);
}

void KernelGenerator::visit(const ir::OpSequence &op_seq)
{
  assert(!_return_fn_seq);
  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_reg);

  auto dyn_tensor_manager = _tensor_builder->dynamicTensorManager();
  auto dyn_shape_inferer = std::make_unique<exec::DynamicShapeInferer>(
      _graph.operands(), dyn_tensor_manager, _tensor_reg);

  _return_fn_seq = std::make_unique<exec::FunctionSequence>();

  // Prepare to handle dynamic tensors later
  auto dyn_ctx = std::make_shared<exec::FunctionSequence::DynamicTensorCtx>();
  {
    dyn_ctx->op_seq = &op_seq;
    dyn_ctx->operations = &_graph.operations();
    dyn_ctx->dynamic_shape_inferer = std::move(dyn_shape_inferer);
    dyn_ctx->tensor_registry = _tensor_reg;
    dyn_ctx->dynamic_tensor_manager = _tensor_builder->dynamicTensorManager();

    _return_fn_seq->dynamic_tensor_ctx(dyn_ctx);
  }
  _return_fn_seq->enableDynamicShapeInferer(true);

  for (const auto &op_idx : op_seq.operations())
  {
    const auto &node = _graph.operations().at(op_idx);
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
    auto input_tensor = getTensor(input_index);

    input_tensors.emplace_back(input_tensor);
  }

  std::vector<std::shared_ptr<backend::ITensor>> output_tensors;
  exec::DynAllocInfoMap outputs_dyn_alloc_info;
  for (const auto output_index : node.getOutputs())
  {
    auto output_tensor = getTensor(output_index);

    output_tensors.emplace_back(output_tensor);
    const auto output_tensor_builder = getTensorBuilder(output_index);
    outputs_dyn_alloc_info[output_tensor] = exec::DynAllocInfo{output_index};
  }

  // IfLayer just set ExecutorMap instead of then and else executor to avoid complexity of
  // creating executor recusively
  const auto cond_tensor = input_tensors.front();
  input_tensors.erase(input_tensors.begin());
  auto fn = std::make_unique<::onert::backend::controlflow::kernel::IfLayer>(
      cond_tensor, input_tensors, output_tensors, node.getOutputs(), _graph, outputs_dyn_alloc_info,
      then_subg_index, else_subg_index, _executor_map);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Permute &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  // Add PermuteLayer
  std::vector<std::shared_ptr<ITensor>> output_tensors{getTensor(output_index)};
  std::vector<std::shared_ptr<ITensor>> input_tensors{getTensor(input_index)};
  std::unordered_map<std::shared_ptr<ITensor>, exec::DynAllocInfo> outputs_dyn_alloc_info;
  const auto output_tensor_builder = getTensorBuilder(output_index);
  VERBOSE(PERMUTE_FIND_TB) << output_index << " -> " << output_tensor_builder.get() << std::endl;
  assert(output_tensor_builder != nullptr);
  outputs_dyn_alloc_info[output_tensors.at(0)] = exec::DynAllocInfo{output_index};

  auto fn =
      std::make_unique<kernel::PermuteLayer>(input_tensors, output_tensors, outputs_dyn_alloc_info);

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
    auto input_tensor = getTensor(input_index);

    input_tensors.emplace_back(input_tensor);
  }

  std::vector<std::shared_ptr<backend::ITensor>> output_tensors;
  std::unordered_map<std::shared_ptr<ITensor>, exec::DynAllocInfo> outputs_dyn_alloc_info;
  for (const auto output_index : node.getOutputs())
  {
    auto output_tensor = getTensor(output_index);

    output_tensors.emplace_back(output_tensor);

    const auto output_tensor_builder = getTensorBuilder(output_index);
    outputs_dyn_alloc_info[output_tensor] = exec::DynAllocInfo{output_index};
  }

  // WhileLayer just set ExecutorMap instead of cond and body executor to avoid complexity of
  // creating executor recusively
  auto fn = std::make_unique<::onert::backend::controlflow::kernel::WhileLayer>(
      input_tensors, output_tensors, node.getOutputs(), _graph, outputs_dyn_alloc_info,
      cond_subg_index, body_subg_index, _executor_map);

  _return_fn = std::move(fn);
}

std::shared_ptr<backend::ITensor> KernelGenerator::getTensor(const ir::OperandIndex &index)
{
  std::shared_ptr<backend::ITensor> ret = _tensor_builder_set.getITensor(index);
  assert(ret != nullptr);
  return ret;
}

std::shared_ptr<backend::ITensorBuilder>
KernelGenerator::getTensorBuilder(const ir::OperandIndex &index)
{
  std::shared_ptr<backend::ITensorBuilder> ret;
  for (auto tensor_builder : _tensor_builder_set)
  {
    auto reg = tensor_builder->tensorRegistry();
    auto tensor = reg->getITensor(index);
    if (tensor)
    {
      ret = tensor_builder;
      break;
    }
  }
  assert(ret != nullptr);
  return ret;
}

} // namespace controlflow
} // namespace backend
} // namespace onert
