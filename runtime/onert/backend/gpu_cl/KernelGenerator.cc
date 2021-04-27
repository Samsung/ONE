/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <backend/basic/KernelGeneratorBase.h>

#include "KernelGenerator.h"
#include "ClTensorRegistry.h"
#include "ClFunction.h"
#include "TensorManager.h"
#include "../open_cl/kernels/Elementwise.h"
#include "../open_cl/selectors/Subgraph.h"
#include "../open_cl/selectors/SimpleSelectors.h"

#include "ir/Operations.h"
#include "ir/Operations.Include.h"
#include "ir/Index.h"
#include "ir/DataType.h"
#include "ir/InternalType.h"
#include "exec/NopFunction.h"
#include "exec/FunctionSequence.h"
#include "util/logging.h"
#include "util/Utils.h"
#include <stdexcept>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

KernelGenerator::KernelGenerator(const ir::Graph &graph,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder,
                                 const std::shared_ptr<ClTensorRegistry<TensorManager>> &tensor_reg,
                                 const std::shared_ptr<CreationContext> &creation_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()),
    _operations_ctx(graph.operations()), _current_layout{graph.layout()},
    _tensor_builder(tensor_builder), _tensor_reg(tensor_reg), _creation_context(creation_context)
{
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();
  ret->enableDynamicShapeInferer(false);

  const auto &op = _graph.operations().at(ind);
  op.accept(*this);
  ret->append(releaseFunction());
  return ret;
}

void KernelGenerator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  // const auto arithmetic_type = node.param().arithmetic_type;

  OperationDef op_def;

  // switch (arithmetic_type)
  // {
  // case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
  //   break;

  // default:
  //   break;
  // }

  auto tensor_reserver = _tensor_reg->getTensorReserver();
  op_def.precision = CalculationsPrecision::F32;
  op_def.src_tensors.push_back(tensor_reserver.Get(lhs_index.value()).descriptor);
  auto lhs_shape = tensor_reserver.Get(lhs_index.value()).shape;

  op_def.src_tensors.push_back(tensor_reserver.Get(rhs_index.value()).descriptor);
  auto rhs_shape = tensor_reserver.Get(rhs_index.value()).shape;

  op_def.dst_tensors.push_back(tensor_reserver.Get(ofm_index.value()).descriptor);
  auto out_shape = tensor_reserver.Get(ofm_index.value()).shape;

  std::unique_ptr<GPUOperation> gpu_op;

  std::vector<int> channels(2);
  channels[0] = lhs_shape.c;
  channels[1] = rhs_shape.c;
  SelectAdd(op_def, channels, out_shape.c, &gpu_op);

  auto fn = std::make_unique<ClFunction>();
  auto ofm_tensor = _tensor_reg->getClTensor(ofm_index);
  auto lhs_tensor = _tensor_reg->getClTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getClTensor(rhs_index);
  gpu_op->SetSrc(lhs_tensor->handle(), 0);
  gpu_op->SetSrc(rhs_tensor->handle(), 1);
  gpu_op->SetDst(ofm_tensor->handle(), 0);

  fn->configure(std::move(gpu_op), _creation_context);

  // TODO Add cl_node and merge..

  // for (auto& gpu_op : gpu_subgraph.operations) {
  //   CLNode cl_node;
  //   cl_node.operation = std::move(gpu_op.operation);
  //   cl_node.inputs.resize(gpu_op.input_ids.size());
  //   for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
  //     int id = gpu_op.input_ids[j];
  //     if (id >= 0) {
  //       cl_node.inputs[j] = id;
  //     }
  //   }
  //   cl_node.outputs.resize(gpu_op.output_ids.size());
  //   for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
  //     int id = gpu_op.output_ids[j];
  //     if (id >= 0) {
  //       cl_node.outputs[j] = id;
  //     }
  //   }
  //   nodes_.push_back(std::move(cl_node));
  // }

  _return_fn = std::move(fn);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
