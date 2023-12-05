/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ops/BulkLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>
#include <exec/DynamicShapeInferer.h>

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace trix
{

KernelGenerator::KernelGenerator(const ir::Graph &graph,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder,
                                 const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
                                 const std::shared_ptr<DevContext> &dev_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()), _operations_ctx{graph.operations()},
    _tensor_builder(tensor_builder), _tensor_reg{tensor_reg}, _dev_context{dev_context}
{
  // DO NOTHING
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

void KernelGenerator::visit(const ir::operation::Bulk &node)
{
  using ir::operation::Bulk;

  std::vector<IPortableTensor *> output_tensors;
  for (const auto &ofm_idx : node.getOutputs())
    output_tensors.emplace_back(_tensor_reg->getPortableTensor(ofm_idx));

  std::vector<const IPortableTensor *> input_tensors;
  for (const auto &ifm_idx : node.getInputs())
    input_tensors.emplace_back(_tensor_reg->getPortableTensor(ifm_idx));

  // parameters
  const auto binary_path = node.param().binary_path;

  auto fn = std::make_unique<ops::BulkLayer>();

  fn->configure(input_tensors, output_tensors, binary_path, _dev_context);

  _return_fn = std::move(fn);
}

} // namespace trix
} // namespace backend
} // namespace onert
