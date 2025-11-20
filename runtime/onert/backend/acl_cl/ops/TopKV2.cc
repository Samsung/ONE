/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "../KernelGenerator.h"
#include "../Validator.h"

#include <AclKernelGen.h>

namespace onert::backend::acl_cl
{

void Validator::visit(const ir::operation::TopKV2 &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::TopKV2 &node)
{
  const auto outputValues_index{node.getOutputs().at(ir::operation::TopKV2::Output::OUTPUT_VALUES)};
  const auto outputIndices_index{
    node.getOutputs().at(ir::operation::TopKV2::Output::OUTPUT_INDICES)};

  const auto inputData_index{node.getInputs().at(ir::operation::TopKV2::Input::INPUT)};

  // Currently, we only support the vector input.
  assert(_ctx.at(inputData_index).shape().rank() == 1 ||
         _ctx.at(inputData_index).shape().rank() == 2);

  const auto k = node.param().k;

  auto values_tensor = _tensor_reg->getAclTensor(outputValues_index);
  auto indices_tensor = _tensor_reg->getAclTensor(outputIndices_index);
  auto input_tensor = _tensor_reg->getAclTensor(inputData_index);

  auto fn = acl_common::generateLayer<arm_compute::CLTopKV2>(
    input_tensor->handle(), k, values_tensor->handle(), indices_tensor->handle());

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
