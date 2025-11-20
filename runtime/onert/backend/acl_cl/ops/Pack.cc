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

void Validator::visit(const ir::operation::Pack &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Pack &node)
{
  const auto output_index{node.getOutputs().at(0)};
  auto axis{node.param().axis};

  const auto output_rank = _ctx.at(output_index).shape().rank();

  std::vector<ir::OperandIndex> input_indexes;
  for (const auto &input_index : node.getInputs())
    input_indexes.emplace_back(input_index);

  auto output = _tensor_reg->getAclTensor(output_index)->handle();
  std::vector<arm_compute::ICLTensor *> inputs;
  for (const auto &input_index : input_indexes)
    inputs.emplace_back(_tensor_reg->getAclTensor(input_index)->handle());

  if (axis < 0)
    axis += output_rank;
  axis = acl_common::ToARMComputeAxis(output_rank, axis).value();

  // Disable applied dim_correction
  for (const auto &input_index : input_indexes)
  {
    const auto &input_tensor = _tensor_reg->getAclTensor(input_index);
    if (input_tensor->num_dimensions() != input_tensor->info()->num_dimensions())
    {
      // This means that high dimension's value is 1 and input tensor is applied dim_correction
      acl_common::disableDimCorrection(input_tensor);
    }
  }

  auto fn = acl_common::generateLayer<arm_compute::CLStackLayer>(inputs, axis, output);

  // Revert disabling applied dim_correction
  for (const auto &input_index : input_indexes)
  {
    const auto &input_tensor = _tensor_reg->getAclTensor(input_index);
    if (input_tensor->dimension(0) == 1)
    {
      acl_common::enableDimCorrection(input_tensor);
    }
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
