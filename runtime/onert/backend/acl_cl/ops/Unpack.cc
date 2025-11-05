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

void Validator::visit(const ir::operation::Unpack &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Unpack::Input::INPUT)};
  auto axis{node.param().axis};

  const auto input_rank = _ctx.at(input_index).shape().rank();

  std::vector<ir::OperandIndex> output_indexes;
  for (const auto &output_index : node.getOutputs())
    output_indexes.emplace_back(output_index);

  auto input_tensor = _tensor_reg->getAclTensor(input_index);
  std::vector<arm_compute::ICLTensor *> outputs;
  for (const auto &output_index : output_indexes)
    outputs.emplace_back(_tensor_reg->getAclTensor(output_index)->handle());

  if (axis < 0)
    axis += input_rank;
  axis = acl_common::ToARMComputeAxis(input_rank, axis).value();

  // Disable applied dim_correction
  if (input_tensor->num_dimensions() != input_tensor->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and input tensor is applied dim_correction
    acl_common::disableDimCorrection(input_tensor);
  }

  auto fn =
    acl_common::generateLayer<arm_compute::CLUnstack>(input_tensor->handle(), outputs, axis);

  // Revert disabling applied dim_correction
  if (input_tensor->dimension(0) == 1)
  {
    acl_common::enableDimCorrection(input_tensor);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
