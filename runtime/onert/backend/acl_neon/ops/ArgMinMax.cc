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

namespace onert::backend::acl_neon
{

void Validator::visit(const ir::operation::ArgMinMax &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::ArgMinMax &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ArgMinMax::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::ArgMinMax::Input::AXIS)};

  const auto ifm_rank = _ctx.at(ifm_index).shape().rank();

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);

  int axis_value = _ctx.at(axis_index).asScalar<int32_t>();
  if (axis_value < 0)
  {
    axis_value += ifm_rank;
  }
  assert(axis_value >= 0 && axis_value < ifm_rank);
  const auto fixed_axis = acl_common::ToARMComputeAxis(ifm_rank, axis_value).value();
  auto reduce_type = node.param().is_arg_max ? ::arm_compute::ReductionOperation::ARG_IDX_MAX
                                             : ::arm_compute::ReductionOperation::ARG_IDX_MIN;

  auto fn = acl_common::generateLayer<arm_compute::NEArgMinMaxLayer>(
    ifm_tensor->handle(), fixed_axis, ofm_tensor->handle(), reduce_type);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
