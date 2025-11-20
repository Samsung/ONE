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

void Validator::visit(const ir::operation::Reduce &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Reduce &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reduce::Input::INPUT)};
  const auto axes_index{node.getInputs().at(ir::operation::Reduce::Input::AXES)};
  const auto keep_dims{node.param().keep_dims};
  const auto reduce_type = node.param().reduce_type;

  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  auto input_tensor = _tensor_reg->getAclTensor(input_index);

  // Convert to ACL axes taking into account negative values and possible duplicates.
  const auto &axes = _ctx.at(axes_index);
  const auto input_rank = _ctx.at(input_index).shape().rank();

  std::unique_ptr<arm_compute::IFunction> fn;
  if (reduce_type == ir::operation::Reduce::ReduceType::MEAN)
  {
    const auto acl_axes = acl_common::asCoordinates(axes, input_rank);
    fn = acl_common::generateLayer<arm_compute::CLReduceMean>(input_tensor->handle(), acl_axes,
                                                              keep_dims, output_tensor->handle());
  }
  else
  {
    const auto acl_axes = acl_common::asSet(axes, input_rank);

    fn = acl_common::generateLayer<arm_compute::CLReduceOperation>(
      _tensor_builder->acl_tensor_manager()->internal_buffer_manager(), input_tensor->handle(),
      output_tensor->handle(), acl_axes, keep_dims, acl_common::convertReduceType(reduce_type));
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
