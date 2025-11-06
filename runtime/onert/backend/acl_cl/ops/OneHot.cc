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

void Validator::visit(const ir::operation::OneHot &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::OneHot &node)
{
  const auto output_idx{node.getOutputs().at(0)};
  const auto indices_idx{node.getInputs().at(ir::operation::OneHot::Input::INDICES)};
  const auto depth_idx{node.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto onvalue_idx{node.getInputs().at(ir::operation::OneHot::Input::ON_VALUE)};
  const auto offvalue_idx{node.getInputs().at(ir::operation::OneHot::Input::OFF_VALUE)};
  const auto depth = _ctx.at(depth_idx).asScalar<int32_t>();
  assert(depth > 0);

  auto output_tensor = _tensor_reg->getAclTensor(output_idx);
  auto indices_tensor = _tensor_reg->getAclTensor(indices_idx);
  auto onvalue_tensor = _tensor_reg->getAclTensor(onvalue_idx);

  const size_t output_rank = _ctx.at(output_idx).shape().rank();
  int32_t axis = node.param().axis == -1 ? output_rank - 1 : node.param().axis;
  axis = acl_common::ToARMComputeAxis(output_rank, axis).value();

  if (output_tensor->num_dimensions() != output_tensor->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and output_tensor is applied dim_correction
    acl_common::disableDimCorrection(output_tensor);
  }

  std::unique_ptr<::arm_compute::IFunction> fn;
  const auto &offvalue = _ctx.at(offvalue_idx);
  if (offvalue.isConstant())
  {
    fn = acl_common::generateLayer<arm_compute::CLOneHot>(
      indices_tensor->handle(), onvalue_tensor->handle(), output_tensor->handle(),
      acl_common::asPixelValue(offvalue), static_cast<uint32_t>(depth), axis);
  }
  else
  {
    auto offvalue_tensor = _tensor_reg->getAclTensor(offvalue_idx);
    fn = acl_common::generateLayer<arm_compute::CLOneHot>(
      indices_tensor->handle(), onvalue_tensor->handle(), offvalue_tensor->handle(),
      output_tensor->handle(), static_cast<uint32_t>(depth), axis);
  }

  if (output_tensor->dimension(0) == 1)
  {
    acl_common::enableDimCorrection(output_tensor);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
