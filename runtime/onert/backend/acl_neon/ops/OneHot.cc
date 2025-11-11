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

void Validator::visit(const ir::operation::OneHot &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::OneHot &node)
{
  const auto out_idx{node.getOutputs().at(0)};
  const auto indices_idx{node.getInputs().at(ir::operation::OneHot::Input::INDICES)};
  const auto depth_idx{node.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto onvalue_idx{node.getInputs().at(ir::operation::OneHot::Input::ON_VALUE)};
  const auto offvalue_idx{node.getInputs().at(ir::operation::OneHot::Input::OFF_VALUE)};

  auto output_tensor = _tensor_reg->getAclTensor(out_idx);
  auto indices_tensor = _tensor_reg->getAclTensor(indices_idx);
  auto depth_tensor = _tensor_reg->getAclTensor(depth_idx);
  auto onvalue_tensor = _tensor_reg->getAclTensor(onvalue_idx);
  auto offvalue_tensor = _tensor_reg->getAclTensor(offvalue_idx);

  const size_t output_rank = _ctx.at(out_idx).shape().rank();
  int32_t axis = node.param().axis == -1 ? output_rank - 1 : node.param().axis;
  axis = acl_common::ToARMComputeAxis(output_rank, axis).value();

  auto fn = acl_common::generateLayer<arm_compute::NEOneHot>(
    indices_tensor->handle(), depth_tensor->handle(), onvalue_tensor->handle(),
    offvalue_tensor->handle(), output_tensor->handle(), axis);
  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
