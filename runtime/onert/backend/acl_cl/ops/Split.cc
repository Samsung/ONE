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

void Validator::visit(const ir::operation::Split &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Split &node)
{
  const auto ifm_index{node.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Split::Input::AXIS)};

  assert(node.param().num_splits == static_cast<int>(node.getOutputs().size()));
  if (!_ctx.at(axis_index).isConstant())
  {
    throw std::runtime_error("Non-constant axis_index NYI for acl_cl backend");
  }

  const auto ifm_rank = _ctx.at(ifm_index).shape().rank();
  std::vector<ir::OperandIndex> output_indexes;
  for (const auto &output : node.getOutputs())
    output_indexes.emplace_back(output);

  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);
  std::vector<arm_compute::ICLTensor *> output_tensors;
  for (const auto &ofm_ind : output_indexes)
    output_tensors.emplace_back(_tensor_reg->getAclTensor(ofm_ind)->handle());

  auto axis = _ctx.at(axis_index).asScalar<int32_t>();
  if (axis < 0)
    axis += ifm_rank;
  axis = acl_common::ToARMComputeAxis(ifm_rank, axis).value();

  auto fn =
    acl_common::generateLayer<arm_compute::CLSplit>(ifm_tensor->handle(), output_tensors, axis);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
