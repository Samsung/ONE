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

void Validator::visit(const ir::operation::Reverse &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Reverse &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Reverse::Input::INPUT)};
  const auto axis_index{node.getInputs().at(ir::operation::Reverse::Input::AXIS)};

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);
  auto axis_tensor = _tensor_reg->getAclTensor(axis_index);

  // WORKAROUND: acl-cl backend only allow U32 type for axis
  //             ConstantInitializer will resolve S32 type to U32 type
  if (_ctx.at(axis_index).isConstant() &&
      (axis_tensor->handle()->info()->data_type() == arm_compute::DataType::S32))
  {
    axis_tensor->handle()->info()->set_data_type(arm_compute::DataType::U32);
  }

  auto fn = acl_common::generateLayer<arm_compute::CLReverse>(
    ifm_tensor->handle(), ofm_tensor->handle(), axis_tensor->handle(), false);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
