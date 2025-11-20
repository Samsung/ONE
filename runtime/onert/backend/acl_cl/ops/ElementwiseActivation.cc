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

void Validator::visit(const ir::operation::ElementwiseActivation &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::ElementwiseActivation &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::ElementwiseActivation::Input::INPUT)};

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);

  const ::arm_compute::ActivationLayerInfo act_info =
    acl_common::asActivationLayerInfo(node.param().op_type, node.param().alpha, node.param().beta);

  auto fn = acl_common::generateLayer<arm_compute::CLActivationLayer>(
    ifm_tensor->handle(), ofm_tensor->handle(), act_info);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
