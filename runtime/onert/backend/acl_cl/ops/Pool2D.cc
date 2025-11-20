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

#include <AclActivationBuilder.h>
#include <AclKernelGen.h>

namespace onert::backend::acl_cl
{

void Validator::visit(const ir::operation::Pool2D &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Pool2D &node)
{
  auto raw_fn = acl_common::kernelGenPool2D<::arm_compute::CLPoolingLayer>(
    node, _ctx, _tensor_reg, acl_common::convertPoolType(node.param().op_type));

  const auto ofm_index{node.getOutputs().at(0)};
  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  const auto activation = node.param().activation;

  using ActivationBuilder =
    acl_common::AclActivationBuilder<::arm_compute::ICLTensor, ::arm_compute::CLActivationLayer,
                                     acl_common::AclFunction>;

  _return_fn = std::make_unique<exec::FunctionSequence>(
    acl_common::asAclFunction(std::move(raw_fn)),
    ActivationBuilder::generate(activation, ofm_tensor->handle()));
}

} // namespace onert::backend::acl_cl
