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

namespace onert::backend::acl_neon
{

using ActivationBuilder =
  acl_common::AclActivationBuilder<::arm_compute::ITensor, ::arm_compute::NEActivationLayer,
                                   acl_common::AclFunction>;

void Validator::visit(const ir::operation::FullyConnected &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::FullyConnected &node)
{
  const auto output_index{node.getOutputs().at(0)};
  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  const auto activation = node.param().activation;
  if (node.param().weights_format == ir::FullyConnectedWeightsFormat::Shuffled16x1Float32)
    throw std::runtime_error(
      "KernelGenerator(acl_neon): FullyConnected 16x1Float32 weights is not supported.");

  auto fn = acl_common::kernelGenFullyConnected<acl_common::AclFunction, ::arm_compute::ITensor,
                                                ::arm_compute::NEFullyConnectedReshapingLayer>(
    node, _ctx, _tensor_builder, _tensor_reg);
  _return_fn = std::make_unique<exec::FunctionSequence>(
    std::move(fn), ActivationBuilder::generate(activation, output_tensor->handle()));
}

} // namespace onert::backend::acl_neon
