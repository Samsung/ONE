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
#include <AclActivationBuilder.h>

namespace onert::backend::acl_cl
{

void Validator::visit(const ir::operation::InstanceNorm &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::InstanceNorm &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::InstanceNorm::Input::INPUT)};
  const auto gamma_index{node.getInputs().at(ir::operation::InstanceNorm::Input::GAMMA)};
  const auto beta_index{node.getInputs().at(ir::operation::InstanceNorm::Input::BETA)};

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);
  auto epsilon = node.param().epsilon;
  auto activation = node.param().activation;

  if (!_ctx.at(gamma_index).isConstant() || !_ctx.at(beta_index).isConstant())
    throw std::runtime_error{"Non-constant gamma or beta for acl_cl backend InstanceNorm"};

  auto gamma = _ctx.at(gamma_index).asScalar<float>();
  auto beta = _ctx.at(beta_index).asScalar<float>();
  auto fn = acl_common::generateLayer<arm_compute::CLInstanceNormalizationLayer>(
    ifm_tensor->handle(), ofm_tensor->handle(), gamma, beta, epsilon);

  using ActivationBuilder =
    acl_common::AclActivationBuilder<::arm_compute::ICLTensor, ::arm_compute::CLActivationLayer,
                                     acl_common::AclFunction>;

  _return_fn = std::make_unique<exec::FunctionSequence>(
    acl_common::asAclFunction(std::move(fn)),
    ActivationBuilder::generate(activation, ofm_tensor->handle()));
}

} // namespace onert::backend::acl_cl
