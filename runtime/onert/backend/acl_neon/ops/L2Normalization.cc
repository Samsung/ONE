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

void Validator::visit(const ir::operation::L2Normalization &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::L2Normalization &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::L2Normalization::Input::INPUT)};

  // {CL|Neon}L2Normalization performs the reduction only along dimension 0
  // L2 Normalization always performs the reduction along the depth axis
  // Thus, we repurpose {CL|Neon}NormalizationLayers to act as depthwise L2 normalizations by
  // choosing normalization parameters as below

  const auto &ifm_shape = _ctx.at(ifm_index).shape();
  // TODO Support optional constant dimension that normalization would be performed on
  const auto normalization_axis = _ctx.at(ifm_index).shape().rank() - 1;
  int32_t radius =
    2 * ifm_shape.dim(normalization_axis) + 1; // normSize = depth(last dimension) * 2 + 1
  float alpha = 1.0f;                          // In the implementation to make alpha_ become 1
  float beta = 0.5f;                           // pow(reduction, -0.5) = 1 / sqrt(reduction)
  float bias = 0.0f;                           // Don't offset the reduction.

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);

  const auto norm_info = ::arm_compute::NormalizationLayerInfo(::arm_compute::NormType::CROSS_MAP,
                                                               radius, alpha, beta, bias, false);

  auto fn = acl_common::generateLayer<arm_compute::NENormalizationLayer>(
    ifm_tensor->handle(), ofm_tensor->handle(), norm_info);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
