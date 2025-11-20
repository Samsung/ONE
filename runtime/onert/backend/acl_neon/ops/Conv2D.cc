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

void Validator::visit(const ir::operation::Conv2D &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Conv2D &node)
{
  using ir::operation::Conv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(Conv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto stride = node.param().stride;
  const auto padding =
    ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, ker_width, ker_height);
  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getAclTensor(ker_index);
  auto bias_tensor = _tensor_reg->getAclTensor(bias_index);

  const auto conv_info = acl_common::asPadStrideInfo(padding, stride);
  const auto act_info = acl_common::asActivationLayerInfo(activation);

  auto fn = acl_common::generateLayer<arm_compute::NEConvolutionLayer>(
    _tensor_builder->acl_tensor_manager()->internal_buffer_manager(), ifm_tensor->handle(),
    ker_tensor->handle(), bias_tensor->handle(), ofm_tensor->handle(), conv_info,
    ::arm_compute::WeightsInfo(), ::arm_compute::Size2D(1U, 1U), act_info);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
