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

void Validator::visit(const ir::operation::TransposeConv &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::TransposeConv &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ker_index{node.getInputs().at(ir::operation::TransposeConv::Input::KERNEL)};
  const auto ifm_index{node.getInputs().at(ir::operation::TransposeConv::Input::INPUT)};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature();

  const auto stride = node.param().stride;

  assert((node.param().padding.type == ir::PaddingType::SAME) ||
         (node.param().padding.type == ir::PaddingType::VALID));
  auto padding = ir::calculatePadding(node.param().padding, ofm_shape, ifm_shape, stride,
                                      ker_shape.W, ker_shape.H);

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getAclTensor(ker_index);

  const auto tconv_info = acl_common::asPadStrideInfo(padding, stride);

  auto fn = acl_common::generateLayer<arm_compute::NEDeconvolutionLayer>(
    ifm_tensor->handle(), ker_tensor->handle(), nullptr, ofm_tensor->handle(), tconv_info);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
