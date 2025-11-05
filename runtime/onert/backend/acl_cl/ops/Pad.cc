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

void Validator::visit(const ir::operation::Pad &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto pad_index{node.getInputs().at(ir::operation::Pad::Input::PAD)};
  const auto output_index{node.getOutputs().at(0)};
  assert(_ctx.at(pad_index).data());

  auto rank = _ctx.at(input_index).shape().rank();
  auto pad_base = _ctx.at(pad_index).data()->base();

  auto input_type = _ctx.at(input_index).typeInfo();
  auto data_type = acl_common::asDataType(input_type.type());
  auto quant_info = ::arm_compute::QuantizationInfo(input_type.scale(), input_type.zero_point());
  const auto pixel_value = ::arm_compute::PixelValue(0, data_type, quant_info);

  auto input = _tensor_reg->getAclTensor(input_index)->handle();
  auto output = _tensor_reg->getAclTensor(output_index)->handle();

  ::arm_compute::PaddingList padding_list;
  padding_list.resize(rank);
  for (int32_t n = 0; n < rank; ++n)
  {
    const int32_t *from = reinterpret_cast<const int32_t *>(pad_base) + (n * 2);

    const auto axis = acl_common::ToARMComputeAxis(rank, n).value();
    padding_list[axis] = ::arm_compute::PaddingInfo{from[0], from[1]};
  }

  // Disable applied dim_correction
  const auto &input_tensor = _tensor_reg->getAclTensor(input_index);
  if (input_tensor->num_dimensions() != input_tensor->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and input tensor is applied dim_correction
    acl_common::disableDimCorrection(input_tensor);
  }

  auto fn =
    acl_common::generateLayer<arm_compute::CLPadLayerEx>(input, output, padding_list, pixel_value);

  // NOTE Do not revert disabling applied dim_correction for 4D.
  // It would produce a mistach of result by incorrect offset_first_element in
  // ICLKernel::add_tensor_argument<3>().
  // We have to disable applied dim_correction and not to revert enabling for the kernel that slices
  // 4D to 3D because slicing arm_compute::Window can causes incorrect offset_first_element if the
  // used tensor is 4D and the tensor's high dimention is 1
  if (input_tensor->num_dimensions() < 4 && input_tensor->dimension(0) == 1)
  {
    acl_common::enableDimCorrection(input_tensor);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
