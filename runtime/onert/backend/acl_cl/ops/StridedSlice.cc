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

void Validator::visit(const ir::operation::StridedSlice &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::StridedSlice &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  const auto starts_index{node.getInputs().at(ir::operation::StridedSlice::Input::STARTS)};
  const auto ends_index{node.getInputs().at(ir::operation::StridedSlice::Input::ENDS)};
  const auto strides_index{node.getInputs().at(ir::operation::StridedSlice::Input::STRIDES)};

  auto outputData_tensor = _tensor_reg->getAclTensor(output_index);
  auto inputData_tensor = _tensor_reg->getAclTensor(input_index);

  // Set initializers for indices data such as order of inputData
  int input_rank = _ctx.at(input_index).shape().rank();
  std::vector<int32_t> starts;
  std::vector<int32_t> ends;
  std::vector<int32_t> strides;
  starts.resize(input_rank, 0);
  ends.resize(input_rank, 0);
  strides.resize(input_rank, 0);
  {
    assert(_ctx.at(starts_index).data());
    assert(_ctx.at(ends_index).data());
    assert(_ctx.at(strides_index).data());
    auto startData_base = _ctx.at(starts_index).data()->base();
    auto endData_base = _ctx.at(ends_index).data()->base();
    auto stridesData_base = _ctx.at(strides_index).data()->base();
    [[maybe_unused]] const int startData_size = _ctx.at(starts_index).shape().num_elements();
    [[maybe_unused]] const int endData_size = _ctx.at(ends_index).shape().num_elements();
    [[maybe_unused]] const int stridesData_size = _ctx.at(strides_index).shape().num_elements();

    using ir::DataType;

    assert(_ctx.at(starts_index).typeInfo().type() == DataType::INT32);
    assert(_ctx.at(ends_index).typeInfo().type() == DataType::INT32);
    assert(_ctx.at(strides_index).typeInfo().type() == DataType::INT32);
    assert(startData_size == input_rank);
    assert(endData_size == input_rank);
    assert(stridesData_size == input_rank);

    assert(startData_base != nullptr);
    for (int n = 0; n < input_rank; ++n)
    {
      auto axis = ::onert::backend::acl_common::ToARMComputeAxis(input_rank, n).value();

      int32_t start_value = *(reinterpret_cast<const int32_t *>(startData_base) + n);
      starts[axis] = start_value;

      int32_t end_value = *(reinterpret_cast<const int32_t *>(endData_base) + n);
      ends[axis] = end_value;

      int32_t strides_value = *(reinterpret_cast<const int32_t *>(stridesData_base) + n);
      strides[axis] = strides_value;
    }
  }

  // Set mask bits such as order of inputData
  const auto begin_mask = acl_common::ReorderBits<int32_t>(node.param().begin_mask, input_rank);
  const auto end_mask = acl_common::ReorderBits<int32_t>(node.param().end_mask, input_rank);
  const auto shrink_axis_mask =
    acl_common::ReorderBits<int32_t>(node.param().shrink_axis_mask, input_rank);

  ::arm_compute::Coordinates starts_set;
  ::arm_compute::Coordinates ends_set;
  ::arm_compute::BiStrides strides_set;

  for (size_t i = 0; i < starts.size(); ++i)
  {
    starts_set.set(i, starts[i]);
    ends_set.set(i, ends[i]);
    strides_set.set(i, strides[i]);
  }

  // Disable applied dim_correction
  if (inputData_tensor->num_dimensions() != inputData_tensor->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and input tensor is applied dim_correction
    acl_common::disableDimCorrection(inputData_tensor);
  }

  auto fn = acl_common::generateLayer<arm_compute::CLStridedSlice>(
    inputData_tensor->handle(), outputData_tensor->handle(), starts_set, ends_set, strides_set,
    begin_mask, end_mask, shrink_axis_mask);

  // Revert disabling applied dim_correction
  if (inputData_tensor->dimension(0) == 1)
  {
    acl_common::enableDimCorrection(inputData_tensor);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
