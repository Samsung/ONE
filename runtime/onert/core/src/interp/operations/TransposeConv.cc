/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/operation/TransposeConv.h>
#include <misc/polymorphic_downcast.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/TransposeConv.h"

namespace onert
{
namespace interp
{
namespace
{

void prepareTransposeConv(ExecEnv *env, const ir::Operation &node)
{
  const auto ifm_index = node.getInputs().at(ir::operation::TransposeConv::INPUT);
  const auto ker_index = node.getInputs().at(ir::operation::TransposeConv::KERNEL);
  const auto ofm_shape_index = node.getInputs().at(ir::operation::TransposeConv::OUTPUT_SHAPE);
  const auto ofm_index = node.getOutputs().at(0);

  const auto ifm_tensor = env->tensorAt(ifm_index);
  const auto ker_tensor = env->tensorAt(ker_index);
  const auto ofm_shape_tensor = env->tensorAt(ofm_shape_index);

  assert(ifm_tensor->getShape().rank() == 4);
  assert(ker_tensor->getShape().rank() == 4);
  assert(ofm_shape_tensor->getShape().rank() == 1);

  UNUSED_RELEASE(ifm_tensor);
  UNUSED_RELEASE(ker_tensor);
  UNUSED_RELEASE(ofm_shape_tensor);

  const auto output_info = env->graph().operands().at(ofm_index).info();
  if (output_info.total_size() == 0)
  {
    // TODO: Handle unspecified output shape
    throw std::runtime_error{"Interp(TConv): NYI unspecified output shape"};
  }
  else
  {
    env->allocateIfNeeded(ofm_index, output_info);
  }

  auto ofm_tensor = env->tensorAt(ofm_index);
  UNUSED_RELEASE(ofm_tensor);

  // Handle same ifm & ofm data type only
  if (ifm_tensor->data_type() != ofm_tensor->data_type())
  {
    throw std::runtime_error{"Interp(TConv): Different I/O data dype"};
  }

  if (ofm_tensor->getShape().rank() != 4)
  {
    throw std::runtime_error{"Interp(TConv): Invalid output rank"};
  }
}

void invoke(const ITensor *ifm_tensor, const ITensor *ker_tensor, const ITensor *ofm_tensor,
            const ir::operation::TransposeConv::Param &param)
{
  const auto ifm_shape = ifm_tensor->tensorInfo().shape().asFeature(ir::Layout::NHWC);
  const auto ofm_shape = ofm_tensor->tensorInfo().shape().asFeature(ir::Layout::NHWC);
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto ker_shape = ker_tensor->tensorInfo().shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);
  const auto padding =
    ir::calculatePadding(param.padding, ofm_shape, ifm_shape, param.stride, ker_width, ker_height);

  nnfw::cker::TransposeConvParams cker_param;
  cker_param.padding_values.width = padding.left;
  cker_param.padding_values.height = padding.top;
  cker_param.stride_width = param.stride.horizontal;
  cker_param.stride_height = param.stride.vertical;
  cker_param.dilation_width_factor = 1;
  cker_param.dilation_height_factor = 1;

  const auto cker_ifm_shape = convertShape(ifm_tensor->tensorInfo().shape());
  const auto cker_ker_shape = convertShape(ker_tensor->tensorInfo().shape());
  const auto cker_ofm_shape = convertShape(ofm_tensor->tensorInfo().shape());
  const float *ifm_ptr = reinterpret_cast<const float *>(ifm_tensor->bufferRO());
  const float *ker_ptr = reinterpret_cast<const float *>(ker_tensor->bufferRO());
  float *ofm_ptr = reinterpret_cast<float *>(ofm_tensor->buffer());

  nnfw::cker::TransposeConv(cker_param, cker_ifm_shape, ifm_ptr, cker_ker_shape, ker_ptr,
                            cker_ofm_shape, ofm_ptr);
}

void invokeTransposeConv(const ExecEnv *env, const ir::Operation &node)
{
  const auto &tconv_node =
    nnfw::misc::polymorphic_downcast<const ir::operation::TransposeConv &>(node);

  const auto ifm_index = node.getInputs().at(ir::operation::TransposeConv::INPUT);
  const auto ker_index = node.getInputs().at(ir::operation::TransposeConv::KERNEL);
  const auto ofm_index = node.getOutputs().at(0);

  const auto ifm_tensor = env->tensorAt(ifm_index);
  const auto ker_tensor = env->tensorAt(ker_index);
  const auto ofm_tensor = env->tensorAt(ofm_index);

  const auto data_type = ifm_tensor->data_type();
  if (data_type == ir::DataType::FLOAT32)
  {
    invoke(ifm_tensor, ker_tensor, ofm_tensor, tconv_node.param());
  }
  else
  {
    throw std::runtime_error{"Interp(TConv): Support float32 only"};
  }
}

} // namespace

OpKernel *getTransposeConv()
{
  static OpKernel kernel = {prepareTransposeConv, invokeTransposeConv};
  return &kernel;
}

} // namespace interp
} // namespace onert
