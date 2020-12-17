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

#include <cker/operation/AveragePool.h>
#include <cker/operation/MaxPool.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/Pool2D.h"
#include "util/Utils.h"
#include "util/ShapeInference.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace interp
{
namespace pool2d
{

void preparePool2D(ExecEnv *env, const ir::Operation &node)
{
  const auto &pool_node = nnfw::misc::polymorphic_downcast<const ir::operation::Pool2D &>(node);
  const auto in_index = node.getInputs().at(pool_node.INPUT);
  const auto out_index = node.getOutputs().at(0);

  const auto in_tensor = env->tensorAt(in_index);
  UNUSED_RELEASE(in_tensor);

  assert(in_tensor->getShape().rank() == 4);

  const auto output_info = env->graph().operands().at(out_index).info();
  if (output_info.total_size() == 0)
  {
    // Handle unspecified output shape
    const auto infered_output_shape =
      shape_inference::inferPoolShape(in_tensor->tensorInfo().shape(), pool_node.param());
    env->allocateIfNeeded(
      out_index, ir::OperandInfo::createStaticInfo(infered_output_shape, output_info.typeInfo()));
  }
  else
  {
    env->allocateIfNeeded(out_index, output_info);
  }

  auto out_tensor = env->tensorAt(out_index);
  UNUSED_RELEASE(out_tensor);

  // Handle same ifm & ofm data type only
  assert(in_tensor->data_type() == out_tensor->data_type());
  assert(out_tensor->getShape().rank() == 4);
}

template <typename T>
void invoke(const nnfw::cker::PoolParams &params, const nnfw::cker::Shape &in_shape,
            const T *in_ptr, const nnfw::cker::Shape &out_shape, T *out_ptr,
            ir::operation::Pool2D::PoolType op_type)
{
  switch (op_type)
  {
    case ir::operation::Pool2D::PoolType::AVG:
      nnfw::cker::AveragePool<T>(params, in_shape, in_ptr, out_shape, out_ptr);
      break;
    case ir::operation::Pool2D::PoolType::MAX:
      nnfw::cker::MaxPool<T>(params, in_shape, in_ptr, out_shape, out_ptr);
      break;
    default:
      throw std::runtime_error{"Interp(Pool2D): NYI unsupported operation"};
      break;
  }
}

void invokePool2DOps(const ExecEnv *env, const ir::Operation &node)
{
  const auto &pool_node = nnfw::misc::polymorphic_downcast<const ir::operation::Pool2D &>(node);

  const auto in_index = node.getInputs().at(0);
  const auto out_index = node.getOutputs().at(0);

  // Check lhs shape is same with rhs (with broadcast)
  const auto in_tensor = env->tensorAt(in_index);
  const auto out_tensor = env->tensorAt(out_index);

  // TODO support NCHW frontend
  const auto ifm_shape = in_tensor->tensorInfo().shape().asFeature(ir::Layout::NHWC);
  const auto ofm_shape = out_tensor->tensorInfo().shape().asFeature(ir::Layout::NHWC);
  const auto param = pool_node.param();
  const auto padding =
    ir::calculatePadding(param.padding, ifm_shape, ofm_shape, param.stride, param.kw, param.kh);
  // Calculate
  nnfw::cker::PoolParams cker_param;
  cker_param.filter_width = param.kw;
  cker_param.filter_height = param.kh;
  cker_param.padding_values.width = padding.left;
  cker_param.padding_values.height = padding.top;
  cker_param.stride_width = param.stride.horizontal;
  cker_param.stride_height = param.stride.vertical;

  const auto data_type = in_tensor->data_type();
  if (data_type == ir::DataType::FLOAT32)
  {
    calculateActivationRange(param.activation, &cker_param.float_activation_min,
                             &cker_param.float_activation_max);

    const auto in_shape = convertShape(in_tensor->tensorInfo().shape());
    const auto out_shape = convertShape(out_tensor->tensorInfo().shape());
    const float *in_ptr = reinterpret_cast<const float *>(in_tensor->bufferRO());
    float *out_ptr = reinterpret_cast<float *>(out_tensor->buffer());
    // Now, invoke() supports only Pool2D in float
    invoke<float>(cker_param, in_shape, in_ptr, out_shape, out_ptr, param.op_type);
  }
  else
  {
    throw std::runtime_error{"NYI: Support float only"};
  }
}
} // namespace pool2d

OpKernel *getPool2D()
{
  static OpKernel kernel = {pool2d::preparePool2D, pool2d::invokePool2DOps};
  return &kernel;
}

} // namespace interp
} // namespace onert
