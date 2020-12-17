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

#include <cker/operation/SoftMax.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/Softmax.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace interp
{
namespace
{

void prepareSoftMax(ExecEnv *env, const ir::Operation &node)
{
  const auto in_index = node.getInputs().at(0);
  const auto out_index = node.getOutputs().at(0);

  const auto in_tensor = env->tensorAt(in_index);
  UNUSED_RELEASE(in_tensor);

  assert((in_tensor->getShape().rank() == 4) || (in_tensor->getShape().rank() == 2));

  // Output shape should be same with input
  // Output type is pre-defined in model
  const auto output_shape = env->graph().operands().at(in_index).info().shape();
  const auto output_type = env->graph().operands().at(out_index).info().typeInfo();

  const auto output_info = ir::OperandInfo::createStaticInfo(output_shape, output_type);
  env->allocateIfNeeded(out_index, output_info);

  auto out_tensor = env->tensorAt(out_index);
  UNUSED_RELEASE(out_tensor);

  // Check output shape is same with input
  assert(out_tensor->getShape().rank() == out_tensor->getShape().rank());
  for (int32_t i = 0; i < in_tensor->getShape().rank(); i++)
  {
    assert(in_tensor->getShape().dim(i) == out_tensor->getShape().dim(i));
  }
}

void invoke(const ITensor *in_tensor, const ITensor *out_tensor,
            const ir::operation::Softmax::Param &param)
{
  const float *in_ptr = reinterpret_cast<const float *>(in_tensor->bufferRO());
  float *out_ptr = reinterpret_cast<float *>(out_tensor->buffer());

  float beta = param.beta;

  if (in_tensor->getShape().rank() == 2)
  {
    uint32_t batch_size = in_tensor->getShape().dim(0);
    uint32_t input_size = in_tensor->getShape().dim(1);

    nnfw::cker::Softmax(in_ptr, input_size, batch_size, beta, out_ptr);
  }
  else if (in_tensor->getShape().rank() == 4)
  {
    const auto in_shape = convertShape(in_tensor->tensorInfo().shape());
    const auto out_shape = convertShape(out_tensor->tensorInfo().shape());

    nnfw::cker::SoftmaxParams cker_param;
    cker_param.beta = beta;

    nnfw::cker::Softmax(cker_param, in_shape, in_ptr, out_shape, out_ptr);
  }
  else
  {
    throw std::runtime_error{"Unsuported input dimension: support 2D or 4D"};
  }
}

void invokeSoftMax(const ExecEnv *env, const ir::Operation &node)
{
  const auto &softmax_node = nnfw::misc::polymorphic_downcast<const ir::operation::Softmax &>(node);

  const auto in_index = node.getInputs().at(0);
  const auto out_index = node.getOutputs().at(0);

  const auto in_tensor = env->tensorAt(in_index);
  const auto out_tensor = env->tensorAt(out_index);

  const auto in_data_type = in_tensor->data_type();
  const auto out_data_type = out_tensor->data_type();
  if ((in_data_type == ir::DataType::FLOAT32) && (out_data_type == ir::DataType::FLOAT32))
  {
    invoke(in_tensor, out_tensor, softmax_node.param());
  }
  else
  {
    throw std::runtime_error{"NYI: Support float32 only"};
  }
}

} // namespace

OpKernel *getSoftmax()
{
  static OpKernel kernel = {prepareSoftMax, invokeSoftMax};
  return &kernel;
}

} // namespace interp
} // namespace onert
