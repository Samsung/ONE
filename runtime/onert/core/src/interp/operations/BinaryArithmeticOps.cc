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

#include <cker/operation/BinaryArithmeticOps.h>

#include "OperationUtil.h"

#include "interp/Registration.h"
#include "ir/operation/Add.h"
#include "ir/operation/Sub.h"
#include "ir/operation/Mul.h"
#include "misc/polymorphic_downcast.h"
#include "cker/Types.h"

namespace onert
{
namespace interp
{
namespace
{

enum class OpType
{
  ADD,
  SUB,
  MUL
};

template <typename node_type> void prepareAdd(ExecEnv *env, const ir::Operation &node)
{
  const auto &add_node = nnfw::misc::polymorphic_downcast<const node_type &>(node);

  const auto lhs_index = node.getInputs().at(add_node.LHS);
  const auto rhs_index = node.getInputs().at(add_node.RHS);
  const auto out_index = node.getOutputs().at(0);

  const auto lhs_tensor = env->tensorAt(lhs_index);
  const auto rhs_tensor = env->tensorAt(rhs_index);

  // Check shape and type lhs is same with rhs
  // TODO Util function to compare TensorInfo
  if (lhs_tensor->data_type() != rhs_tensor->data_type())
  {
    throw std::runtime_error{"Interp(Add): Different input types"};
  }

  bool try_broadcast = (lhs_tensor->tensorInfo().shape() != rhs_tensor->tensorInfo().shape());
  if (try_broadcast)
  {
    bool success = true;
    auto out_shape = calcBroadcastShape(lhs_tensor->tensorInfo().shape(),
                                        rhs_tensor->tensorInfo().shape(), success);
    if (!success)
    {
      throw std::runtime_error{"Interp(Add): Fail to brodcasting"};
    }

    auto output_info =
        ir::OperandInfo::createStaticInfo(out_shape, lhs_tensor->tensorInfo().typeInfo());
    // We can handle already allocated (ex. model output)
    env->allocateIfNeeded(out_index, output_info);
  }
  else
  {
    // Output's shape and type is same with input
    auto output_info = lhs_tensor->tensorInfo();
    // We can handle already allocated (ex. model output)
    env->allocateIfNeeded(out_index, output_info);
  }

  auto out_tensor = env->tensorAt(out_index);
  // Check shape and type lhs is same with output
  // TODO Util function to compare TensorInfo
  if (lhs_tensor->data_type() != out_tensor->data_type())
  {
    throw std::runtime_error{"Interp(Add): Invalid output type"};
  }
}

inline void setActivationParams(float min, float max, nnfw::cker::BinaryArithmeticOpParam *params)
{
  params->float_activation_min = min;
  params->float_activation_max = max;
}

inline void setActivationParams(int32_t min, int32_t max,
                                nnfw::cker::BinaryArithmeticOpParam *params)
{
  params->quantized_activation_min = min;
  params->quantized_activation_max = max;
}

template <typename raw_type, typename param_type, OpType op_type>
void invoke(const ITensor *lhs_tensor, const ITensor *rhs_tensor, const ITensor *out_tensor,
            const param_type &param)
{
  const auto lhs_buffer = lhs_tensor->bufferRO();
  const auto rhs_buffer = rhs_tensor->bufferRO();
  auto out_buffer = out_tensor->buffer();

  nnfw::cker::BinaryArithmeticOpParam cker_param;
  raw_type activation_min, activation_max;
  calculateActivationRange(param.activation, &activation_min, &activation_max);
  setActivationParams(activation_min, activation_max, &cker_param);
  const raw_type *lhs_ptr = reinterpret_cast<const raw_type *>(lhs_buffer);
  const raw_type *rhs_ptr = reinterpret_cast<const raw_type *>(rhs_buffer);
  raw_type *out_ptr = reinterpret_cast<raw_type *>(out_buffer);

  const auto cker_op_type =
      (op_type == OpType::ADD)
          ? nnfw::cker::BinaryArithmeticOpType::ADD
          : ((op_type == OpType::SUB) ? nnfw::cker::BinaryArithmeticOpType::SUB
                                      : nnfw::cker::BinaryArithmeticOpType::MUL);

  const bool need_broadcast = nnfw::cker::ProcessBroadcastShapes(
      convertShape(lhs_tensor->tensorInfo().shape()),
      convertShape(rhs_tensor->tensorInfo().shape()), &cker_param);

  if (need_broadcast)
  {
    const auto lhs_shape = convertShape(lhs_tensor->tensorInfo().shape());
    const auto rhs_shape = convertShape(rhs_tensor->tensorInfo().shape());
    const auto out_shape = convertShape(out_tensor->tensorInfo().shape());
    nnfw::cker::BroadcastBinaryArithmeticOp<cker_op_type>(cker_param, lhs_shape, lhs_ptr, rhs_shape,
                                                          rhs_ptr, out_shape, out_ptr);
    return;
  }

  const auto lhs_shape = convertShape(lhs_tensor->tensorInfo().shape());
  const auto rhs_shape = convertShape(rhs_tensor->tensorInfo().shape());
  const auto out_shape = convertShape(out_tensor->tensorInfo().shape());
  nnfw::cker::BinaryArithmeticOp<cker_op_type>(cker_param, lhs_shape, lhs_ptr, rhs_shape, rhs_ptr,
                                               out_shape, out_ptr);
}

template <typename node_type, typename param_type, OpType op_type>
void invokeAdd(const ExecEnv *env, const ir::Operation &node)
{
  const auto &arithmetic_node = nnfw::misc::polymorphic_downcast<const node_type &>(node);

  const auto lhs_index = node.getInputs().at(arithmetic_node.LHS);
  const auto rhs_index = node.getInputs().at(arithmetic_node.RHS);
  const auto out_index = node.getOutputs().at(0);
  const auto lhs_tensor = env->tensorAt(lhs_index);
  const auto rhs_tensor = env->tensorAt(rhs_index);
  const auto out_tensor = env->tensorAt(out_index);
  const auto data_type = lhs_tensor->data_type();

  if (data_type == ir::DataType::INT32)
  {
    invoke<int32_t, param_type, op_type>(lhs_tensor, rhs_tensor, out_tensor,
                                         arithmetic_node.param());
  }
  else if (data_type == ir::DataType::FLOAT32)
  {
    invoke<float, param_type, op_type>(lhs_tensor, rhs_tensor, out_tensor, arithmetic_node.param());
  }
  else
  {
    throw std::runtime_error{"NYI: Unsupported data type"};
  }
}
} // namespace

OpKernel *getAdd()
{
  static OpKernel kernel = {prepareAdd<ir::operation::Add>,
                            invokeAdd<ir::operation::Add, ir::operation::Add::Param, OpType::ADD>};
  return &kernel;
}

OpKernel *getSub()
{
  static OpKernel kernel = {prepareAdd<ir::operation::Sub>,
                            invokeAdd<ir::operation::Sub, ir::operation::Sub::Param, OpType::SUB>};
  return &kernel;
}

OpKernel *getMul()
{
  static OpKernel kernel = {prepareAdd<ir::operation::Mul>,
                            invokeAdd<ir::operation::Mul, ir::operation::Mul::Param, OpType::MUL>};
  return &kernel;
}

} // namespace interp
} // namespace onert
