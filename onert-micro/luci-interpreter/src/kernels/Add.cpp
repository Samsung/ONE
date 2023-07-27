/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Builders.h"
#include "kernels/Utils.h"

#include "kernels/BinaryOpCommon.h"

#include "PALAdd.h"

namespace luci_interpreter
{

void configure_kernel_CircleAdd(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::TISOKernel kernel(cur_op, runtime_graph);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input1()) ==
                         Tensor::element_type(kernel.input2()));
  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input1()) ==
                         Tensor::element_type(kernel.input2()));

#ifndef DIS_QUANT
  if (Tensor::element_type(kernel.input1()) == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(Tensor::zero_points(kernel.input1())->size() == 1 &&
                           Tensor::zero_points(kernel.input2())->size() == 1);
    LUCI_INTERPRETER_CHECK(Tensor::zero_point(kernel.input1()) == 0 &&
                           Tensor::zero_point(kernel.input2()) == 0 &&
                           Tensor::zero_point(kernel.output()) == 0);
  }
#endif // DIS_QUANT
}

void execute_kernel_CircleAdd(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::TISOKernel kernel(cur_op, runtime_graph);

  const auto *options = cur_op->builtin_options_as_AddOptions();

  luci_interpreter::RuntimeShape input_shape1 =
    kernels::getTensorRuntimeShape(kernel.input1(), runtime_graph);
  luci_interpreter::RuntimeShape input_shape2 =
    kernels::getTensorRuntimeShape(kernel.input2(), runtime_graph);

  bool is_inplace = runtime_graph->is_inplace_op(cur_op);

  const auto status = runtime_graph->getOperatorStatus(cur_op);

  switch (Tensor::element_type(kernel.input1()))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
    {
      auto tiso_func = luci_interpreter_pal::Add<float>;
      auto broadcast_tiso_func = luci_interpreter_pal::BroadcastAdd4DSlow<float>;
      if (is_inplace)
      {
        kernels::evalTISOInplaceKernel<float>(tiso_func, broadcast_tiso_func, &kernel, options,
                                              std::move(input_shape1), std::move(input_shape2),
                                              status);
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOKernel<float>(tiso_func, broadcast_tiso_func, &kernel, &kernel_data,
                                       options, std::move(input_shape1), std::move(input_shape2),
                                       status);
      }
    }
    break;
#endif // DIS_FLOAT
    case DataType::S64:
    {
      auto tiso_func = luci_interpreter_pal::Add<int64_t>;
      auto broadcast_tiso_func = luci_interpreter_pal::BroadcastAdd4DSlow<int64_t>;
      if (is_inplace)
      {
        kernels::evalTISOInplaceKernel<int64_t>(tiso_func, broadcast_tiso_func, &kernel, options,
                                                std::move(input_shape1), std::move(input_shape2),
                                                status);
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOKernel<int64_t>(tiso_func, broadcast_tiso_func, &kernel, &kernel_data,
                                         options, std::move(input_shape1), std::move(input_shape2),
                                         status);
      }
    }
    break;
    case DataType::S32:
    {
      auto tiso_func = luci_interpreter_pal::Add<int32_t>;
      auto broadcast_tiso_func = luci_interpreter_pal::BroadcastAdd4DSlow<int32_t>;
      if (is_inplace)
      {
        kernels::evalTISOInplaceKernel<int32_t>(tiso_func, broadcast_tiso_func, &kernel, options,
                                                std::move(input_shape1), std::move(input_shape2),
                                                status);
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOKernel<int32_t>(tiso_func, broadcast_tiso_func, &kernel, &kernel_data,
                                         options, std::move(input_shape1), std::move(input_shape2),
                                         status);
      }
    }
    break;
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
