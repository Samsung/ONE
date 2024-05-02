/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/SelectV2.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>
// TODO use select.h when version up
// #include <tensorflow/lite/kernels/internal/reference/select.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

SelectV2::SelectV2(const Tensor *condition, const Tensor *t, const Tensor *e, Tensor *output)
  : Kernel({condition, t, e}, {output})
{
}

void SelectV2::configure()
{
  LUCI_INTERPRETER_CHECK(condition()->element_type() == DataType::BOOL);
  LUCI_INTERPRETER_CHECK(t()->element_type() == e()->element_type());
  LUCI_INTERPRETER_CHECK(t()->element_type() == output()->element_type());

  auto cond_shape = condition()->shape();
  auto t_shape = t()->shape();
  auto e_shape = e()->shape();

  output()->resize(
    calculateShapeForBroadcast(cond_shape, calculateShapeForBroadcast(t_shape, e_shape)));
}

void SelectV2::execute() const
{
  auto t_type = t()->element_type();
  switch (t_type)
  {
    case DataType::FLOAT32:
      evaluate<float>();
      break;
    case DataType::S32:
      evaluate<int32_t>();
      break;
    default:
      throw std::runtime_error("luci-intp SelectV2 unsupported type.");
  }
}

template <typename T> void SelectV2::evaluate() const
{
  const auto condition_shape = getTensorShape(condition());
  const auto condition_data = getTensorData<bool>(condition());
  const auto t_shape = getTensorShape(t());
  const auto t_data = getTensorData<T>(t());
  const auto e_shape = getTensorShape(e());
  const auto e_data = getTensorData<T>(e());
  const auto output_shape = getTensorShape(output());
  auto output_data = getTensorData<T>(output());

  tflite::reference_ops::BroadcastSelect5DSlow<bool, T>(
    condition_shape, condition_data, t_shape, t_data, e_shape, e_data, output_shape, output_data);
}

} // namespace kernels
} // namespace luci_interpreter
