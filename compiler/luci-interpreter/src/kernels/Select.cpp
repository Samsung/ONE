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

#include "kernels/Select.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>
// TODO use select.h when version up
// #include <tensorflow/lite/kernels/internal/reference/select.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Select::Select(const Tensor *condition, const Tensor *t, const Tensor *e, Tensor *output)
  : Kernel({condition, t, e}, {output})
{
  // NOTE _requires_broadcast is for SelectV2
  _requires_broadcast = false;
  _has_low_rank_input_condition = false;
}

void Select::configure()
{
  LUCI_INTERPRETER_CHECK(condition()->element_type() == DataType::BOOL);
  LUCI_INTERPRETER_CHECK(t()->element_type() == e()->element_type());
  LUCI_INTERPRETER_CHECK(t()->element_type() == output()->element_type());

  auto cond_shape = condition()->shape();
  auto cond_num_dims = cond_shape.num_dims();
  auto t_shape = t()->shape();

  bool is_input_condition_scalar = cond_num_dims == 0;
  bool has_rank_one_input_condition = cond_num_dims == 1 && cond_shape.dim(0) == t_shape.dim(0);

  _has_low_rank_input_condition = is_input_condition_scalar || has_rank_one_input_condition;

  output()->resize(calculateShapeForBroadcast(t()->shape(), e()->shape()));
}

void Select::execute() const
{
  switch (t()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-intp Select unsupported type.");
  }
}

void Select::evalFloat() const
{
  const auto condition_shape = getTensorShape(condition());
  const auto condition_data = getTensorData<bool>(condition());
  const auto t_shape = getTensorShape(t());
  const auto t_data = getTensorData<float>(t());
  const auto e_shape = getTensorShape(e());
  const auto e_data = getTensorData<float>(e());
  const auto output_shape = getTensorShape(output());
  auto output_data = getTensorData<float>(output());

  if (_has_low_rank_input_condition)
  {
    tflite::reference_ops::RankOneSelect(condition_shape, condition_data, t_shape, t_data, e_shape,
                                         e_data, output_shape, output_data);
  }
  else if (_requires_broadcast)
  {
    // TODO support broadcast kernel when upgrade to TF2.10.x or above
    assert(false);
  }
  else
  {
    tflite::reference_ops::Select(condition_shape, condition_data, t_shape, t_data, e_shape, e_data,
                                  output_shape, output_data);
  }
}

} // namespace kernels
} // namespace luci_interpreter
