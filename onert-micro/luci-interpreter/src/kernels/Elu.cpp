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

#include "kernels/Elu.h"
#include "kernels/Utils.h"

#include "PALElu.h"

namespace luci_interpreter
{

namespace kernels
{

Elu::Elu(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Elu::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(input()->shape());
}

void Elu::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::Elu(getTensorShape(input()), getTensorData<float>(input()),
                                getTensorShape(output()), getTensorData<float>(output()));
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
