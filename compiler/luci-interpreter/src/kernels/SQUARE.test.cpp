/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/SQUARE.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(SquareTest, Float)
{
  Shape input_shape{3, 1, 2};
  std::vector<float> input_data1{1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data1);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Square kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{1.0, 0.0, 1.0, 121.0, 4.0, 2.0736};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
