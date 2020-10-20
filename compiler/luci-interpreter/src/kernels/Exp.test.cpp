/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Exp.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(ExpTest, Float)
{
  Shape input_shape{1, 1, 7};
  std::vector<float> input_data{0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Exp kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  std::vector<int32_t> ref_output_shape{1, 1, 7};
  std::vector<float> ref_output_data{std::exp(0.0f),   std::exp(1.0f),    std::exp(-1.0f),
                                     std::exp(100.0f), std::exp(-100.0f), std::exp(0.01f),
                                     std::exp(-0.01f)};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
