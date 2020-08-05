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
#include "kernels/L2Normalize.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(L2NormalizeTest, Float)
{
  std::vector<float> input_data = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  L2NormParams params{};
  params.activation = Activation::NONE;

  L2Normalize kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{-0.55, 0.3, 0.35, 0.6, -0.35, 0.05};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

// TODO Uint8Quantized
// Implement GetDequantizedOutput Function.
// Create Test for Uint8 Case

} // namespace
} // namespace kernels
} // namespace luci_interpreter
