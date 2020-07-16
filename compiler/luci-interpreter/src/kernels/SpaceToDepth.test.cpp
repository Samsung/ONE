/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SpaceToDepth.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(SpaceToDepthTest, Float)
{
  std::vector<float> input_data{1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1};
  Shape input_shape{1, 2, 2, 2};
  Tensor input_tensor{DataType::FLOAT32, input_shape, {{}, {}}, ""};
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(float));
  std::vector<float> output_data{1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1};
  std::vector<int32_t> output_shape{1, 1, 1, 8};
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SpaceToDepthParams params{};
  params.block_size = 2;

  SpaceToDepth kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(SpaceToDepthTest, Uint8)
{
  std::pair<float, int32_t> quant_params = quantizationParams<uint8_t>(-12.7f, 12.8f);
  std::vector<uint8_t> input_data = quantize<uint8_t>({1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1},
                                                      quant_params.first, quant_params.second);
  Shape input_shape{1, 2, 2, 2};
  Tensor input_tensor{DataType::U8, input_shape, {{quant_params.first}, {quant_params.second}}, ""};
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(uint8_t));
  std::vector<float> output_data{1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1};
  std::vector<int32_t> output_shape{1, 1, 1, 8};
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_params.first, quant_params.second);

  SpaceToDepthParams params{};
  params.block_size = 2;

  SpaceToDepth kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantize(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                         output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
