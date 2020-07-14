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

#include "kernels/Squeeze.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(SqueezeTest, Float)
{
  std::vector<float> input_data{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  Shape input_shape = {1, 24, 1};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SqueezeParams params{};
  params.squeeze_dims_count = 2;
  params.squeeze_dims[0] = -1;
  params.squeeze_dims[1] = 0;

  Squeeze kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                               9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                               17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0})));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({24}));
}

TEST(SqueezeTest, Uint8)
{
  // There is No Uint8 case on TFlite Test, So I just add Quantization Parameters.
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-127.f, 128.f);
  std::vector<uint8_t> quant_input_data =
      quantize<uint8_t>({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                         13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
                        quant_param.first, quant_param.second);
  Tensor input_tensor{DataType::U8, {1, 24, 1}, {{quant_param.first}, {quant_param.second}}, ""};
  input_tensor.writeData(quant_input_data.data(), quant_input_data.size() * sizeof(uint8_t));
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  SqueezeParams params{};
  params.squeeze_dims_count = 2;
  params.squeeze_dims[0] = -1;
  params.squeeze_dims[1] = 0;

  Squeeze kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantize(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                         output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                               9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                               17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0})));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({24}));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
