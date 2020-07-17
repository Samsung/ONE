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

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data,
           DataType element_type, int8_t squeeze_dims_count, std::vector<int32_t> squeeze_dims)
{
  std::pair<float, int32_t> quant_param = quantizationParams<T>(-127.f, 128.f);
  Tensor input_tensor{element_type, input_shape, {{quant_param.first}, {quant_param.second}}, ""};
  std::vector<T> quant_input_data = quantize<T>(input_data, quant_param.first, quant_param.second);
  input_tensor.writeData(quant_input_data.data(), quant_input_data.size() * sizeof(T));

  Tensor output_tensor = makeOutputTensor(element_type, quant_param.first, quant_param.second);

  SqueezeParams params{};
  params.squeeze_dims_count = squeeze_dims_count;
  for (size_t i = 0; i < squeeze_dims.size(); i++)
  {
    params.squeeze_dims[i] = squeeze_dims.at(i);
  }

  Squeeze kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantize(extractTensorData<T>(output_tensor), output_tensor.scale(),
                         output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <typename T> class SqueezeTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(SqueezeTest, DataTypes);

TYPED_TEST(SqueezeTest, TotalTest)
{
  Check<TypeParam>(
      /*input_shape=*/{1, 24, 1}, /*output_shape=*/{24},
      /*input_data=*/{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
      /*output_data=*/{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
      getElementType<TypeParam>(), 2, {-1, 0});
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
