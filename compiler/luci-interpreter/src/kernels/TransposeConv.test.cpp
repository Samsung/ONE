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

#include "kernels/TransposeConv.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T, typename B>
void Check(std::initializer_list<int32_t> output_shape_shape,
           std::initializer_list<int32_t> weight_shape,
           std::initializer_list<int32_t> input_data_shape,
           std::initializer_list<int32_t> bias_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<int32_t> output_shape_data, std::initializer_list<T> weight_data,
           std::initializer_list<T> input_data_data, std::initializer_list<B> bias_data,
           std::initializer_list<T> output_data, luci::Padding padding, int32_t stride_height,
           int32_t stride_width, DataType element_type)
{
  Tensor output_shape_tensor{element_type, output_shape_shape, {}, ""};
  output_shape_tensor.writeData(output_shape_data.begin(), output_shape_data.size() * sizeof(T));
  Tensor weight_tensor{element_type, weight_shape, {}, ""};
  weight_tensor.writeData(weight_data.begin(), weight_data.size() * sizeof(T));
  Tensor input_data_tensor{element_type, input_data_shape, {}, ""};
  input_data_tensor.writeData(input_data_data.begin(), input_data_data.size() * sizeof(T));

  Tensor output_tensor = makeOutputTensor(element_type);

  TransposeConvParams params{};
  params.padding = padding;
  params.stride_height = stride_height;
  params.stride_width = stride_width;

  if (bias_data.size() != 0)
  {
    Tensor bias_tensor = makeInputTensor<getElementType<B>()>(bias_shape, bias_data);
    TransposeConv kernel(&output_shape_tensor, &weight_tensor, &input_data_tensor, &bias_tensor,
                         &output_tensor, params);
    kernel.configure();
    kernel.execute();
  }
  else
  {
    TransposeConv kernel(&output_shape_tensor, &weight_tensor, &input_data_tensor, nullptr,
                         &output_tensor, params);
    kernel.configure();
    kernel.execute();
  }
  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
}

TEST(TransposeConvTest, FloatSimple)
{
  Check<float, float>(
      /*outputShape_shape=*/{4}, /*weight_shape=*/{1, 3, 3, 1}, /*input_shape=*/{1, 4, 4, 1},
      /*bias_shape=*/{}, /*output_shape=*/{1, 4, 4, 1}, /*outputShape_data=*/{1, 4, 4, 1},
      /*weight_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      /*bias_data=*/{},
      /*output_data=*/{29, 62, 83, 75, 99, 192, 237, 198, 207, 372, 417, 330, 263, 446, 485, 365},
      /*params.padding=*/luci::Padding::SAME, /*stride_height=*/1, /*stride_width=*/1,
      getElementType<float>());

  SUCCEED();
}

TEST(TransposeConvTest, FloatTwoFiltersTest)
{
  Check<float, float>(
      /*outputShape_shape=*/{4}, /*weight_shape=*/{1, 3, 3, 2}, /*input_shape=*/{1, 4, 4, 2},
      /*bias_shape=*/{}, /*output_shape=*/{1, 4, 4, 1}, /*outputShape_data=*/{1, 4, 4, 1},
      /*weight_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
      /*input_data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
      /*bias_data=*/{},
      /*output_data=*/{184, 412, 568, 528, 678, 1347, 1689, 1434, 1494, 2715, 3057, 2442, 1968,
                       3352, 3652, 2760},
      /*params.padding=*/luci::Padding::SAME, /*stride_height=*/1, /*stride_width=*/1,
      getElementType<float>());

  SUCCEED();
}

TEST(TransposeConvTest, SimpleBiasTest)
{
  Check<float, float>(
      /*outputShape_shape=*/{4}, /*weight_shape=*/{2, 3, 3, 1},
      /*input_shape=*/{1, 2, 2, 1},
      /*bias_shape=*/{2}, /*output_shape=*/{1, 4, 4, 1}, /*outputShape_data=*/{1, 5, 5, 2},
      /*weight_data=*/{1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
      /*input_data=*/{1, 2, 3, 4},
      /*bias_data=*/{3, 4},
      /*output_data=*/{4,  6,  6,  8,  10, 14, 9,  12, 13, 16, 10,  12,  12, 14, 28, 32, 21,
                       24, 25, 28, 19, 24, 27, 32, 65, 76, 45, 52,  57,  64, 24, 28, 30, 34,
                       64, 72, 39, 44, 47, 52, 42, 46, 48, 52, 106, 114, 63, 68, 71, 76},
      /*params.padding=*/luci::Padding::VALID, /*stride_height=*/2, /*stride_width=*/2,
      getElementType<float>());

  SUCCEED();
}

// TODO Uint8Simple
// Implement GetDequantizedOutput Function.
// Create Test for Uint8 Case

// TODO Uint8FiltersTest
// Implement GetDequantizedOutput Function.
// Create Test for Uint8 Case

} // namespace
} // namespace kernels
} // namespace luci_interpreter
