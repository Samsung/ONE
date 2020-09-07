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

#include "kernels/ResizeNearestNeighbor.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> size_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<float> input_data,
           std::initializer_list<int32_t> size_data, std::initializer_list<float> output_data,
           bool align_corners, bool half_pixel_centers)
{
  std::pair<float, int32_t> quant_param =
      quantizationParams<T>(std::min(input_data) < 0 ? std::min(input_data) : 0.f,
                            std::max(input_data) > 0 ? std::max(input_data) : 0.f);
  Tensor input_tensor{
      getElementType<T>(), input_shape, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor size_tensor{DataType::S32, size_shape, {}, ""};

  if (std::is_floating_point<T>::value)
  {
    input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));
  }
  else
  {
    std::vector<T> quantized_input_value =
        quantize<T>(input_data, quant_param.first, quant_param.second);
    input_tensor.writeData(quantized_input_value.data(), quantized_input_value.size() * sizeof(T));
  }
  size_tensor.writeData(size_data.begin(), size_data.size() * sizeof(int32_t));

  Tensor output_tensor =
      makeOutputTensor(getElementType<T>(), quant_param.first, quant_param.first);

  ResizeNearestNeighborParams params{};
  params.align_corners = align_corners;
  params.half_pixel_centers = half_pixel_centers;

  ResizeNearestNeighbor kernel(&input_tensor, &size_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  if (std::is_floating_point<T>::value)
  {
    EXPECT_THAT(extractTensorData<T>(output_tensor), ElementsAreArray(ArrayFloatNear(output_data)));
  }
  else
  {
    EXPECT_THAT(dequantize<T>(extractTensorData<T>(output_tensor), output_tensor.scale(),
                              output_tensor.zero_point()),
                ElementsAreArray(ArrayFloatNear(output_data, output_tensor.scale())));
  }
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <typename T> class ResizeNearestNeighborTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(ResizeNearestNeighborTest, DataTypes);

TYPED_TEST(ResizeNearestNeighborTest, SimpleTest)
{
  Check<TypeParam>({2, 2, 2, 1}, {2}, {2, 3, 3, 1},
                   {
                       3, 6,  //
                       9, 12, //
                       4, 10, //
                       10, 16 //
                   },
                   {3, 3},
                   {
                       3, 3, 6,    //
                       3, 3, 6,    //
                       9, 9, 12,   //
                       4, 4, 10,   //
                       4, 4, 10,   //
                       10, 10, 16, //
                   },
                   false, false);
}

TEST(ResizeNearestNeighborTest, InputShapeInvalid_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2}, {
                                                                          3, 6,  //
                                                                          9, 12, //
                                                                          4, 10, //
                                                                          10, 16 //
                                                                      });
  Tensor size_tensor = makeInputTensor<DataType::S32>({2}, {3, 3});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeNearestNeighborParams params{};
  params.align_corners = false;
  params.half_pixel_centers = false;

  ResizeNearestNeighbor kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ResizeNearestNeighborTest, SizeShapeInvalid_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2, 1}, {
                                                                             3, 6,  //
                                                                             9, 12, //
                                                                             4, 10, //
                                                                             10, 16 //
                                                                         });
  Tensor size_tensor = makeInputTensor<DataType::S32>({2, 1}, {3, 3});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeNearestNeighborParams params{};
  params.align_corners = false;
  params.half_pixel_centers = false;

  ResizeNearestNeighbor kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ResizeNearestNeighborTest, SizeDimInvalid_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2, 1}, {
                                                                             3, 6,  //
                                                                             9, 12, //
                                                                             4, 10, //
                                                                             10, 16 //
                                                                         });
  Tensor size_tensor = makeInputTensor<DataType::S32>({3}, {3, 3, 1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeNearestNeighborParams params{};
  params.align_corners = false;
  params.half_pixel_centers = false;

  ResizeNearestNeighbor kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
