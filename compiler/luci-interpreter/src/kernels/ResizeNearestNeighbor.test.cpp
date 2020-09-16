/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved
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
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor size_tensor = makeInputTensor<DataType::S32>(size_shape, size_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeNearestNeighborParams params{};
  params.align_corners = align_corners;
  params.half_pixel_centers = half_pixel_centers;

  ResizeNearestNeighbor kernel(&input_tensor, &size_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<T>(output_tensor), FloatArrayNear(output_data));
}

template <>
void Check<uint8_t>(std::initializer_list<int32_t> input_shape,
                    std::initializer_list<int32_t> size_shape,
                    std::initializer_list<int32_t> output_shape,
                    std::initializer_list<float> input_data,
                    std::initializer_list<int32_t> size_data,
                    std::initializer_list<float> output_data, bool align_corners,
                    bool half_pixel_centers)
{
  std::pair<float, int32_t> quant_param =
      quantizationParams<uint8_t>(std::min(input_data) < 0 ? std::min(input_data) : 0.f,
                                  std::max(input_data) > 0 ? std::max(input_data) : 0.f);
  Tensor input_tensor =
      makeInputTensor<DataType::U8>(input_shape, quant_param.first, quant_param.second, input_data);
  Tensor size_tensor = makeInputTensor<DataType::S32>(size_shape, size_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.first);

  ResizeNearestNeighborParams params{};
  params.align_corners = align_corners;
  params.half_pixel_centers = half_pixel_centers;

  ResizeNearestNeighbor kernel(&input_tensor, &size_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, output_tensor.scale()));
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

TYPED_TEST(ResizeNearestNeighborTest, AlignCenterTest)
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
                       3, 6, 6,    //
                       9, 12, 12,  //
                       9, 12, 12,  //
                       4, 10, 10,  //
                       10, 16, 16, //
                       10, 16, 16, //
                   },
                   true, false);
}

TYPED_TEST(ResizeNearestNeighborTest, HalfPixelCenterTest)
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
                       3, 6, 6,    //
                       9, 12, 12,  //
                       9, 12, 12,  //
                       4, 10, 10,  //
                       10, 16, 16, //
                       10, 16, 16, //
                   },
                   false, true);
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
