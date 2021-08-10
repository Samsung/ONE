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

#include "kernels/ResizeBilinear.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

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
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>(size_shape, size_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeBilinearParams params{};
  params.align_corners = align_corners;
  params.half_pixel_centers = half_pixel_centers;

  ResizeBilinear kernel(&input_tensor, &size_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
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
  // On TFlite example use Uint8 value it self, so this means quant param scale 1.0f and zero
  // point 0.
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, 1.0, 0, input_data, memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>(size_shape, size_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1.0, 0);

  ResizeBilinearParams params{};
  params.align_corners = align_corners;
  params.half_pixel_centers = half_pixel_centers;

  ResizeBilinear kernel(&input_tensor, &size_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, output_tensor.scale()));
}

template <typename T> class ResizeBilinearTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(ResizeBilinearTest, DataTypes);

TYPED_TEST(ResizeBilinearTest, SimpleTest)
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
                     3, 5, 6,    //
                     7, 9, 10,   //
                     9, 11, 12,  //
                     4, 8, 10,   //
                     8, 12, 14,  //
                     10, 14, 16, //
                   },
                   false, false);
  SUCCEED();
}

TEST(ResizeBilinearTest, HalfPixelCenterFloatTest)
{
  Check<float>({2, 2, 2, 1}, {2}, {2, 3, 3, 1},
               {
                 1, 2, //
                 3, 4, //
                 1, 2, //
                 3, 4  //
               },
               {3, 3},
               {
                 1, 1.5, 2, //
                 2, 2.5, 3, //
                 3, 3.5, 4, //
                 1, 1.5, 2, //
                 2, 2.5, 3, //
                 3, 3.5, 4, //
               },
               false, true);
  SUCCEED();
}

TEST(ResizeBilinearTest, HalfPixelCenterUint8Test)
{
  Check<uint8_t>({2, 2, 2, 1}, {2}, {2, 3, 3, 1},
                 {
                   3, 6,  //
                   9, 12, //
                   4, 10, //
                   12, 16 //
                 },
                 {3, 3},
                 {
                   2, 4, 6,    //
                   6, 7, 9,    //
                   9, 10, 12,  //
                   4, 7, 10,   //
                   8, 10, 13,  //
                   12, 14, 16, //
                 },
                 false, true);
  SUCCEED();
}

TEST(ResizeBilinearTest, InputShapeInvalid_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2},
                                                           {
                                                             3, 6,  //
                                                             9, 12, //
                                                             4, 10, //
                                                             10, 16 //
                                                           },
                                                           memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>({2}, {3, 3}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeBilinearParams params{};
  params.align_corners = false;
  params.half_pixel_centers = false;

  ResizeBilinear kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ResizeBilinearTest, SizeShapeInvalid_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2, 1},
                                                           {
                                                             3, 6,  //
                                                             9, 12, //
                                                             4, 10, //
                                                             10, 16 //
                                                           },
                                                           memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>({2, 1}, {3, 3}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeBilinearParams params{};
  params.align_corners = false;
  params.half_pixel_centers = false;

  ResizeBilinear kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ResizeBilinearTest, SizeDimInvalid_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2, 1},
                                                           {
                                                             3, 6,  //
                                                             9, 12, //
                                                             4, 10, //
                                                             10, 16 //
                                                           },
                                                           memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>({3}, {3, 3, 1}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeBilinearParams params{};
  params.align_corners = false;
  params.half_pixel_centers = false;

  ResizeBilinear kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ResizeBilinearTest, InvalidParams_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 2, 1},
                                                           {
                                                             3, 6,  //
                                                             9, 12, //
                                                             4, 10, //
                                                             10, 16 //
                                                           },
                                                           memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>({2}, {3, 3}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ResizeBilinearParams params{};
  params.align_corners = true;
  params.half_pixel_centers = true;

  ResizeBilinear kernel(&input_tensor, &size_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
