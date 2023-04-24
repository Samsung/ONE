/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/StridedSlice.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(StridedSliceTest, Float)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Shape input_shape{2, 3, 2};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape begin_shape{3};
  std::vector<int32_t> begin_data{0, 0, 0};
  Shape end_shape{3};
  std::vector<int32_t> end_data{1, 3, 2};
  Shape strides_shape{3};
  std::vector<int32_t> strides_data{1, 1, 1};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor begin_tensor =
    makeInputTensor<DataType::S32>(begin_shape, begin_data, memory_manager.get());
  Tensor end_tensor = makeInputTensor<DataType::S32>(end_shape, end_data, memory_manager.get());
  Tensor strides_tensor =
    makeInputTensor<DataType::S32>(strides_shape, strides_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  StridedSliceParams params{};
  params.begin_mask = 0;
  params.end_mask = 0;
  params.ellipsis_mask = 0;
  params.new_axis_mask = 0;
  params.shrink_axis_mask = 1;

  StridedSlice kernel(&input_tensor, &begin_tensor, &end_tensor, &strides_tensor, &output_tensor,
                      params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<int32_t> output_shape{3, 2};
  std::vector<float> output_data{1, 2, 3, 4, 5, 6};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(StridedSliceTest, Uint8)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Shape input_shape{2, 3, 2};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape begin_shape{3};
  std::vector<int32_t> begin_data{0, 0, 0};
  Shape end_shape{3};
  std::vector<int32_t> end_data{1, 3, 2};
  Shape strides_shape{3};
  std::vector<int32_t> strides_data{1, 1, 1};
  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, 1.0f, 0, input_data, memory_manager.get());
  Tensor begin_tensor =
    makeInputTensor<DataType::S32>(begin_shape, begin_data, memory_manager.get());
  Tensor end_tensor = makeInputTensor<DataType::S32>(end_shape, end_data, memory_manager.get());
  Tensor strides_tensor =
    makeInputTensor<DataType::S32>(strides_shape, strides_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1.0f, 0);

  StridedSliceParams params{};
  params.begin_mask = 0;
  params.end_mask = 0;
  params.ellipsis_mask = 0;
  params.new_axis_mask = 0;
  params.shrink_axis_mask = 1;

  StridedSlice kernel(&input_tensor, &begin_tensor, &end_tensor, &strides_tensor, &output_tensor,
                      params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<int32_t> output_shape{3, 2};
  std::vector<float> output_data{1, 2, 3, 4, 5, 6};
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
