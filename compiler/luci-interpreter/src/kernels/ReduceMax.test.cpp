/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/ReduceMax.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ReduceMaxTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(ReduceMaxTest, FloatNotKeepDims)
{
  std::vector<float> input_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                   9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  std::vector<int32_t> axis_data{1, 0, -3, -3};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({4, 3, 2}, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({4}, axis_data, _memory_manager.get());
  Tensor temp_index(DataType::S32, Shape({}), {}, "");
  Tensor resolved_axes(DataType::S32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ReducerParams params{};
  params.keep_dims = false;

  ReduceMax kernel(&input_tensor, &axis_tensor, &output_tensor, &temp_index, &resolved_axes,
                   params);
  kernel.configure();
  _memory_manager->allocate_memory(temp_index);
  _memory_manager->allocate_memory(resolved_axes);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{23, 24};
  std::initializer_list<int32_t> ref_output_shape{2};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(ReduceMaxTest, FloatKeepDims)
{
  std::vector<float> input_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                   9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  std::vector<int32_t> axis_data{0, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({4, 3, 2}, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({2}, axis_data, _memory_manager.get());
  Tensor temp_index(DataType::S32, Shape({}), {}, "");
  Tensor resolved_axes(DataType::S32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ReducerParams params{};
  params.keep_dims = true;

  ReduceMax kernel(&input_tensor, &axis_tensor, &output_tensor, &temp_index, &resolved_axes,
                   params);
  kernel.configure();
  _memory_manager->allocate_memory(temp_index);
  _memory_manager->allocate_memory(resolved_axes);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{20, 22, 24};
  std::initializer_list<int32_t> ref_output_shape{1, 3, 1};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(ReduceMaxTest, BoolNotKeepDims)
{
  std::vector<uint8_t> input_data = {true, true,  false, false, true, false, false, true,
                                     true, true,  false, false, true, true,  false, true,
                                     true, false, true,  false, true, false, false, true};

  std::vector<int32_t> axis_data{1, 0, -3, -3};
  Tensor input_tensor =
    makeInputTensor<DataType::BOOL>({4, 3, 2}, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({4}, axis_data, _memory_manager.get());
  Tensor temp_index(DataType::S32, Shape({}), {}, "");
  Tensor resolved_axes(DataType::S32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  ReducerParams params{};
  params.keep_dims = false;

  ReduceMax kernel(&input_tensor, &axis_tensor, &output_tensor, &temp_index, &resolved_axes,
                   params);
  kernel.configure();
  _memory_manager->allocate_memory(temp_index);
  _memory_manager->allocate_memory(resolved_axes);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<bool> ref_output_data{true, true};
  std::initializer_list<int32_t> ref_output_shape{2};
  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(ReduceMaxTest, BoolKeepDims)
{
  std::vector<uint8_t> input_data = {true, true,  false, false, true, false, false, true,
                                     true, true,  false, false, true, true,  false, true,
                                     true, false, true,  false, true, false, false, true};

  std::vector<int32_t> axis_data{0, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::BOOL>({4, 3, 2}, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({2}, axis_data, _memory_manager.get());
  Tensor temp_index(DataType::S32, Shape({}), {}, "");
  Tensor resolved_axes(DataType::S32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  ReducerParams params{};
  params.keep_dims = true;

  ReduceMax kernel(&input_tensor, &axis_tensor, &output_tensor, &temp_index, &resolved_axes,
                   params);
  kernel.configure();
  _memory_manager->allocate_memory(temp_index);
  _memory_manager->allocate_memory(resolved_axes);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<bool> ref_output_data{true, true, true};
  std::initializer_list<int32_t> ref_output_shape{1, 3, 1};
  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
