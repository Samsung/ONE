/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/CumSum.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{
using namespace testing;

class CumSumTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(CumSumTest, Float)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> axis_data{2};
  Shape input_shape{1, 1, 2, 4};
  Shape axis_shape(0);
  std::vector<float> output_data{1, 2, 3, 4, 6, 8, 10, 12};
  std::vector<int32_t> output_shape{1, 1, 2, 4};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Tensor axis_tensor = makeInputTensor<DataType::S32>(axis_shape, axis_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  CumSumParams params{false, false};

  CumSum kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST_F(CumSumTest, Float_Reverse)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> axis_data{2};
  Shape input_shape{1, 1, 2, 4};
  Shape axis_shape(0);
  std::vector<float> output_data{6, 8, 10, 12, 5, 6, 7, 8};
  std::vector<int32_t> output_shape{1, 1, 2, 4};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Tensor axis_tensor = makeInputTensor<DataType::S32>(axis_shape, axis_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  CumSumParams params{false, true};

  CumSum kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST_F(CumSumTest, Float_Exclusive)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> axis_data{2};
  Shape input_shape{1, 1, 4, 2};
  Shape axis_shape(0);
  std::vector<float> output_data{0, 0, 1, 2, 4, 6, 9, 12};
  std::vector<int32_t> output_shape{1, 1, 4, 2};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Tensor axis_tensor = makeInputTensor<DataType::S32>(axis_shape, axis_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  CumSumParams params{true, false};

  CumSum kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST_F(CumSumTest, InputShapeInvalid_NEG)
{
  std::vector<float> input_data{1};
  std::vector<int32_t> axis_data{2};
  Shape input_shape(0);
  Shape axis_shape(0);

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Tensor axis_tensor = makeInputTensor<DataType::S32>(axis_shape, axis_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  CumSumParams params{false, false};

  CumSum kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(CumSumTest, AxisShapeInvalid_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> axis_data{2};
  Shape input_shape{1, 1, 2, 4};
  Shape axis_shape{1};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Tensor axis_tensor = makeInputTensor<DataType::S32>(axis_shape, axis_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  CumSumParams params{false, false};

  CumSum kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(CumSumTest, AxisTypeInvalid_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> axis_data{2};
  Shape input_shape{1, 1, 2, 4};
  Shape axis_shape(0);

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Tensor axis_tensor =
    makeInputTensor<DataType::FLOAT32>(axis_shape, axis_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  CumSumParams params{false, false};

  CumSum kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace

} // namespace kernels
} // namespace luci_interpreter
