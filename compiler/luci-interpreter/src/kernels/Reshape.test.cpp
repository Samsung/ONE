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

#include "kernels/Reshape.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ReshapeTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

// TODO Test types other than FLOAT32.

TEST_F(ReshapeTest, Regular)
{
  Shape input_shape{1, 2, 2, 3};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape shape_shape{2};
  std::vector<int32_t> shape_data{3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor shape_tensor =
    makeInputTensor<DataType::S32>(shape_shape, shape_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Reshape kernel(&input_tensor, &shape_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(input_data));
}

TEST_F(ReshapeTest, UnknownDimension)
{
  Shape input_shape{2, 1, 2, 3};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape shape_shape{3};
  std::vector<int32_t> shape_data{2, -1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor shape_tensor =
    makeInputTensor<DataType::S32>(shape_shape, shape_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Reshape kernel(&input_tensor, &shape_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(input_data));
}

TEST_F(ReshapeTest, SupportS64)
{
  Shape input_shape{2, 1, 2, 3};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape shape_shape{3};
  std::vector<int64_t> shape_data{2, -1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor shape_tensor =
    makeInputTensor<DataType::S64>(shape_shape, shape_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Reshape kernel(&input_tensor, &shape_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(input_data));
}

TEST_F(ReshapeTest, SupportS16_NEG)
{
  Shape input_shape{2, 1, 2, 3};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape shape_shape{3};
  std::vector<int16_t> shape_data{2, -1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor shape_tensor =
    makeInputTensor<DataType::S16>(shape_shape, shape_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Reshape kernel(&input_tensor, &shape_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
