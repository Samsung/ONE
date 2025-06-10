/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/RoPE.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class RoPETest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(RoPETest, floatTest)
{
  Shape input_shape{1, 1, 1, 4};
  std::vector<float> input_data{0, 1.0, 2.0, 3.0};

  Shape sin_shape{1, 1, 1, 4};
  std::vector<float> sin_data{0.5, 1.0, 1.0, 0.5};

  Shape cos_shape{1, 1, 1, 4};
  std::vector<float> cos_data{1.0, 0.5, 0.5, 1.0};

  Shape ref_output_shape{1, 1, 1, 4};
  std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor sin_table = makeInputTensor<DataType::FLOAT32>(sin_shape, sin_data, _memory_manager.get());
  Tensor cos_table = makeInputTensor<DataType::FLOAT32>(cos_shape, cos_data, _memory_manager.get());

  RoPEParams params{};
  params.mode = RoPEMode::GPT_NEOX;

  RoPE kernel(&input_tensor, &sin_table, &cos_table, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 1, 4}));
}

TEST_F(RoPETest, Unsupported_dims_NEG)
{
  Shape input_shape{1, 1, 3};
  std::vector<float> input_data{0, 1.0, 2.0};

  Shape sin_shape{1, 1, 3};
  std::vector<float> sin_data{0.5, 1.0, 1.0};

  Shape cos_shape{1, 1, 3};
  std::vector<float> cos_data{1.0, 0.5, 0.5};

  Shape ref_output_shape{1, 1, 3};
  std::vector<float> ref_output_data{-1.0, -2.5, 1.0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor sin_table = makeInputTensor<DataType::FLOAT32>(sin_shape, sin_data, _memory_manager.get());
  Tensor cos_table = makeInputTensor<DataType::FLOAT32>(cos_shape, cos_data, _memory_manager.get());

  RoPEParams params{};
  params.mode = RoPEMode::GPT_NEOX;

  RoPE kernel(&input_tensor, &sin_table, &cos_table, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(RoPETest, Unsupported_mode_NEG)
{
  Shape input_shape{1, 1, 1, 4};
  std::vector<float> input_data{0, 1.0, 2.0, 3.0};

  Shape sin_shape{1, 1, 1, 4};
  std::vector<float> sin_data{0.5, 1.0, 1.0, 0.5};

  Shape cos_shape{1, 1, 1, 4};
  std::vector<float> cos_data{1.0, 0.5, 0.5, 1.0};

  Shape ref_output_shape{1, 1, 1, 4};
  std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor sin_table = makeInputTensor<DataType::FLOAT32>(sin_shape, sin_data, _memory_manager.get());
  Tensor cos_table = makeInputTensor<DataType::FLOAT32>(cos_shape, cos_data, _memory_manager.get());

  RoPEParams params{};
  params.mode = RoPEMode::GPT_J;

  RoPE kernel(&input_tensor, &sin_table, &cos_table, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(RoPETest, Invalid_input_sin_table_NEG)
{
  Shape input_shape{1, 1, 1, 4};
  std::vector<float> input_data{0, 1.0, 2.0, 3.0};

  Shape sin_shape{1, 1, 1, 3};
  std::vector<float> sin_data{0.5, 1.0, 1.0};

  Shape cos_shape{1, 1, 1, 4};
  std::vector<float> cos_data{1.0, 0.5, 0.5, 1.0};

  Shape ref_output_shape{1, 1, 1, 4};
  std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor sin_table = makeInputTensor<DataType::FLOAT32>(sin_shape, sin_data, _memory_manager.get());
  Tensor cos_table = makeInputTensor<DataType::FLOAT32>(cos_shape, cos_data, _memory_manager.get());

  RoPEParams params{};
  params.mode = RoPEMode::GPT_NEOX;

  RoPE kernel(&input_tensor, &sin_table, &cos_table, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(RoPETest, Invalid_input_cos_table_NEG)
{
  Shape input_shape{1, 1, 1, 4};
  std::vector<float> input_data{0, 1.0, 2.0, 3.0};

  Shape sin_shape{1, 1, 1, 4};
  std::vector<float> sin_data{0.5, 1.0, 1.0, 0.5};

  Shape cos_shape{1, 1, 1, 3};
  std::vector<float> cos_data{1.0, 0.5, 0.5};

  Shape ref_output_shape{1, 1, 1, 4};
  std::vector<float> ref_output_data{-1.0, -2.5, 1.0, 3.5};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor sin_table = makeInputTensor<DataType::FLOAT32>(sin_shape, sin_data, _memory_manager.get());
  Tensor cos_table = makeInputTensor<DataType::FLOAT32>(cos_shape, cos_data, _memory_manager.get());

  RoPEParams params{};
  params.mode = RoPEMode::GPT_NEOX;

  RoPE kernel(&input_tensor, &sin_table, &cos_table, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
