/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SelectV2.h"
#include "kernels/TestUtils.h"

#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class SelectV2Test : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

std::vector<unsigned char> c_data_single{0};

std::vector<unsigned char> c_data{
  1, 1, 1, // Row 1
  0, 0, 0, // Row 2
};

std::vector<float> t_data_single{-0.5};

std::vector<float> t_data{
  0.5, 0.7, 0.9, // Row 1
  1,   0,   -1,  // Row 2
};

std::vector<float> e_data{
  0.9, 0.7, 0.5, // Row 1
  -1,  0,   1,   // Row 2
};

std::vector<float> ref_output_data{
  0.5, 0.7, 0.9, // Row 1
  -1,  0,   1,   // Row 2
};

std::vector<float> ref_broadcast_output_data{
  -0.5, -0.5, -0.5, // Row 1
  0.9,  0.7,  0.5,  // Row 1
  -0.5, -0.5, -0.5, // Row 3
  -1,   0,    1,    // Row 4
};

TEST_F(SelectV2Test, FloatSimple)
{
  Tensor c_tensor = makeInputTensor<DataType::BOOL>({2, 3}, c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, t_data, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SelectV2 kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 3}));
}

TEST_F(SelectV2Test, FloatBroadcast4D)
{
  Tensor c_tensor = makeInputTensor<DataType::BOOL>({1, 2, 3, 1}, c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({1}, t_data_single, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 1, 3, 1}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SelectV2 kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_broadcast_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 2, 3, 1}));
}

TEST_F(SelectV2Test, Invalid_C_Type_NEG)
{
  std::vector<float> i_c_data{
    1, 1, 1, // Row 1
    0, 0, 0, // Row 2
  };

  Tensor c_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, i_c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, t_data, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SelectV2 kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(SelectV2Test, Invalid_O_Type_NEG)
{
  Tensor c_tensor = makeInputTensor<DataType::BOOL>({2, 3}, c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, t_data, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  SelectV2 kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
