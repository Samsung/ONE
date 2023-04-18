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

#include "kernels/Select.h"
#include "kernels/TestUtils.h"

#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class SelectTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

std::vector<unsigned char> c_data{
  1, 1, 1, // Row 1
  0, 0, 0, // Row 2
};

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

TEST_F(SelectTest, FloatSimple)
{
  Tensor c_tensor = makeInputTensor<DataType::BOOL>({2, 3}, c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, t_data, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Select kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 3}));
}

TEST_F(SelectTest, Invalid_C_Type_NEG)
{
  std::vector<float> i_c_data{
    1, 1, 1, // Row 1
    0, 0, 0, // Row 2
  };

  Tensor c_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, i_c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, t_data, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Select kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(SelectTest, Invalid_O_Type_NEG)
{
  Tensor c_tensor = makeInputTensor<DataType::BOOL>({2, 3}, c_data, _memory_manager.get());
  Tensor t_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, t_data, _memory_manager.get());
  Tensor e_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, e_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Select kernel(&c_tensor, &t_tensor, &e_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
