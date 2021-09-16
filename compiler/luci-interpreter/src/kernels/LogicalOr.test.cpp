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

#include "kernels/LogicalOr.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class LogicalOrTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(LogicalOrTest, Basic)
{
  Tensor input1_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, false, true},
                                                         _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, true, false},
                                                         _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor),
              ::testing::ElementsAre(true, false, true, true));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAre(1, 1, 1, 4));
}

TEST_F(LogicalOrTest, Broadcast)
{
  Tensor input1_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, false, true},
                                                         _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::BOOL>({1, 1, 1, 1}, {false}, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor),
              ::testing::ElementsAre(true, false, false, true));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAre(1, 1, 1, 4));
}

TEST_F(LogicalOrTest, MismatchInputType_NEG)
{
  Tensor input1_tensor =
    makeInputTensor<DataType::S32>({1, 1, 1, 4}, {1, 0, 0, 1}, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::BOOL>({1, 1, 1, 1}, {false}, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::S32);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(LogicalOrTest, InputTypeInvalid_NEG)
{
  Tensor input1_tensor =
    makeInputTensor<DataType::S32>({1, 1, 1, 4}, {1, 0, 0, 1}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S32>({1, 1, 1, 1}, {0}, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
