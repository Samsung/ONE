/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/UnidirectionalSequenceLSTM.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class UnidirectionalSequenceLSTMTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

// NOTE from NoCifgNoPeepholeNoProjectionNoClippingUnidirectionalLstmTest
TEST_F(UnidirectionalSequenceLSTMTest, FloatTest)
{
  // TODO implement
  SUCCEED();
}

TEST_F(UnidirectionalSequenceLSTMTest, Unsupported_Type_Configure_NEG)
{
  // TODO implement
  SUCCEED();
}

TEST_F(UnidirectionalSequenceLSTMTest, Invalid_Input_Shape_NEG)
{
  // TODO implement
  SUCCEED();
}

TEST_F(UnidirectionalSequenceLSTMTest, Invalid_Input_Shape_2_NEG)
{
  // TODO implement
  SUCCEED();
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
