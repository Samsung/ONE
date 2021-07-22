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

#include "kernels/LOGICAL_OR.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(LogicalOrTest, Basic)
{
  Tensor input1_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, false, true});
  Tensor input2_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, true, false});

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor),
              ::testing::ElementsAre(true, false, true, true));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAre(1, 1, 1, 4));
}

TEST(LogicalOrTest, Broadcast)
{
  Tensor input1_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, false, true});
  Tensor input2_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 1}, {false});

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor),
              ::testing::ElementsAre(true, false, false, true));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAre(1, 1, 1, 4));
}

TEST(LogicalOrTest, MismatchInputType_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::S32>({1, 1, 1, 4}, {1, 0, 0, 1});
  Tensor input2_tensor = makeInputTensor<DataType::BOOL>({1, 1, 1, 1}, {false});

  Tensor output_tensor = makeOutputTensor(DataType::S32);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(LogicalOrTest, InputTypeInvalid_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::S32>({1, 1, 1, 4}, {1, 0, 0, 1});
  Tensor input2_tensor = makeInputTensor<DataType::S32>({1, 1, 1, 1}, {0});

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalOr kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
