/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/LogicalAnd.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(LogicalAndTest, Basic)
{
  Shape input_shape{1, 1, 1, 4};
  Tensor input_tensor1 = makeInputTensor<DataType::BOOL>(input_shape, {true, false, false, true});
  Tensor input_tensor2 = makeInputTensor<DataType::BOOL>(input_shape, {true, false, true, false});
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalAnd kernel(&input_tensor1, &input_tensor2, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAre(true, false, false, false));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAre(1, 1, 1, 4));
}

TEST(LogicalAndTest, Broadcast)
{
  Tensor input_tensor1 = makeInputTensor<DataType::BOOL>({1, 1, 1, 4}, {true, false, false, true});
  Tensor input_tensor2 = makeInputTensor<DataType::BOOL>({1, 1, 1, 1}, {true});
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  LogicalAnd kernel(&input_tensor1, &input_tensor2, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAre(true, false, false, true));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAre(1, 1, 1, 4));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
