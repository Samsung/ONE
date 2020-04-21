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

#include "kernels/Add.h"
#include "kernels/TestCommon.h"

namespace
{

using namespace luci_interpreter;
using namespace luci_interpreter::testing;

TEST(AddTest, Float)
{
  auto input1 = getTensorFrom<DataType::FLOAT32>({1, 2, 3, 4}, {2, 2});
  auto input2 = getTensorFrom<DataType::FLOAT32>({-1, -1, -1, -1}, {2, 2});
  auto output = getEmptyTensor(DataType::FLOAT32, {2, 2});

  AddParams params{};
  params.activation = Activation::NONE;

  kernels::Add kernel{input1.get(), input2.get(), output.get(), params};
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output.get()),
              ElementsAreArray(ArrayFloatNear({0, 1, 2, 3})));
}

TEST(AddTest, Quantized)
{
  AffineQuantization quant{{5.0f}, {0}};
  auto input1 = getTensorFrom<DataType::U8>({1, 2, 3, 4}, {2, 2}, quant);

  auto input2 = getTensorFrom<DataType::U8>({1, 1, 2, 5}, {2, 2}, quant);
  auto output = getEmptyTensor(DataType::U8, {2, 2}, quant);

  AddParams params{};
  params.activation = Activation::NONE;

  kernels::Add kernel{input1.get(), input2.get(), output.get(), params};
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractDequantizedData(output.get()),
              ElementsAreArray(ArrayFloatNear({10.0f, 15.0f, 25.0f, 45.0f})));
}

} // namespace
