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

#include "kernels/Concatenation.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(ConcatenationTest, Float)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data);
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, input2_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  // Try different 'axis' and expect different results.
  {
    params.axis = 0;

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                ElementsAreArray(ArrayFloatNear({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})));
  }
  {
    params.axis = -2; // Same as '0'.

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                ElementsAreArray(ArrayFloatNear({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})));
  }
  {
    params.axis = 1;

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                ElementsAreArray(ArrayFloatNear({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12})));
  }
  {
    params.axis = -1; // Same as '1'.

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                ElementsAreArray(ArrayFloatNear({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12})));
  }
}

TEST(ConcatenationTest, Unsupported_Configure_Type_NEG)
{
  std::vector<int8_t> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<int8_t> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor = makeInputTensor<DataType::S8>({2, 3}, input1_data);
  Tensor input2_tensor = makeInputTensor<DataType::S8>({2, 3}, input2_data);
  Tensor output_tensor = makeOutputTensor(DataType::S8);
  ConcatenationParams params{};

  params.axis = -1;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ConcatenationTest, Invalid_Axis_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data);
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, input2_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = -3;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
