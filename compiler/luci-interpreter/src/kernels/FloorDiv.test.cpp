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

#include "kernels/FloorDiv.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(FloorDivTest, FloatSimple)
{
  std::initializer_list<int32_t> x_shape{2, 3};
  std::vector<float> x_data{
      0.5, 2.4,  3.1,  // Row 1
      1.0, -1.9, -2.8, // Row 2
  };

  std::initializer_list<int32_t> y_shape = x_shape;
  std::vector<float> y_data{
      2.0, 0.5,  3.0,  // Row 1
      1.0, -1.0, -2.0, // Row 2
  };

  std::initializer_list<int32_t> ref_output_shape = x_shape;
  std::vector<float> ref_output_data{
      0, 4, 1, // Row 1
      1, 1, 1, // Row 2
  };

  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>(x_shape, x_data);
  Tensor y_tensor = makeInputTensor<DataType::FLOAT32>(y_shape, y_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorDiv kernel(&x_tensor, &y_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(FloorDivTest, FloatBroadcast)
{
  std::initializer_list<int32_t> x_shape{1, 3};
  std::vector<float> x_data{
      0.5, 2.4, -3.1, // Row 1
  };

  std::initializer_list<int32_t> y_shape{3, 3};
  std::vector<float> y_data{
      1.0, 1.0,  1.0,  // Row 1
      2.0, -0.5, -2.0, // Row 2
      0.3, 0.7,  0.9,  // Row 3
  };

  std::initializer_list<int32_t> ref_output_shape{3, 3};
  std::vector<float> ref_output_data{
      0, 2,  -4, // Row 1
      0, -5, 1,  // Row 2
      1, 3,  -4, // Row 3
  };

  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>(x_shape, x_data);
  Tensor y_tensor = makeInputTensor<DataType::FLOAT32>(y_shape, y_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorDiv kernel(&x_tensor, &y_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(FloorDivTest, Input_Output_Type_Mismatch_NEG)
{
  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor y_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  FloorDiv kernel(&x_tensor, &y_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(FloorDivTest, Input_Type_Mismatch_NEG)
{
  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1});
  Tensor y_tensor = makeInputTensor<DataType::U8>({1}, {1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorDiv kernel(&x_tensor, &y_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
