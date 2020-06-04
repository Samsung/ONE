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

#include "kernels/Transpose.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(TransposeTest, SmallFloat)
{
  Shape input_shape{1, 2, 3, 1};
  std::vector<float> input_data{1, 2, 3, 4, 5, 6};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Shape perm_shape{4};
  std::vector<int32_t> perm_data{2, 1, 3, 0};
  Tensor perm_tensor = makeInputTensor<DataType::S32>(perm_shape, perm_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Transpose kernel(&input_tensor, &perm_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{1, 4, 2, 5, 3, 6};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

TEST(TransposeTest, LargeFloat)
{
  Shape input_shape{2, 3, 4, 5};
  std::vector<float> input_data{
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  //
      12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  //
      24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  //
      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
      48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  //
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  //
      72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  //
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  //
      96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, //
      108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119  //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Shape perm_shape{4};
  std::vector<int32_t> perm_data{2, 0, 1, 3};
  Tensor perm_tensor = makeInputTensor<DataType::S32>(perm_shape, perm_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Transpose kernel(&input_tensor, &perm_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,  //
      60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104, //
      5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,  //
      65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109, //
      10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,  //
      70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114, //
      15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,  //
      75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119  //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

TEST(TransposeTest, LargeFloatCheck)
{
  Shape input_shape{1, 1, 1, 120};
  std::vector<float> input_data{
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  //
      12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  //
      24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  //
      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
      48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  //
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  //
      72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  //
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  //
      96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, //
      108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119  //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Shape perm_shape{4};
  std::vector<int32_t> perm_data{0, 1, 2, 3};
  Tensor perm_tensor = makeInputTensor<DataType::S32>(perm_shape, perm_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Transpose kernel(&input_tensor, &perm_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  //
      12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  //
      24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  //
      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
      48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  //
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  //
      72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  //
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  //
      96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, //
      108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119  //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
