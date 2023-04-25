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

#include "kernels/Transpose.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> perm_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<T> input_data,
           std::initializer_list<int32_t> perm_data, std::initializer_list<T> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor perm_tensor = makeInputTensor<DataType::S32>(perm_shape, perm_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  Transpose kernel(&input_tensor, &perm_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
}

template <typename T> class TransposeTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(TransposeTest, DataTypes);

TYPED_TEST(TransposeTest, Small3D)
{
  Check<TypeParam>(/*input_shape=*/{2, 3, 4}, /*perm_shape=*/{3}, /*output_shape=*/{4, 2, 3},
                   /*input_data=*/{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                   /*perm_data=*/{2, 0, 1},
                   /*output_data=*/{0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                    2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});
}

TYPED_TEST(TransposeTest, Large4D)
{
  Check<TypeParam>(
    /*input_shape=*/{2, 3, 4, 5}, /*perm_shape=*/{4}, /*output_shape=*/{4, 2, 3, 5},
    /*input_data=*/{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
                    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                    30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
                    45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                    60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
                    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119},
    /*perm_data=*/{2, 0, 1, 3},
    /*output_data=*/{0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
                     60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
                     5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
                     65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
                     10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
                     70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
                     15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
                     75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119});
}

TYPED_TEST(TransposeTest, Large2D)
{
  Check<TypeParam>(
    /*input_shape=*/{10, 12}, /*perm_shape=*/{2}, /*output_shape=*/{12, 10},
    /*input_data=*/{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
                    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                    30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
                    45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                    60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
                    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119},
    /*perm_data=*/{1, 0},
    /*output_data=*/{0,  12, 24, 36,  48,  60, 72, 84, 96,  108, 1,  13, 25, 37,  49,
                     61, 73, 85, 97,  109, 2,  14, 26, 38,  50,  62, 74, 86, 98,  110,
                     3,  15, 27, 39,  51,  63, 75, 87, 99,  111, 4,  16, 28, 40,  52,
                     64, 76, 88, 100, 112, 5,  17, 29, 41,  53,  65, 77, 89, 101, 113,
                     6,  18, 30, 42,  54,  66, 78, 90, 102, 114, 7,  19, 31, 43,  55,
                     67, 79, 91, 103, 115, 8,  20, 32, 44,  56,  68, 80, 92, 104, 116,
                     9,  21, 33, 45,  57,  69, 81, 93, 105, 117, 10, 22, 34, 46,  58,
                     70, 82, 94, 106, 118, 11, 23, 35, 47,  59,  71, 83, 95, 107, 119});
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
