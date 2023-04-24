/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Cast.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T1, typename T2>
void Check(std::initializer_list<int32_t> shape, std::initializer_list<T1> input_data,
           std::initializer_list<T2> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType input_type = getElementType<T1>();
  constexpr DataType output_type = getElementType<T2>();

  Tensor input_tensor = makeInputTensor<input_type>(shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(output_type);

  Cast kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T2>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), shape);
}

template <typename T>
void CheckBoolTo(std::initializer_list<int32_t> shape, std::initializer_list<bool> input_data,
                 std::initializer_list<T> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType input_type = loco::DataType::BOOL;
  constexpr DataType output_type = getElementType<T>();
  std::vector<typename DataTypeImpl<input_type>::Type> input_data_converted;
  for (auto elem : input_data)
  {
    input_data_converted.push_back(elem);
  }

  Tensor input_tensor =
    makeInputTensor<input_type>(shape, input_data_converted, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(output_type);

  Cast kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), shape);
}

template <typename T> class CastTest : public ::testing::Test
{
};

using IntDataTypes =
  ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(CastTest, IntDataTypes);

TYPED_TEST(CastTest, FloatToInt)
{
  Check<float, TypeParam>(/*shape=*/{1, 1, 1, 4},
                          /*input_data=*/
                          {
                            1.0f, 9.0f, 7.0f, 3.0f, //
                          },
                          /*output_data=*/
                          {
                            1, 9, 7, 3, //
                          });
  SUCCEED();
}

TYPED_TEST(CastTest, IntToFloat)
{
  Check<TypeParam, float>(/*shape=*/{1, 1, 1, 4},
                          /*input_data=*/
                          {
                            1, 9, 7, 3, //
                          },
                          /*output_data=*/
                          {
                            1.0f, 9.0f, 7.0f, 3.0f, //
                          });
  SUCCEED();
}

template <typename T1, typename T2> void check_int()
{
  Check<T1, T2>(/*shape=*/{1, 1, 1, 4},
                /*input_data=*/
                {
                  1, 9, 7, 3, //
                },
                /*output_data=*/
                {
                  1, 9, 7, 3, //
                });
  SUCCEED();
}

TYPED_TEST(CastTest, IntToInt)
{
  check_int<TypeParam, uint8_t>();
  check_int<TypeParam, uint16_t>();
  check_int<TypeParam, uint32_t>();
  check_int<TypeParam, uint64_t>();
  check_int<TypeParam, int8_t>();
  check_int<TypeParam, int16_t>();
  check_int<TypeParam, int32_t>();
  check_int<TypeParam, int64_t>();
  SUCCEED();
}

TYPED_TEST(CastTest, IntToBool)
{
  Check<TypeParam, bool>(/*shape=*/{1, 1, 1, 4},
                         /*input_data=*/
                         {
                           1, 0, 7, 0, //
                         },
                         /*output_data=*/
                         {
                           true, false, true, false, //
                         });
  SUCCEED();
}

TYPED_TEST(CastTest, BoolToInt)
{
  CheckBoolTo<TypeParam>(/*shape=*/{1, 1, 1, 4},
                         /*input_data=*/
                         {
                           true, false, false, true, //
                         },
                         /*output_data=*/
                         {
                           1, 0, 0, 1, //
                         });
  SUCCEED();
}

TEST(CastTest, FloatToBool)
{
  Check<float, bool>(/*shape=*/{1, 1, 1, 4},
                     /*input_data=*/
                     {
                       1.0f, 0.0f, 7.0f, 0.0f, //
                     },
                     /*output_data=*/
                     {
                       true, false, true, false, //
                     });
  SUCCEED();
}

TEST(CastTest, BoolToFloat)
{
  CheckBoolTo<float>(/*shape=*/{1, 1, 1, 4},
                     /*input_data=*/
                     {
                       true, false, false, true, //
                     },
                     /*output_data=*/
                     {
                       1.0f, 0.0f, 0.0f, 1.0f, //
                     });
  SUCCEED();
}

TEST(CastTest, FloatToFloat)
{
  Check<float, float>(/*shape=*/{1, 1, 1, 4},
                      /*input_data=*/
                      {
                        1.0f, 0.0f, 7.0f, 0.0f, //
                      },
                      /*output_data=*/
                      {
                        1.0f, 0.0f, 7.0f, 0.0f, //
                      });
  SUCCEED();
}

TEST(CastTest, BoolToBool)
{
  CheckBoolTo<bool>(/*shape=*/{1, 1, 1, 4},
                    /*input_data=*/
                    {
                      true, true, false, false, //
                    },
                    /*output_data=*/
                    {
                      true, true, false, false, //
                    });
  SUCCEED();
}

TEST(CastTest, UnsupportedType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 1, 2, 4},
                                                           {
                                                             1, 2, 7, 8, //
                                                             1, 9, 7, 3, //
                                                           },
                                                           memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::Unknown);

  Cast kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
  SUCCEED();
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
