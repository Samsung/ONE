/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Unpack.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(int axis, Shape input_shape, std::initializer_list<T> input_data,
           const std::vector<std::initializer_list<int32_t>> &exp_output_shape,
           std::vector<std::initializer_list<T>> exp_output_data)
{
  std::unique_ptr<MManager> memory_manager = std::make_unique<SimpleMManager>();
  constexpr DataType element_type = getElementType<T>();
  const int num_outputs = input_shape.dim(axis < 0 ? axis + input_shape.num_dims() : axis);

  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i)
  {
    output_tensors.push_back(makeOutputTensor(element_type));
  }

  std::vector<Tensor *> output_tensor_ptrs(num_outputs);
  for (int i = 0; i < num_outputs; ++i)
  {
    output_tensor_ptrs[i] = &output_tensors[i];
  }

  UnpackParams params{};
  params.axis = axis;

  Unpack kernel(&input_tensor, std::move(output_tensor_ptrs), params);
  kernel.configure();
  for (int i = 0; i < num_outputs; i++)
  {
    memory_manager->allocate_memory(&output_tensors[i]);
  }
  kernel.execute();

  for (int i = 0; i < num_outputs; ++i)
  {
    EXPECT_THAT(extractTensorData<T>(output_tensors[i]),
                ::testing::ElementsAreArray(exp_output_data[i]));
  }
}

template <typename T> class UnpackTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(UnpackTest, DataTypes);

TYPED_TEST(UnpackTest, ThreeOutputs)
{
  Check<TypeParam>(/*axis=*/0, /*input_shape=*/{3, 2},
                   /*input_data=*/{1, 2, 3, 4, 5, 6},
                   /*exp_output_shape=*/{{2}, {2}, {2}},
                   /*exp_output_data=*/{{1, 2}, {3, 4}, {5, 6}});
}

TYPED_TEST(UnpackTest, ThreeOutputsAxisOne)
{
  Check<TypeParam>(/*axis=*/1, /*input_shape=*/{3, 2},
                   /*input_data=*/{1, 2, 3, 4, 5, 6},
                   /*exp_output_shape=*/{{3}, {3}},
                   /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}});
}

TYPED_TEST(UnpackTest, ThreeOutputsNegativeAxisOne)
{
  Check<TypeParam>(/*axis=*/-1, /*input_shape=*/{3, 2},
                   /*input_data=*/{1, 2, 3, 4, 5, 6},
                   /*exp_output_shape=*/{{3}, {3}},
                   /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}});
}

TYPED_TEST(UnpackTest, ThreeOutputsNegativeAxisTwo)
{
  Check<TypeParam>(/*axis=*/-2, /*input_shape=*/{3, 2},
                   /*input_data=*/{1, 2, 3, 4, 5, 6},
                   /*exp_output_shape=*/{{2}, {2}, {2}},
                   /*exp_output_data=*/{{1, 2}, {3, 4}, {5, 6}});
}

TYPED_TEST(UnpackTest, OneOutput)
{
  Check<TypeParam>(/*axis=*/0, /*input_shape=*/{1, 6},
                   /*input_data=*/{1, 2, 3, 4, 5, 6},
                   /*exp_output_shape=*/{{6}},
                   /*exp_output_data=*/{{1, 2, 3, 4, 5, 6}});
}

TYPED_TEST(UnpackTest, ThreeDimensionsTwoOutputs)
{
  Check<TypeParam>(/*axis=*/2, /*input_shape=*/{2, 2, 2},
                   /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8},
                   /*exp_output_shape=*/{{2, 2}, {2, 2}},
                   /*exp_output_data=*/{{1, 3, 5, 7}, {2, 4, 6, 8}});
}

TYPED_TEST(UnpackTest, FiveDimensionsTwoOutputs)
{
  Check<TypeParam>(
    /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
    /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
    /*exp_output_data=*/
    {{1, 2, 5, 6, 9, 10, 13, 14}, {3, 4, 7, 8, 11, 12, 15, 16}});
}

TYPED_TEST(UnpackTest, VectorToScalar)
{
  Check<TypeParam>(/*axis=*/0, /*input_shape=*/{5},
                   /*input_data=*/{1, 2, 3, 4, 5},
                   /*exp_output_shape=*/{{}, {}, {}, {}, {}},
                   /*exp_output_data=*/{{1}, {2}, {3}, {4}, {5}});
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
