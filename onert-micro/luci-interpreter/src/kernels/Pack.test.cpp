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

#include "kernels/Pack.h"
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
void Check(std::vector<std::initializer_list<int32_t>> input_shapes,
           std::initializer_list<int32_t> output_shape, std::vector<std::vector<T>> input_datas,
           std::initializer_list<T> output_data, int32_t axis)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  std::vector<const Tensor *> inputs(input_datas.size());
  std::vector<Tensor> tmp_inputs;
  for (int i = 0; i < input_datas.size(); i++)
  {
    if (std::is_same<T, float>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, int64_t>::value)
    {
      tmp_inputs.push_back(Tensor(element_type, input_shapes[i], {}, ""));
      memory_manager->allocate_memory(tmp_inputs[i]);
      tmp_inputs[i].writeData(input_datas[i].data(), input_datas[i].size() * sizeof(T));
    }
    else if (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value)
    {
      tmp_inputs.push_back(Tensor(element_type, input_shapes[i], {{1.0f / 255}, {128}}, ""));
      memory_manager->allocate_memory(tmp_inputs[i]);
      tmp_inputs[i].writeData(input_datas[i].data(), input_datas[i].size() * sizeof(T));
    }
    else
    {
      assert((std::is_same<T, int16_t>::value) && "unexpected dtype is tested");
      tmp_inputs.push_back(Tensor(element_type, input_shapes[i], {{1.0f}, {0}}, ""));
      memory_manager->allocate_memory(tmp_inputs[i]);
      tmp_inputs[i].writeData(input_datas[i].data(), input_datas[i].size() * sizeof(T));
    }
  }
  for (int i = 0; i < input_datas.size(); i++)
  {
    inputs[i] = &tmp_inputs[i];
  }

  Tensor output_tensor = makeOutputTensor(element_type);
  if (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value)
  {
    output_tensor = makeOutputTensor(element_type, 1.0f / 255, 128);
  }
  else if (std::is_same<T, int16_t>::value)
  {
    output_tensor = makeOutputTensor(element_type, 1.0f, 0);
  }

  PackParams params{};
  params.axis = axis;
  params.values_count = input_datas.size();
  Pack kernel(inputs, &output_tensor, params);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <typename T> class PackTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<uint8_t, int8_t, int16_t, int32_t, int64_t, float>;
TYPED_TEST_SUITE(PackTest, DataTypes);

TYPED_TEST(PackTest, ThreeInputs)
{
  Check<TypeParam>(/*input_shapes=*/{{2}, {2}, {2}},
                   /*output_shape=*/{3, 2},
                   /*input_datas=*/
                   {{1, 4}, {2, 5}, {3, 6}},
                   /*output_data=*/
                   {1, 4, 2, 5, 3, 6}, /*axis=*/0);

  SUCCEED();
}

TYPED_TEST(PackTest, NegAxis)
{
  Check<TypeParam>(/*input_shapes=*/{{2}, {2}, {2}},
                   /*output_shape=*/{2, 3},
                   /*input_datas=*/
                   {{1, 4}, {2, 5}, {3, 6}},
                   /*output_data=*/
                   {1, 2, 3, 4, 5, 6}, /*axis=*/-1);

  SUCCEED();
}

TEST(Pack, MismatchingInputValuesCount_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  std::vector<float> input1_data{1, 4};
  std::vector<float> input2_data{2, 5};
  std::vector<float> input3_data{3, 6};
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({2}, input1_data, memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({2}, input2_data, memory_manager.get());
  Tensor input3_tensor = makeInputTensor<DataType::FLOAT32>({2}, input3_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  PackParams params{};
  {
    params.axis = 0;
    params.values_count = 2;

    Pack kernel({&input1_tensor, &input2_tensor, &input3_tensor}, &output_tensor, params);
    EXPECT_ANY_THROW(kernel.configure());
  }
}

TEST(Pack, InvalidInputAxis_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  std::vector<float> input1_data{1, 4};
  std::vector<float> input2_data{2, 5};
  std::vector<float> input3_data{3, 6};
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({2}, input1_data, memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({2}, input2_data, memory_manager.get());
  Tensor input3_tensor = makeInputTensor<DataType::FLOAT32>({2}, input3_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  PackParams params{};
  {
    params.axis = 2;
    params.values_count = 3;

    Pack kernel({&input1_tensor, &input2_tensor, &input3_tensor}, &output_tensor, params);
    EXPECT_ANY_THROW(kernel.configure());
  }
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
