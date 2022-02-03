/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Quantize.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class QuantizeTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(QuantizeTest, FloatUint8)
{
  std::vector<float> input_data{-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};

  std::vector<float> ref_output_data{0, 1, 2, 3, 4, 251, 252, 253, 254, 255};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 5}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, /*scale*/ 0.5, /*zero_point*/ 127);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 5}));
}

TEST_F(QuantizeTest, FloatInt8)
{
  std::vector<float> input_data{-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};

  std::vector<float> ref_output_data{-128, -127, -126, -125, -124, 123, 124, 125, 126, 127};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 5}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, /*scale*/ 0.5, /*zero_point*/ -1);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 5}));
}

TEST_F(QuantizeTest, FloatInt16)
{
  std::vector<float> input_data{-63.5, -63, -3, -2, -1, 1, 2, 3, 63.5, 64};

  std::vector<float> ref_output_data{-12700, -12600, -600, -400, -200, 200, 400, 600, 12700, 12800};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 5}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, /*scale*/ 0.005, /*zero_point*/ 0);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int16_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 5}));
}

TEST_F(QuantizeTest, Int16Int16)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> ref_output_data{2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

  Tensor input_tensor = makeInputTensor<DataType::S16>(
    {1, 1, 2, 5}, /*scale*/ 1.0, /*zero_point*/ 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, /*scale*/ 0.5, /*zero_point*/ 0);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int16_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 2, 5}));
}

TEST_F(QuantizeTest, Int8Int8)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> ref_output_data{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

  Tensor input_tensor = makeInputTensor<DataType::S8>(
    {1, 1, 2, 5}, /*scale*/ 0.5, /*zero_point*/ -1, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, /*scale*/ 0.5, /*zero_point*/ -1);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 2, 5}));
}

TEST_F(QuantizeTest, Uint8Uint8)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> ref_output_data{129, 131, 133, 135, 137, 139, 141, 143, 145, 147};

  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 1, 2, 5}, /*scale*/ 0.5, /*zero_point*/ 127, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, /*scale*/ 0.5, /*zero_point*/ 127);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 2, 5}));
}

TEST_F(QuantizeTest, Int16Int8)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> ref_output_data{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

  Tensor input_tensor = makeInputTensor<DataType::S16>(
    {1, 1, 2, 5}, /*scale*/ 1.0, /*zero_point*/ 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, /*scale*/ 0.5, /*zero_point*/ -1);

  Quantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 2, 5}));
}

TEST_F(QuantizeTest, InvalidInputType_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>({1, 1, 2, 5}, 0.5, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, /*scale*/ 0.5, /*zero_point*/ -1);

  Quantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(QuantizeTest, InvalidOutputTypeForFloatInput_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 2, 5}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Quantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(QuantizeTest, InvalidOutputTypeForInt16Input_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 1, 2, 5}, 0.5, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Quantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(QuantizeTest, InvalidOutputTypeForInt8Input_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::S8>({1, 1, 2, 5}, 0.5, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Quantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(QuantizeTest, InvalidOutputTypeForUint8Input_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::U8>({1, 1, 2, 5}, 0.5, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  Quantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(QuantizeTest, InvalidInputZeroPoint_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 1, 2, 5}, 0.5, -1, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.5, 0);

  Quantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
