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

#include "kernels/FullyConnected.h"
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
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> weights_shape,
           std::initializer_list<int32_t> bias_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> weights_data,
           std::initializer_list<float> bias_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor weights_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_shape, weights_data, memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<T>(output_tensor), FloatArrayNear(output_data));
}

template <>
void Check<int8_t>(std::initializer_list<int32_t> input_shape,
                   std::initializer_list<int32_t> weights_shape,
                   std::initializer_list<int32_t> bias_shape,
                   std::initializer_list<int32_t> output_shape,
                   std::initializer_list<float> input_data,
                   std::initializer_list<float> weights_data,
                   std::initializer_list<float> bias_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  const float quantized_tolerance = getTolerance(-127, 128, 255);
  std::pair<float, int32_t> input_quant_param = quantizationParams<int8_t>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<int8_t>(-127, 128);
  Tensor input_tensor =
    makeInputTensor<DataType::S8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, memory_manager.get());
  Tensor weights_tensor =
    makeInputTensor<DataType::S8>(weights_shape, input_quant_param.first, input_quant_param.second,
                                  weights_data, memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::S32>(bias_shape, input_quant_param.first * input_quant_param.first, 0,
                                   bias_data, memory_manager.get());
  Tensor output_tensor =
    makeOutputTensor(DataType::S8, output_quant_param.first, output_quant_param.second);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, quantized_tolerance));
}

template <>
void Check<uint8_t>(
  std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> weights_shape,
  std::initializer_list<int32_t> bias_shape, std::initializer_list<int32_t> output_shape,
  std::initializer_list<float> input_data, std::initializer_list<float> weights_data,
  std::initializer_list<float> bias_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  const float quantized_tolerance = getTolerance(-127, 128, 255);
  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<uint8_t>(-127, 128);
  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, memory_manager.get());
  Tensor weights_tensor =
    makeInputTensor<DataType::U8>(weights_shape, input_quant_param.first, input_quant_param.second,
                                  weights_data, memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::S32>(bias_shape, input_quant_param.first * input_quant_param.first, 0,
                                   bias_data, memory_manager.get());
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, quantized_tolerance));
}

template <typename T> class FullyConnectedTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t, int8_t>;
TYPED_TEST_SUITE(FullyConnectedTest, DataTypes);

TYPED_TEST(FullyConnectedTest, Simple)
{
  Check<TypeParam>({3, 2, 2, 1}, {3, 6}, {3}, {2, 3},
                   {
                     -3, -5, 5, 4, 9, -2,  // batch = 0
                     -3, -2, -4, 9, -8, 1, // batch = 1
                   },
                   {
                     -3, -7, 4, -4, -6, 4, // unit = 0
                     3, 5, 2, 3, -3, -8,   // unit = 1
                     -3, 7, 4, 9, 0, -5,   // unit = 2
                   },
                   {-1, -5, -8},
                   {
                     0, 0, 32,   // batch = 0
                     22, 11, 47, // batch = 1
                   });
}

TEST(FullyConnectedTest, SimpleS4)
{
  std::initializer_list<int32_t> input_shape{1, 2};
  std::initializer_list<int32_t> weights_shape{4, 2};
  std::initializer_list<int32_t> bias_shape{4};
  std::initializer_list<int32_t> output_shape{1, 4};
  std::initializer_list<float> input_data{
    1, 3, // batch = 0
  };
  std::initializer_list<int8_t> weights_initializer{
    0,  1,  // unit = 0
    0,  0,  // unit = 1
    -1, -1, // unit = 2
    0,  0,  // unit = 3
  };
  std::initializer_list<float> bias_data{0, 1, 2, 3};
  std::initializer_list<float> output_data{
    1.5, 1, 0, 3, // batch = 0
  };
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  std::vector<int8_t> quantized_data(weights_initializer);
  Tensor weights_tensor(DataType::S4, weights_shape, {{0.5}, {0}}, "");
  memory_manager->allocate_memory(weights_tensor);
  weights_tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(int8_t));
  Tensor weights_scratch(DataType::FLOAT32, weights_shape, {}, "");
  memory_manager->allocate_memory(weights_scratch);

  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  const float quantized_tolerance = getTolerance(-8, 7, 15);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor,
                        &weights_scratch, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear(output_data, quantized_tolerance));
}

TEST(FullyConnectedTest, SimpleU4PerTensor)
{
  std::initializer_list<int32_t> input_shape{1, 2};
  std::initializer_list<int32_t> weights_shape{4, 2};
  std::initializer_list<int32_t> bias_shape{4};
  std::initializer_list<int32_t> output_shape{1, 4};
  std::initializer_list<float> input_data{
    1, 3, // batch = 0
  };
  std::initializer_list<uint8_t> weights_initializer{
    8, 9, // unit = 0
    8, 8, // unit = 1
    7, 7, // unit = 2
    8, 8, // unit = 3
  };
  std::initializer_list<float> bias_data{0, 1, 2, 3};
  std::initializer_list<float> output_data{
    1.5, 1, 0, 3, // batch = 0
  };
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  std::vector<uint8_t> quantized_data(weights_initializer);
  Tensor weights_tensor(DataType::U4, weights_shape, {{0.5}, {8}}, "");
  memory_manager->allocate_memory(weights_tensor);
  weights_tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(uint8_t));
  Tensor weights_scratch(DataType::FLOAT32, weights_shape, {}, "");
  memory_manager->allocate_memory(weights_scratch);

  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  const float quantized_tolerance = getTolerance(0, 15, 15);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor,
                        &weights_scratch, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear(output_data, quantized_tolerance));
}

TEST(FullyConnectedTest, SimpleU4PerChannel)
{
  std::initializer_list<int32_t> input_shape{1, 2};
  std::initializer_list<int32_t> weights_shape{4, 2};
  std::initializer_list<int32_t> bias_shape{4};
  std::initializer_list<int32_t> output_shape{1, 4};
  std::initializer_list<float> input_data{
    1, 3, // batch = 0
  };
  std::initializer_list<uint8_t> weights_initializer{
    8, 9, // unit = 0
    8, 8, // unit = 1
    7, 7, // unit = 2
    8, 8, // unit = 3
  };
  std::initializer_list<float> bias_data{0, 1, 2, 3};
  std::initializer_list<float> output_data{
    1.5, 1, 0, 3, // batch = 0
  };
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  std::vector<uint8_t> quantized_data(weights_initializer);
  Tensor weights_tensor(DataType::U4, weights_shape, {{0.5, 0.5, 0.5, 0.5}, {8, 8, 8, 8}, 0}, "");
  memory_manager->allocate_memory(weights_tensor);
  weights_tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(uint8_t));
  Tensor weights_scratch(DataType::FLOAT32, weights_shape, {}, "");
  memory_manager->allocate_memory(weights_scratch);

  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  const float quantized_tolerance = getTolerance(0, 15, 15);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor,
                        &weights_scratch, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear(output_data, quantized_tolerance));
}

TEST(FullyConnectedTest, SimpleU4WrongBiasType_NEG)
{
  std::initializer_list<int32_t> input_shape{1, 2};
  std::initializer_list<int32_t> weights_shape{4, 2};
  std::initializer_list<int32_t> bias_shape{4};
  std::initializer_list<int32_t> output_shape{1, 4};
  std::initializer_list<float> input_data{
    1, 3, // batch = 0
  };
  std::initializer_list<uint8_t> weights_initializer{
    8, 9, // unit = 0
    8, 8, // unit = 1
    7, 7, // unit = 2
    8, 8, // unit = 3
  };
  std::initializer_list<uint8_t> bias_data{0, 1, 2, 3};
  std::initializer_list<float> output_data{
    1.5, 1, 0, 3, // batch = 0
  };
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  std::vector<uint8_t> quantized_data(weights_initializer);
  Tensor weights_tensor(DataType::U4, weights_shape, {{0.5, 0.5}, {8, 8}, 1}, "");
  memory_manager->allocate_memory(weights_tensor);
  weights_tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(uint8_t));
  Tensor weights_scratch(DataType::FLOAT32, weights_shape, {}, "");
  memory_manager->allocate_memory(weights_scratch);

  Tensor bias_tensor = makeInputTensor<DataType::U8>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  const float quantized_tolerance = getTolerance(0, 15, 15);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor,
                        &weights_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(FullyConnectedTest, SimpleU4WrongInputType_NEG)
{
  std::initializer_list<int32_t> input_shape{1, 2};
  std::initializer_list<int32_t> weights_shape{4, 2};
  std::initializer_list<int32_t> bias_shape{4};
  std::initializer_list<int32_t> output_shape{1, 4};
  std::initializer_list<uint8_t> input_data{
    1, 3, // batch = 0
  };
  std::initializer_list<uint8_t> weights_initializer{
    8, 9, // unit = 0
    8, 8, // unit = 1
    7, 7, // unit = 2
    8, 8, // unit = 3
  };
  std::initializer_list<float> bias_data{0, 1, 2, 3};
  std::initializer_list<float> output_data{
    1.5, 1, 0, 3, // batch = 0
  };
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, input_data, memory_manager.get());
  std::vector<uint8_t> quantized_data(weights_initializer);
  Tensor weights_tensor(DataType::U4, weights_shape, {{0.5, 0.5}, {8, 8}, 1}, "");
  memory_manager->allocate_memory(weights_tensor);
  weights_tensor.writeData(quantized_data.data(), quantized_data.size() * sizeof(uint8_t));
  Tensor weights_scratch(DataType::FLOAT32, weights_shape, {}, "");
  memory_manager->allocate_memory(weights_scratch);

  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  const float quantized_tolerance = getTolerance(0, 15, 15);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor,
                        &weights_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(FullyConnectedTest, InvalidBiasType_NEG)
{
  Shape input_shape{3, 2, 2, 1};
  std::vector<float> input_data{
    -3, -5, 5,  4, 9,  -2, // batch = 0
    -3, -2, -4, 9, -8, 1,  // batch = 1
  };
  Shape weights_shape{3, 6};
  std::vector<float> weights_data{
    -3, -7, 4, -4, -6, 4,  // unit = 0
    3,  5,  2, 3,  -3, -8, // unit = 1
    -3, 7,  4, 9,  0,  -5, // unit = 2
  };
  Shape bias_shape{3};
  std::vector<int32_t> bias_data{-1, -5, -8};

  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor weights_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_shape, weights_data, memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(FullyConnectedTest, InvalidWeightShapeDim_NEG)
{
  Shape input_shape{3, 2, 2, 1};
  std::vector<float> input_data{
    -3, -5, 5,  4, 9,  -2, // batch = 0
    -3, -2, -4, 9, -8, 1,  // batch = 1
  };
  Shape weights_shape{1, 3, 6};
  std::vector<float> weights_data{
    -3, -7, 4, -4, -6, 4,  // unit = 0
    3,  5,  2, 3,  -3, -8, // unit = 1
    -3, 7,  4, 9,  0,  -5, // unit = 2
  };
  Shape bias_shape{3};
  std::vector<float> bias_data{-1, -5, -8};

  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor weights_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_shape, weights_data, memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(FullyConnectedTest, BiasElementNumWeightDimMismatch_NEG)
{
  Shape input_shape{3, 2, 2, 1};
  std::vector<float> input_data{
    -3, -5, 5,  4, 9,  -2, // batch = 0
    -3, -2, -4, 9, -8, 1,  // batch = 1
  };
  Shape weights_shape{6, 3};
  std::vector<float> weights_data{
    -3, -7, 4,  // unit = 0
    -4, -6, 4,  // unit = 1
    3,  5,  2,  // unit = 2
    3,  -3, -8, // unit = 3
    -3, 7,  4,  // unit = 4
    9,  0,  -5, // unit = 5
  };
  Shape bias_shape{3};
  std::vector<float> bias_data{-1, -5, -8};

  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor weights_tensor =
    makeInputTensor<DataType::FLOAT32>(weights_shape, weights_data, memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
