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

#include "kernels/PRelu.h"
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
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> alpha_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<T> input_data,
           std::initializer_list<T> alpha_data, std::initializer_list<T> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor alpha_tensor =
    makeInputTensor<element_type>(alpha_shape, alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);

  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(PReluTest, FloatSimple)
{
  Check<float>(/*input_shape=*/{2, 3}, /*alpha_shape=*/{2, 3},
               /*output_shape=*/{2, 3},
               /*input_data=*/
               {
                 0.0f, 1.0f, 3.0f,   // Row 1
                 1.0f, -1.0f, -2.0f, // Row 2
               },
               /*alpha_data=*/
               {
                 0.0f, 0.5f, 0.1f, // Row 1
                 0.0f, 0.5f, 0.1f, // Row 2
               },
               /*output_data=*/
               {
                 0.0f, 1.0f, 3.0f,   // Row 1
                 1.0f, -0.5f, -0.2f, // Row 2
               });

  SUCCEED();
}

TEST(PReluTest, FloatBroadcast)
{
  Check<float>(/*input_shape=*/{1, 2, 2, 3}, /*alpha_shape=*/{1, 1, 3},
               /*output_shape=*/{1, 2, 2, 3},
               /*input_data=*/
               {
                 0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                 1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                 -1.0f, -1.0f, -1.0f, // Row 2, Column 1
                 -2.0f, -2.0f, -2.0f, // Row 2, Column 2
               },
               /*alpha_data=*/
               {0.0f, 1.0f, 2.0f},
               /*output_data=*/
               {
                 0.0f, 0.0f, 0.0f,   // Row 1, Column 1
                 1.0f, 1.0f, 1.0f,   // Row 1, Column 2
                 0.0f, -1.0f, -2.0f, // Row 2, Column 1
                 0.0f, -2.0f, -4.0f, // Row 2, Column 2
               });

  SUCCEED();
}

float GetTolerance(float min, float max) { return (max - min) / 255.0; }

TEST(PReluTest, Uint8Simple)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> input_data{-0.8f, 0.2f, 0.9f, 0.7f, 0.1f, -0.4f};
  std::vector<float> alpha_data{0.5f, 0.5f, 0.5f, 0.25f, 1.0f, 0.25f};
  std::vector<float> ref_output_data{-0.4f, 0.2f, 0.9f, 0.7f, 0.1f, -0.1f};

  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-1.0f, 1.0f);

  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 3, 1}, quant_param.first, quant_param.second, input_data, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 3, 1}, quant_param.first, quant_param.second, alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, kQuantizedTolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 3, 1}));

  SUCCEED();
}

TEST(PReluTest, Uint8Broadcast)
{
  std::vector<float> input_data{
    0.0f,   0.0f,   0.0f,   // Row 1, Column 1
    0.5f,   0.5f,   0.5f,   // Row 1, Column 2
    -1.0f,  -1.0f,  -1.0f,  // Row 2, Column 1
    -0.25f, -0.25f, -0.25f, // Row 2, Column 2
  };
  std::vector<float> alpha_data{0.0f, 0.5f, -0.5f};
  std::vector<float> ref_output_data{
    0.0f, 0.0f,    0.0f,  // Row 1, Column 1
    0.5f, 0.5f,    0.5f,  // Row 1, Column 2
    0.0f, -0.5f,   0.5f,  // Row 2, Column 1
    0.0f, -0.125f, 0.125f // Row 2, Column 2
  };
  std::vector<float> ref_quant_output_data{
    128, 128, 128, // Row 1, Column 1
    192, 192, 192, // Row 1, Column 2
    128, 64,  192, // Row 2, Column 1
    128, 112, 144  // Row 2, Column 2
  };
  float kQuantizedTolerance = 2 * (1. / 256);
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(kMin, kMax);

  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 2, 3}, quant_param.first, quant_param.second, input_data, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::U8>(
    {1, 1, 3}, quant_param.first, quant_param.second, alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, kQuantizedTolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_quant_output_data));
}

TEST(PReluTest, SInt16_LWQ_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  // Rewrite this test in case layer-wise quantization for sint16 is supported
  std::vector<float> input_data(6); // data is not important
  std::vector<float> alpha_data(6);

  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 2, 3, 1}, 0.1, 0, input_data, memory_manager.get());
  Tensor alpha_tensor =
    makeInputTensor<DataType::S16>({1, 2, 3, 1}, 0.1, 0, alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.1, 0);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, SInt16_CWQ_Simple)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> input_data{-0.8f, 0.2f, 0.9f, -0.7f, 0.1f, -0.4f};
  std::vector<float> alpha_data{0.5f, 0.25f};
  std::vector<float> ref_output_data{-0.4f, 0.2f, 0.9f, -0.175f, 0.1f, -0.1f};

  std::vector<float> alpha_scales{0.05f, 0.025f};
  std::vector<int32_t> zerop{0, 0};
  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 1, 3, 2}, 0.1, 0, input_data, memory_manager.get());
  Tensor alpha_tensor =
    makeInputTensor<DataType::S16>({2}, alpha_scales, zerop, 0, alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.025, 0);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 3, 2}));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(PReluTest, SInt16_CWQ_spatial_alpha_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> input_data(6); // data is not important
  std::vector<float> alpha_data(6);

  std::vector<float> alpha_scales{0.25f, 0.05f};
  std::vector<int32_t> zerop{0, 0};
  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 1, 3, 2}, 0.1, 0, input_data, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::S16>({1, 1, 3, 2}, alpha_scales, zerop, 3,
                                                       alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.1, 0);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, SInt16_CWQ_wrong_dim_quant_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> input_data(6); // data is not important
  std::vector<float> alpha_data(6);

  std::vector<float> alpha_scales{0.25f};
  std::vector<int32_t> zerop{0};
  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 1, 3, 2}, 0.1, 0, input_data, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::S16>({1, 1, 1, 2}, alpha_scales, zerop, 1,
                                                       alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.1, 0);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, SInt16_CWQ_uneven_shape1)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> input_data{-0.8f, 0.2f, 0.9f, -0.7f, 0.1f, -0.4f};
  std::vector<float> alpha_data{0.5f, 0.25f};
  std::vector<float> ref_output_data{-0.4f, 0.2f, 0.9f, -0.175f, 0.1f, -0.1f};

  std::vector<float> alpha_scales{0.05f, 0.025f};
  std::vector<int32_t> zerop{0, 0};
  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 1, 3, 2}, 0.1, 0, input_data, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::S16>({1, 1, 2}, alpha_scales, zerop, 2,
                                                       alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.025, 0);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 3, 2}));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(PReluTest, SInt16_CWQ_uneven_shape2)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> input_data{
    0.0f,   0.0f,   0.0f,   // Row 1, Column 1
    0.5f,   0.5f,   0.5f,   // Row 1, Column 2
    -1.0f,  -1.0f,  -1.0f,  // Row 2, Column 1
    -0.25f, -0.25f, -0.25f, // Row 2, Column 2
  };
  std::vector<float> alpha_data{0.0f, 0.5f, -0.5f};
  std::vector<float> ref_output_data{
    0.0f, 0.0f,    0.0f,  // Row 1, Column 1
    0.5f, 0.5f,    0.5f,  // Row 1, Column 2
    0.0f, -0.5f,   0.5f,  // Row 2, Column 1
    0.0f, -0.125f, 0.125f // Row 2, Column 2
  };

  std::vector<float> alpha_scales{1.f, 0.05f, 0.1f};
  std::vector<int32_t> zerop{0, 0, 0};
  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 2, 2, 3}, 0.01, 0, input_data, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::S16>({1, 1, 1, 3}, alpha_scales, zerop, 3,
                                                       alpha_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.001, 0);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(PReluTest, Input_Output_Type_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, Input_Alpha_Type_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::U8>({1}, {1}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, Invalid_Input_Type_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1}, memory_manager.get());
  Tensor alpha_tensor = makeInputTensor<DataType::S64>({1}, {1}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  EXPECT_ANY_THROW(kernel.execute());
}

TEST(PReluTest, Input_Output_U8_CWQ_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> scales{1.f, 1.f};
  std::vector<int32_t> zerop{0, 0};
  std::vector<float> dummy_data(4, 0.f);
  Tensor input_tensor =
    makeInputTensor<DataType::U8>({2, 2}, scales, zerop, 0, dummy_data, memory_manager.get());
  Tensor alpha_tensor =
    makeInputTensor<DataType::U8>({2, 2}, scales, zerop, 0, dummy_data, memory_manager.get());
  Tensor output_tensor =
    makeInputTensor<DataType::U8>({2, 2}, scales, zerop, 0, dummy_data, memory_manager.get());

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, Input_Output_S16_CWQ_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> scales{1.f, 1.f};
  std::vector<int32_t> zerop{0, 0};
  std::vector<float> dummy_data(4, 0.f);
  Tensor input_tensor =
    makeInputTensor<DataType::S16>({2, 2}, scales, zerop, 0, dummy_data, memory_manager.get());
  Tensor alpha_tensor =
    makeInputTensor<DataType::S16>({2, 2}, scales, zerop, 0, dummy_data, memory_manager.get());
  Tensor output_tensor =
    makeInputTensor<DataType::S16>({2, 2}, scales, zerop, 0, dummy_data, memory_manager.get());

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PReluTest, Mixing_U8_S16_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();
  std::vector<float> dummy_data(4, 0.f);
  Tensor input_tensor =
    makeInputTensor<DataType::U8>({2, 2}, 1.f, 0, dummy_data, memory_manager.get());
  Tensor alpha_tensor =
    makeInputTensor<DataType::S16>({2, 2}, 1.f, 0, dummy_data, memory_manager.get());
  Tensor output_tensor =
    makeInputTensor<DataType::U8>({2, 2}, 1.f, 0, dummy_data, memory_manager.get());

  PRelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
