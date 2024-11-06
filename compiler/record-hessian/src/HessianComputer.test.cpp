/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "record-hessian/HessianComputer.h"

#include <luci/IR/CircleNode.h>
#include <luci_interpreter/Interpreter.h>

#include <gtest/gtest.h>

#include <vector>

using namespace record_hessian;

TEST(HessianComputerTest, recordHessianValidInput)
{
  luci::CircleFullyConnected node;

  std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0};

  luci_interpreter::DataType data_type = luci_interpreter::DataType::FLOAT32;
  luci_interpreter::Shape shape({1, 4});

  std::string tensor_name = "input_tensor";

  luci_interpreter::Tensor input_tensor(data_type, shape, luci_interpreter::AffineQuantization{},
                                        tensor_name);

  size_t data_size = input_data.size() * sizeof(float);
  std::vector<uint8_t> buffer(data_size);

  input_tensor.set_data_buffer(buffer.data());
  input_tensor.writeData(input_data.data(), data_size);

  HessianComputer computer;

  EXPECT_NO_THROW(computer.recordHessian(&node, &input_tensor));
}

TEST(HessianComputerTest, recordHessian_wrong_op_NEG)
{
  luci::CircleAdd node;

  std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0};

  luci_interpreter::DataType data_type = luci_interpreter::DataType::FLOAT32;
  luci_interpreter::Shape shape({1, 2, 2, 1});

  std::string tensor_name = "input_tensor";

  luci_interpreter::Tensor input_tensor(data_type, shape, luci_interpreter::AffineQuantization{},
                                        tensor_name);

  size_t data_size = input_data.size() * sizeof(float);
  std::vector<uint8_t> buffer(data_size);

  input_tensor.set_data_buffer(buffer.data());
  input_tensor.writeData(input_data.data(), data_size);

  HessianComputer computer;

  EXPECT_ANY_THROW(computer.recordHessian(&node, &input_tensor));
}

TEST(HessianComputerTest, recordHessianNullTensor_NEG)
{
  luci::CircleAdd node;
  HessianComputer computer;
  EXPECT_ANY_THROW(computer.recordHessian(&node, nullptr));
}

TEST(HessianComputerTest, unfoldValidInput)
{
  std::vector<float> buf = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  uint32_t input_n = 1, input_h = 2, input_w = 2, input_c = 2;
  uint32_t stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1;
  uint32_t kernel_h = 2, kernel_w = 2, kernel_ic = 2;

  unfold(buf, input_n, input_h, input_w, input_c, stride_h, stride_w, dilation_h, dilation_w,
         kernel_h, kernel_w, kernel_ic);
  std::vector<float> expected_output = {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0};

  EXPECT_EQ(buf, expected_output);
}

TEST(HessianComputerTest, unfoldInvalidInput_NEG)
{
  std::vector<float> buf = {1.0, 2.0, 3.0, 4.0};
  uint32_t input_n = 1, input_h = 2, input_w = 2, input_c = 1;
  uint32_t stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1;
  uint32_t kernel_h = 2, kernel_w = 2, kernel_ic = 2;

  EXPECT_ANY_THROW(unfold(buf, input_n, input_h, input_w, input_c, stride_h, stride_w, dilation_h,
                          dilation_w, kernel_h, kernel_w, kernel_ic));
}
