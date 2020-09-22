/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GenModelTest.h"

#include <memory>

TEST_F(GenModelTest, OneOp_FullyConnected)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ 1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1,
                                  1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1,
                                  1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1,
                                  1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1 };
  std::vector<float> bias_data{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{16, 1}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
      uniformTCD<float>({{1, 3, 2, 1}}, {{2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 6}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}
