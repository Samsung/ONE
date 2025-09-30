/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

TEST_F(GenModelTest, OneOp_DynamicUpdateSlice)
{
  CircleGen cgen;

  int operand = cgen.addTensor({{3, 3}, circle::TensorType::TensorType_FLOAT32});
  int update = cgen.addTensor({{2, 1}, circle::TensorType::TensorType_FLOAT32});

  std::vector<int32_t> indices_data{1, 1};
  uint32_t indices_buf = cgen.addBuffer(indices_data);
  int indices = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, indices_buf});

  int output = cgen.addTensor({{3, 3}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorDynamicUpdateSlice({{operand, update, indices}, {output}});
  cgen.setInputsAndOutputs({operand, update}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4, 5, 6, 7, 8, 9}, {-1, -2}}, {{1, 2, 3, 4, -1, 6, 7, -2, 9}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}
