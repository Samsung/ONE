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

TEST_F(GenModelTest, OneOp_ReverseV2_3D)
{
  CircleGen cgen;

  int in = cgen.addTensor({{4, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{4, 3, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorReverseV2({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "cpu"});
  _context->addTestCase(uniformTCD<float>(
    {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}},
    {{5, 6, 3, 4, 1, 2, 11, 12, 9, 10, 7, 8, 17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}}));

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_ReverseV2_1D)
{
  CircleGen cgen;

  int in = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{0};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorReverseV2({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "cpu"});
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{4, 3, 2, 1}}));

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ReverseV2_3D_DifferentType)
{
  CircleGen cgen;

  int in = cgen.addTensor({{4, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{4, 3, 2}, circle::TensorType::TensorType_INT32});

  cgen.addOperatorReverseV2({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "cpu"});
  _context->addTestCase(uniformTCD<int>(
    {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}},
    {{5, 6, 3, 4, 1, 2, 11, 12, 9, 10, 7, 8, 17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}}));
  _context->expectFailModelLoad();

  SUCCEED();
}
