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

TEST_F(GenModelTest, OneOp_Split)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 4}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});

  int out1 = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out2 = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSplit({{axis, in}, {out1, out2}}, 2);
  cgen.setInputsAndOutputs({in}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6}, {3, 4, 7, 8}}));
  _context->setBackends({"cpu", "acl_cl", "acl_neon"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_SplitNonConstAxis)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 4}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});

  int out1 = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out2 = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSplit({{axis, in}, {out1, out2}}, 2);
  cgen.setInputsAndOutputs({axis, in}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<int32_t>({1})
                          .addInput<float>({1, 2, 3, 4, 5, 6, 7, 8})
                          .addOutput<float>({1, 2, 5, 6})
                          .addOutput<float>({3, 4, 7, 8}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_SplitNegatveSplitNum)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 4}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});

  int out1 = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out2 = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSplit({{axis, in}, {out1, out2}}, -3);
  cgen.setInputsAndOutputs({in}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6}, {3, 4, 7, 8}}));
  _context->setBackends({"cpu", "acl_cl", "acl_neon"});
  _context->expectFailModelLoad();

  SUCCEED();
}
