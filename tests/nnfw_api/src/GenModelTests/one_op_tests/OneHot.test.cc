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

TEST_F(GenModelTest, OneOp_OneHot_OffValueToConst)
{
  CircleGen cgen;
  std::vector<int32_t> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  std::vector<float> off_value_data{0};
  uint32_t off_value_buf = cgen.addBuffer(off_value_data);
  int indices = cgen.addTensor({{1, 2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int on_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int off_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, off_value_buf});
  int axis = 2;
  int out = cgen.addTensor({{1, 2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth, on_value, off_value}, {out}}, axis);
  cgen.setInputsAndOutputs({indices, on_value}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<int32_t>({1, 2, 0, 2})
                          .addInput<float>({1})
                          .addOutput<float>({0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_OneHot_OffValueToNotZero)
{
  CircleGen cgen;
  std::vector<int32_t> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  int indices = cgen.addTensor({{1, 2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int on_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int off_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int axis = 2;
  int out = cgen.addTensor({{1, 2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth, on_value, off_value}, {out}}, axis);
  cgen.setInputsAndOutputs({indices, on_value, off_value}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<int32_t>({1, 2, 0, 2})
                          .addInput<float>({1})
                          .addInput<float>({-1})
                          .addOutput<float>({-1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_OneHot_IndicesValueToNeg_OffValueToConst)
{
  CircleGen cgen;
  std::vector<int32_t> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  std::vector<float> off_value_data{0};
  uint32_t off_value_buf = cgen.addBuffer(off_value_data);
  int indices = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int on_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int off_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, off_value_buf});
  int axis = 2;
  int out = cgen.addTensor({{2, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth, on_value, off_value}, {out}}, axis);
  cgen.setInputsAndOutputs({indices, on_value}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<int32_t>({1, 2, 0, -1})
                          .addInput<float>({1})
                          .addOutput<float>({0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_OneHot_IndicesValueToNeg_OffValueToVar)
{
  CircleGen cgen;
  std::vector<int32_t> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  int indices = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int on_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int off_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int axis = 2;
  int out = cgen.addTensor({{2, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth, on_value, off_value}, {out}}, axis);
  cgen.setInputsAndOutputs({indices, on_value, off_value}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<int32_t>({1, 2, 0, -1})
                          .addInput<float>({1})
                          .addInput<float>({0})
                          .addOutput<float>({0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_OneHot_OneOperand)
{
  CircleGen cgen;
  int indices = cgen.addTensor({{1, 2, 2}, circle::TensorType::TensorType_INT32});
  int axis = 2;
  int out = cgen.addTensor({{1, 2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices}, {out}}, axis);
  cgen.setInputsAndOutputs({indices}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_OneHot_TwoOperands)
{
  CircleGen cgen;
  std::vector<int> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  int indices = cgen.addTensor({{1, 2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int axis = 2;
  int out = cgen.addTensor({{1, 2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth}, {out}}, axis);
  cgen.setInputsAndOutputs({indices}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_OneHot_ThreeOperands)
{
  CircleGen cgen;
  std::vector<int> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  int indices = cgen.addTensor({{1, 2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int on_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int axis = 2;
  int out = cgen.addTensor({{1, 2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth, on_value}, {out}}, axis);
  cgen.setInputsAndOutputs({indices, on_value}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_OneHot_InvalidAxis)
{
  CircleGen cgen;
  std::vector<int> depth_data{3};
  uint32_t depth_buf = cgen.addBuffer(depth_data);
  int indices = cgen.addTensor({{1, 2, 2}, circle::TensorType::TensorType_INT32});
  int depth = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, depth_buf});
  int on_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int off_value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int axis = 4;
  int out = cgen.addTensor({{1, 2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorOneHot({{indices, depth, on_value, off_value}, {out}}, axis);
  cgen.setInputsAndOutputs({indices, on_value, off_value}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}
